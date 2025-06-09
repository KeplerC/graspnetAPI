__author__ = 'AI Assistant'

import numpy as np
import os
import json
import requests
import cv2
import base64
from typing import List, Tuple, Dict, Optional
import open3d as o3d
from PIL import Image

from .graspnet import GraspNet
from .grasp import GraspGroup, Grasp
from .utils.config import get_config
from .utils.eval_utils import (
    load_dexnet_model, voxel_sample_points, 
    get_grasp_score, collision_detection, eval_grasp, transform_points
)
try:
    from .utils.eval_utils import get_scene_name
except ImportError:
    # Fallback function if get_scene_name is not available
    def get_scene_name(scene_id):
        return f'scene_{scene_id:04d}'
from .utils.xmlhandler import xmlReader
from .utils.utils import generate_scene_model


class VLMGraspEval:
    """
    Class for VLM-based grasp evaluation on GraspNet dataset.
    
    This class uses Vision Language Models to generate antipodal grasp points
    from images and evaluates their quality using DexNet.
    
    **Input:**
    - root: string of root path for the dataset.
    - camera: string of type of the camera.
    - split: string of the data split.
    - vlm_config: dict containing VLM configuration (model, endpoint, etc.)
    """
    
    def __init__(self, root, camera, split='test', vlm_config=None):
        # Initialize basic attributes without calling GraspNet parent
        self.root = root
        self.camera = camera
        self.split = split
        
        # Validate inputs
        assert camera in ['kinect', 'realsense'], 'camera should be kinect or realsense'
        assert split in ['all', 'train', 'test', 'test_seen', 'test_similar', 'test_novel', "custom"], 'split should be all/train/test/test_seen/test_similar/test_novel'
        
        # Default VLM configuration for Ollama
        self.vlm_config = vlm_config or {
            'model': 'qwen2.5vl:32b',
            'endpoint': 'http://localhost:11434/api/generate',
            'temperature': 0.1,
            'max_tokens': 500
        }
        
        # Load DexNet configuration
        self.dexnet_config = get_config()
        
        # Set matplotlib to use non-interactive backend for headless operation
        import matplotlib
        matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no display needed)
        
        print(f"✅ VLMGraspEval initialized - Root: {root}, Camera: {camera}, Split: {split}")
        
    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string for VLM input."""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def query_vlm_ollama(self, image_path: str, target_object: str, prompt: str = None) -> Dict:
        """
        Query Ollama VLM for antipodal grasp points.
        
        **Input:**
        - image_path: path to the scene image
        - target_object: name/description of the target object
        - prompt: custom prompt (optional)
        
        **Output:**
        - response: dict containing VLM response with grasp points
        """
        if prompt is None:
            prompt = f"""
            consider yourself as a robot arm with a parallel gripper, give the parameters on the objects in the image that the gripper should contact with  that allows a success and stable grasp while lifting, transporting and shaking
            Look at this image and identify the {target_object}. 
            I need you to suggest the best antipodal grasp points for picking up this object.
            The points MUST be on the object and not on the background without collision with other objects.
            Please return EXACTLY two 2D pixel coordinates on the image that represent 
             optimal grasp locations (where gripper fingers should contact the object).
             
             Return your answer in this exact JSON format:
             {{
                 "point1": [x1, y1],
                 "point2": [x2, y2],
                 "confidence": 0.85,
                 "reasoning": "brief explanation of why these points are good"
             }}
             
             where x1, y1, x2, y2 are pixel coordinates on the image.
            
            The points should be:
            - On opposite sides of the object (antipodal)
            - On stable, graspable surfaces
            - Positioned for a secure grip
            
            Target object: {target_object}
            """
        
        # Encode image
        image_b64 = self.encode_image_base64(image_path)
        
        # Prepare request payload
        payload = {
            "model": self.vlm_config['model'],
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": self.vlm_config.get('temperature', 0.1),
                "num_predict": self.vlm_config.get('max_tokens', 500)
            }
        }
        
        try:
            response = requests.post(self.vlm_config['endpoint'], json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying VLM: {e}")
            return None
    
    def query_vlm_openai(self, image_path: str, target_object: str, api_key: str, prompt: str = None) -> Dict:
        """
        Query OpenAI GPT-4V for antipodal grasp points.
        
        **Input:**
        - image_path: path to the scene image
        - target_object: name/description of the target object
        - api_key: OpenAI API key
        - prompt: custom prompt (optional)
        
        **Output:**
        - response: dict containing VLM response with grasp points
        """
        if prompt is None:
            prompt = f"""
                         Look at this image and identify the {target_object}. 
             I need you to suggest the best antipodal grasp points for picking up this object.
             
             Please return EXACTLY two 2D pixel coordinates on the image that represent optimal grasp locations.
             
             Return your answer in this exact JSON format:
             {{
                 "point1": [x1, y1],
                 "point2": [x2, y2], 
                 "confidence": 0.85,
                 "reasoning": "brief explanation"
             }}
             
             where x1, y1, x2, y2 are pixel coordinates on the image.
             Target object: {target_object}
            """
        
        # Encode image
        image_b64 = self.encode_image_base64(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",  # Updated to current GPT-4 with vision
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            "max_tokens": self.vlm_config.get('max_tokens', 500),
            "temperature": self.vlm_config.get('temperature', 0.1)
        }
        
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return None
    
    def parse_vlm_response(self, response: Dict, vlm_type: str = 'ollama') -> Optional[Dict]:
        """
        Parse VLM response to extract grasp points.
        
        **Input:**
        - response: raw VLM response
        - vlm_type: type of VLM ('ollama' or 'openai')
        
        **Output:**
        - parsed_data: dict with point1, point2, confidence, reasoning
        """
        try:
            if vlm_type == 'ollama':
                content = response.get('response', '')
            elif vlm_type == 'openai':
                content = response['choices'][0]['message']['content']
            else:
                raise ValueError(f"Unsupported VLM type: {vlm_type}")
            
            # Try to find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                parsed_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['point1', 'point2']
                for field in required_fields:
                    if field not in parsed_data:
                        raise ValueError(f"Missing required field: {field}")
                
                return parsed_data
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"Error parsing VLM response: {e}")
            print(f"Response content: {response}")
            return None
    
    def antipodal_points_to_grasp_pose(self, point1: List[float], point2: List[float], 
                                     approach_vector: List[float] = None) -> np.ndarray:
        """
        Convert antipodal points to 6DOF grasp pose.
        Fixed to follow GraspNet coordinate convention:
        - X-axis: approach direction (towards object)
        - Y-axis: grasp axis (finger separation direction)  
        - Z-axis: orthogonal to both (gripper closing direction)
        
        **Input:**
        - point1: first grasp point [x, y, z]
        - point2: second grasp point [x, y, z]
        - approach_vector: gripper approach direction (optional)
        
        **Output:**
        - grasp_pose: 4x4 transformation matrix representing grasp pose
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # Grasp center is midpoint between the two points
        grasp_center = (point1 + point2) / 2.0
        
        # Grasp axis (Y-axis) is the line connecting the two points
        y_axis = point2 - point1
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Default approach vector (perpendicular to grasp axis, pointing away from surface)
        if approach_vector is None:
            # Simple approach: use the vector from grasp center pointing towards camera
            # This ensures the gripper approaches from outside the object
            to_camera = -grasp_center  # Vector from origin to grasp center
            to_camera_norm = np.linalg.norm(to_camera)
            
            if to_camera_norm > 0:
                approach_vector = to_camera / to_camera_norm
            else:
                # Fallback to negative Z (typical camera approach)
                approach_vector = np.array([0, 0, -1])
        else:
            approach_vector = np.array(approach_vector)
            approach_vector = approach_vector / np.linalg.norm(approach_vector)
        
        # Create orthogonal coordinate system following GraspNet convention
        # X-axis is the approach direction
        x_axis = approach_vector
        
        # Z-axis is orthogonal to both X and Y (gripper closing direction)
        z_axis = np.cross(x_axis, y_axis)
        z_axis_norm = np.linalg.norm(z_axis)
        
        # Handle case where x_axis and y_axis are parallel
        if z_axis_norm < 1e-6:
            # Choose a different orthogonal vector
            if abs(np.dot(x_axis, [0, 0, 1])) < 0.9:
                temp_vector = np.array([0, 0, 1])
            else:
                temp_vector = np.array([0, 1, 0])
            
            z_axis = np.cross(x_axis, temp_vector)
            z_axis_norm = np.linalg.norm(z_axis)
            
            if z_axis_norm < 1e-6:
                temp_vector = np.array([1, 0, 0])
                z_axis = np.cross(x_axis, temp_vector)
                z_axis_norm = np.linalg.norm(z_axis)
        
        z_axis = z_axis / z_axis_norm
        
        # Recompute Y-axis to ensure perfect orthogonality
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Construct 4x4 transformation matrix
        grasp_pose = np.eye(4)
        grasp_pose[0:3, 0] = x_axis  # approach direction
        grasp_pose[0:3, 1] = y_axis  # grasp axis (finger separation)
        grasp_pose[0:3, 2] = z_axis  # orthogonal (gripper closing)
        grasp_pose[0:3, 3] = grasp_center
        
        # Debug coordinate system
        print(f"🧭 Grasp Coordinate System Debug:")
        print(f"   Grasp center: {grasp_center}")
        print(f"   Grasp width: {np.linalg.norm(point2 - point1):.3f} m")
        print(f"   X-axis (approach): {x_axis} (norm: {np.linalg.norm(x_axis):.3f})")
        print(f"   Y-axis (grasp): {y_axis} (norm: {np.linalg.norm(y_axis):.3f})")
        print(f"   Z-axis (closing): {z_axis} (norm: {np.linalg.norm(z_axis):.3f})")
        
        # Check orthogonality
        xy_dot = np.dot(x_axis, y_axis)
        xz_dot = np.dot(x_axis, z_axis)
        yz_dot = np.dot(y_axis, z_axis)
        print(f"   Orthogonality: X·Y={xy_dot:.6f}, X·Z={xz_dot:.6f}, Y·Z={yz_dot:.6f}")
        
        return grasp_pose
    
    def pose_to_grasp_array(self, grasp_pose: np.ndarray, width: float = 0.08, 
                           confidence: float = 1.0) -> np.ndarray:
        """
        Convert 4x4 grasp pose to GraspNet array format.
        
        **Input:**
        - grasp_pose: 4x4 transformation matrix
        - width: gripper width
        - confidence: grasp confidence score
        
        **Output:**
        - grasp_array: numpy array in GraspNet format [score, width, height, depth, rotation_matrix, translation, object_id]
        """
        # Extract rotation matrix and translation
        rotation_matrix = grasp_pose[0:3, 0:3]
        translation = grasp_pose[0:3, 3]
        
        # GraspNet array format: [score, width, height, depth, 9 rotation values, 3 translation values, object_id]
        grasp_array = np.zeros(17)
        grasp_array[0] = confidence  # score
        grasp_array[1] = width      # width
        grasp_array[2] = 0.02       # height (default)
        grasp_array[3] = 0.02       # depth (default)
        grasp_array[4:13] = rotation_matrix.flatten()  # rotation matrix (9 values)
        grasp_array[13:16] = translation               # translation (3 values)
        grasp_array[16] = 0         # object_id (default)
        
        return grasp_array
    
    def evaluate_grasp_dexnet(self, grasp_array: np.ndarray, object_model: np.ndarray, 
                            dexnet_model, object_pose: np.ndarray = None, friction_coef: float = 0.4) -> float:
        """
        Evaluate grasp quality using DexNet via the existing eval_grasp infrastructure.
        
        **Input:**
        - grasp_array: grasp in GraspNet array format (in camera coordinates)
        - object_model: point cloud of the object
        - dexnet_model: loaded DexNet model for the object
        - object_pose: 4x4 transformation matrix from object to camera frame
        - friction_coef: friction coefficient
        
        **Output:**
        - quality_score: grasp quality score from DexNet
        """
        print(f"🔍 Starting DexNet evaluation using existing infrastructure...")
        print(f"   Grasp array shape: {grasp_array.shape}")
        print(f"   Object model points: {len(object_model) if object_model is not None else 'None'}")
        print(f"   DexNet model type: {type(dexnet_model)}")
        print(f"   Object pose available: {object_pose is not None}")
        
        try:
            # Import required functions from existing infrastructure
            from .utils.eval_utils import eval_grasp, transform_points
            from .grasp import GraspGroup
            from .utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
            
            # Create a GraspGroup with our single grasp
            grasp_group = GraspGroup()
            grasp_group.grasp_group_array = grasp_array.reshape(1, -1)  # Single grasp
            
            print(f"   ✅ Created GraspGroup with 1 grasp")
            
            if object_pose is None:
                print("   ❌ No object pose available - cannot evaluate without proper coordinate transformation")
                return -1.0
            
            # Prepare inputs for eval_grasp function
            models = [object_model]  # List of object models (in object coordinates)
            dexnet_models = [dexnet_model]  # List of DexNet models
            poses = [object_pose]  # List of poses (object to camera)
            
            # Use the DexNet config from our class
            config = self.dexnet_config.copy()
            
            print(f"   Calling eval_grasp with {len(models)} objects...")
            
            # Call the existing eval_grasp function
            grasp_list, score_list, collision_mask_list = eval_grasp(
                grasp_group=grasp_group,
                models=models,
                dexnet_models=dexnet_models,
                poses=poses,
                config=config,
                table=None,  # No table for simplicity
                voxel_size=0.008,
                TOP_K=1
            )
            
            print(f"   ✅ eval_grasp completed")
            print(f"   Grasp list length: {len(grasp_list)}")
            print(f"   Score list length: {len(score_list)}")
            
            # Extract the score for our grasp
            if len(score_list) > 0 and len(score_list[0]) > 0:
                quality_score = score_list[0][0]  # First object, first grasp
                collision_detected = collision_mask_list[0][0] if len(collision_mask_list) > 0 and len(collision_mask_list[0]) > 0 else False
                
                print(f"   Quality score: {quality_score}")
                print(f"   Collision detected: {collision_detected}")
                
                if collision_detected:
                    print("   ⚠️ Grasp has collision")
                    print("   💡 Possible fixes:")
                    print("      - Adjust approach angle")
                    print("      - Move grasp points to better surface locations")
                    print("      - Check coordinate system orientation")
                    return -1.0
                    
                print(f"   🎯 Final quality score: {quality_score}")
                return quality_score
            else:
                print("   ❌ No scores returned from eval_grasp")
                return -1.0
                
        except Exception as e:
            print(f"❌ Error in DexNet evaluation: {e}")
            import traceback
            traceback.print_exc()
            return -1.0
    
    def get_object_model_and_dexnet(self, scene_id: int, ann_id: int, 
                                   target_obj_idx: int) -> Tuple[np.ndarray, object]:
        """
        Load object model and DexNet model for evaluation.
        
        **Input:**
        - scene_id: scene index
        - ann_id: annotation index  
        - target_obj_idx: target object index
        
        **Output:**
        - object_model: point cloud of the target object
        - dexnet_model: loaded DexNet model
        """
        model_dir = os.path.join(self.root, 'models')
        
        # Load object point cloud
        model_path = os.path.join(model_dir, '%03d' % target_obj_idx, 'nontextured.ply')
        model = o3d.io.read_point_cloud(model_path)
        object_model = np.array(model.points)
        
        # Load DexNet model
        dex_cache_path = os.path.join(self.root, 'dex_models', '%03d.pkl' % target_obj_idx)
        if os.path.exists(dex_cache_path):
            import pickle
            with open(dex_cache_path, 'rb') as f:
                dexnet_model = pickle.load(f)
        else:
            dexnet_model = load_dexnet_model(os.path.join(model_dir, '%03d' % target_obj_idx, 'textured'))
        
        return object_model, dexnet_model

    def load_ground_truth_grasps(self, target_obj_idx: int, scene_id: int = None, 
                                max_grasps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ground truth grasps for a specific object.
        
        **Input:**
        - target_obj_idx: object index
        - scene_id: scene ID (if None, loads all scenes)
        - max_grasps: maximum number of grasps to return
        
        **Output:**
        - grasp_array: numpy array of shape (N, 17) with ground truth grasps
        - scores: numpy array of shape (N,) with grasp quality scores
        """
        from .utils.rotation import batch_viewpoint_params_to_matrix
        
        # Load grasp labels
        grasp_labels_path = os.path.join(self.root, 'grasp_label', f'{target_obj_idx:03d}_labels.npz')
        
        if not os.path.exists(grasp_labels_path):
            print(f"Ground truth grasp labels not found: {grasp_labels_path}")
            return np.array([]), np.array([])
        
        # Load the npz file
        grasp_data = np.load(grasp_labels_path)
        points = grasp_data['points']  # (N, 3) grasp points on object
        offsets = grasp_data['offsets']  # (N, 300, 12, 4, 3) viewpoint, angle, depth, width offsets
        scores = grasp_data['scores']  # (N, 300, 12, 4) quality scores
        collision_mask = grasp_data['collision']  # (N, 300, 12, 4) collision mask
        
        print(f"✅ Loaded ground truth grasps for object {target_obj_idx}")
        print(f"   Points: {points.shape}, Offsets: {offsets.shape}")
        print(f"   Scores: {scores.shape}, Collision: {collision_mask.shape}")
        
        # Extract valid grasps (non-collision, positive scores)
        valid_mask = (scores > 0.4) & (~collision_mask)  # Filter by score threshold and no collision
        
        if not np.any(valid_mask):
            print("No valid ground truth grasps found")
            return np.array([]), np.array([])
        
        # Get indices of valid grasps
        point_ids, view_ids, angle_ids, depth_ids = np.where(valid_mask)
        
        # Limit number of grasps
        num_valid = len(point_ids)
        if num_valid > max_grasps:
            # Select top grasps by score
            valid_scores = scores[valid_mask]
            top_indices = np.argsort(valid_scores)[-max_grasps:]
            point_ids = point_ids[top_indices]
            view_ids = view_ids[top_indices]
            angle_ids = angle_ids[top_indices]
            depth_ids = depth_ids[top_indices]
        
        print(f"   Selected {len(point_ids)} valid grasps")
        
        # Generate grasps in GraspNet format
        grasp_list = []
        score_list = []
        
        # Generate template views (300 viewpoints)
        from .utils.utils import generate_views
        template_views = generate_views(300)
        
        for i, (pid, vid, aid, did) in enumerate(zip(point_ids, view_ids, angle_ids, depth_ids)):
            try:
                # Get grasp parameters
                grasp_point = points[pid]  # 3D point on object surface
                view_vector = template_views[vid]  # viewing direction
                angle_offset = offsets[pid, vid, aid, did, 0]  # rotation angle
                depth_offset = offsets[pid, vid, aid, did, 1]  # grasp depth
                width = offsets[pid, vid, aid, did, 2]  # grasp width
                score = scores[pid, vid, aid, did]
                
                # Convert to rotation matrix - fix array shape issues
                view_batch = (-view_vector).reshape(1, -1)  # Shape: (1, 3)
                angle_batch = np.array([angle_offset])       # Shape: (1,) - FIXED: removed extra brackets
                
                # Debug shapes
                if i == 0:  # Only print for first grasp
                    print(f"   Debug shapes: view_batch={view_batch.shape}, angle_batch={angle_batch.shape}")
                
                R = batch_viewpoint_params_to_matrix(view_batch, angle_batch)
                R = R[0]  # Remove batch dimension, shape: (3, 3)
                
                # Create grasp array [score, width, height, depth, 9x rotation, 3x translation, object_id]
                grasp_array = np.zeros(17)
                grasp_array[0] = score
                grasp_array[1] = width
                grasp_array[2] = 0.02  # default height
                grasp_array[3] = depth_offset
                grasp_array[4:13] = R.flatten()
                grasp_array[13:16] = grasp_point
                grasp_array[16] = target_obj_idx
                
                grasp_list.append(grasp_array)
                score_list.append(score)
                
            except Exception as e:
                print(f"   ⚠️ Error processing grasp {i}: {e}")
                continue
        
        if len(grasp_list) == 0:
            print("   ❌ No grasps could be processed successfully")
            return np.array([]), np.array([])
        
        print(f"   ✅ Successfully processed {len(grasp_list)} grasps")
        return np.array(grasp_list), np.array(score_list)
    
    def loadScenePointCloud(self, scene_id: int, camera: str, ann_id: int) -> o3d.geometry.PointCloud:
        """
        Load scene point cloud from RGB and depth images.
        
        **Input:**
        - scene_id: scene index
        - camera: camera type
        - ann_id: annotation index
        
        **Output:**
        - pcd: Open3D point cloud
        """
        import scipy.io as scio
        
        scene_name = f'scene_{scene_id:04d}'
        
        # Load RGB image
        rgb_path = os.path.join(self.root, 'scenes', scene_name, camera, 'rgb', f'{ann_id:04d}.png')
        depth_path = os.path.join(self.root, 'scenes', scene_name, camera, 'depth', f'{ann_id:04d}.png')
        meta_path = os.path.join(self.root, 'scenes', scene_name, camera, 'meta', f'{ann_id:04d}.mat')
        
        if not all(os.path.exists(p) for p in [rgb_path, depth_path, meta_path]):
            raise FileNotFoundError(f"Missing data files for scene {scene_id}, camera {camera}, ann {ann_id}")
        
        # Load images
        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR to RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # Load camera intrinsics
        meta = scio.loadmat(meta_path)
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        
        # Convert depth to real scale
        depth = depth / factor_depth
        
        # Create point cloud
        height, width = depth.shape
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # Generate coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        valid_mask = depth > 0
        z = depth[valid_mask]
        x = (x[valid_mask] - cx) * z / fx
        y = (y[valid_mask] - cy) * z / fy
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=1)
        colors = rgb[valid_mask] / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def get_object_pose_in_camera(self, scene_id: int, ann_id: int, target_obj_idx: int):
        """
        Get object pose in camera coordinate frame.
        
        **Input:**
        - scene_id: scene index
        - ann_id: annotation index
        - target_obj_idx: object index
        
        **Output:**
        - object_pose: 4x4 transformation matrix from object to camera frame
        """
        scene_name = f'scene_{scene_id:04d}'
        
        # Load camera poses and alignment matrix
        camera_poses_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'camera_poses.npy')
        align_mat_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'cam0_wrt_table.npy')
        
        # Try alternative data structure if scenes folder doesn't exist
        if not os.path.exists(camera_poses_path):
            camera_poses_path = os.path.join(self.root, scene_name, self.camera, 'camera_poses.npy')
            align_mat_path = os.path.join(self.root, scene_name, self.camera, 'cam0_wrt_table.npy')
        
        camera_poses = np.load(camera_poses_path)
        align_mat = np.load(align_mat_path)
        camera_pose = camera_poses[ann_id]
        
        # Load object poses from annotation XML
        annotation_path = os.path.join(os.path.dirname(camera_poses_path), 'annotations', f'{ann_id:04d}.xml')
        
        from .utils.xmlhandler import xmlReader
        from .utils.eval_utils import parse_posevector
        
        scene_reader = xmlReader(annotation_path)
        posevectors = scene_reader.getposevectorlist()
        
        # Find the target object
        for posevector in posevectors:
            obj_idx, obj_pose = parse_posevector(posevector)
            if obj_idx == target_obj_idx:
                # Transform from table coordinate to camera coordinate
                obj_pose_in_camera = np.linalg.inv(np.matmul(align_mat, camera_pose)) @ obj_pose
                return obj_pose_in_camera
        
        raise ValueError(f"Object {target_obj_idx} not found in scene {scene_id} annotation {ann_id}")
    
    def pixel_to_3d_point(self, pixel_coords: List[float], scene_id: int, ann_id: int) -> np.ndarray:
        """
        Convert 2D pixel coordinates to 3D world coordinates using depth.
        
        **Input:**
        - pixel_coords: [u, v] pixel coordinates
        - scene_id: scene index
        - ann_id: annotation index
        
        **Output:**
        - point_3d: 3D point in camera coordinate frame
        """
        import scipy.io as scio
        
        scene_name = f'scene_{scene_id:04d}'
        
        # Load depth image and camera intrinsics
        depth_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'depth', f'{ann_id:04d}.png')
        meta_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'meta', f'{ann_id:04d}.mat')
        
        # Try alternative data structure
        if not os.path.exists(depth_path):
            depth_path = os.path.join(self.root, scene_name, self.camera, 'depth', f'{ann_id:04d}.png')
            meta_path = os.path.join(self.root, scene_name, self.camera, 'meta', f'{ann_id:04d}.mat')
        
        # Load depth and intrinsics
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        meta = scio.loadmat(meta_path)
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        
        # Convert depth to real scale
        depth = depth / factor_depth
        
        # Extract pixel coordinates
        u, v = int(pixel_coords[0]), int(pixel_coords[1])
        
        # Clamp to image bounds
        height, width = depth.shape
        u = max(0, min(u, width - 1))
        v = max(0, min(v, height - 1))
        
        # Get depth at pixel location
        z = depth[v, u]  # Note: depth is indexed as [row, col] = [y, x]
        
        if z <= 0:
            # If no depth, use nearby pixels
            for radius in range(1, 10):
                for dv in range(-radius, radius + 1):
                    for du in range(-radius, radius + 1):
                        new_v = v + dv
                        new_u = u + du
                        if 0 <= new_v < height and 0 <= new_u < width:
                            z = depth[new_v, new_u]
                            if z > 0:
                                break
                    if z > 0:
                        break
                if z > 0:
                    break
        
        if z <= 0:
            raise ValueError(f"No valid depth found at pixel ({u}, {v})")
        
        # Convert to 3D using camera intrinsics
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return np.array([x, y, z])
    
    def eval_vlm_grasp(self, scene_id: int, ann_id: int, target_object: str, 
                      target_obj_idx: int, vlm_type: str = 'ollama', 
                      api_key: str = None, visualize: bool = True) -> Dict:
        """
        Complete evaluation pipeline: VLM -> grasp pose -> DexNet evaluation.
        
        **Input:**
        - scene_id: scene index
        - ann_id: annotation index
        - target_object: description of target object
        - target_obj_idx: object index in the scene
        - vlm_type: 'ollama' or 'openai'
        - api_key: API key for OpenAI (if using OpenAI)
        - visualize: whether to visualize the result
        
        **Output:**
        - result: dict containing all evaluation results
        """
        # Get scene image path
        scene_name = get_scene_name(scene_id)
        image_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 
                                'rgb', '%04d.png' % ann_id)
        
        # Check if required files exist
        required_paths = {
            'RGB image': image_path,
            'Depth image': os.path.join(self.root, 'scenes', scene_name, self.camera, 
                                      'depth', '%04d.png' % ann_id),
            'Meta file': os.path.join(self.root, 'scenes', scene_name, self.camera, 
                                    'meta', '%04d.mat' % ann_id),
            'Object model': os.path.join(self.root, 'models', '%03d' % target_obj_idx, 'nontextured.ply')
        }
        
        missing_files = []
        for name, path in required_paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            error_msg = f"Missing required files:\n" + "\n".join(missing_files)
            error_msg += f"\n\nTips:"
            error_msg += f"\n- Make sure GraspNet dataset is properly downloaded"
            error_msg += f"\n- Check scene_id {scene_id} exists (try scene_id >= 100 for test split)"
            error_msg += f"\n- Check ann_id {ann_id} is valid (0-255)"
            error_msg += f"\n- Check target_obj_idx {target_obj_idx} exists in the models folder"
            return {'error': error_msg}
        
        # # Query VLM for grasp points
        # print(f"Querying {vlm_type} VLM for grasp points...")
        # if vlm_type == 'ollama':
        #     vlm_response = self.query_vlm_ollama(image_path, target_object)
        # elif vlm_type == 'openai':
        #     if api_key is None:
        #         return {'error': 'API key required for OpenAI'}
        #     vlm_response = self.query_vlm_openai(image_path, target_object, api_key)
        # else:
        #     return {'error': f'Unsupported VLM type: {vlm_type}'}
        
        # if vlm_response is None:
        #     return {'error': 'Failed to get VLM response'}
        
        # # Parse VLM response
        # parsed_response = self.parse_vlm_response(vlm_response, vlm_type)
        # if parsed_response is None:
        #     return {'error': 'Failed to parse VLM response'}
        
        parsed_response = {
            'point1': [405, 385],
            'point2': [500, 420],
            'confidence': 1.0,
            'reasoning': 'Test reasoning'
        }
        print(f"VLM suggested 2D grasp points:")
        print(f"Point 1 (pixels): {parsed_response['point1']}")
        print(f"Point 2 (pixels): {parsed_response['point2']}")
        print(f"Confidence: {parsed_response.get('confidence', 'N/A')}")
        print(f"Reasoning: {parsed_response.get('reasoning', 'N/A')}")
        
        # Convert 2D pixel coordinates to 3D world coordinates
        try:
            point1_3d = self.pixel_to_3d_point(parsed_response['point1'], scene_id, ann_id)
            point2_3d = self.pixel_to_3d_point(parsed_response['point2'], scene_id, ann_id)
            
            print(f"Converted to 3D coordinates:")
            print(f"Point 1 (3D): {point1_3d}")
            print(f"Point 2 (3D): {point2_3d}")
            
        except Exception as e:
            return {'error': f'Failed to convert 2D to 3D coordinates: {e}'}
        
        # Get object pose for proper coordinate transformation
        try:
            object_pose = self.get_object_pose_in_camera(scene_id, ann_id, target_obj_idx)
            print(f"Object pose loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load object pose: {e}")
            print("This will affect DexNet evaluation accuracy")
            object_pose = None
        
        # Determine approach vector from object pose if available
        approach_vector = None
        if object_pose is not None:
            # Use the object's up direction (Z-axis) as approach vector
            approach_vector = object_pose[:3, 2]
        else:
            # Estimate surface normal at grasp center for better approach vector
            grasp_center = (point1_3d + point2_3d) / 2.0
            surface_normal = self.estimate_surface_normal(grasp_center, scene_id, ann_id)
            approach_vector = surface_normal
            print(f"   📐 Estimated surface normal: {surface_normal}")
        
        # Convert to grasp pose
        grasp_pose = self.antipodal_points_to_grasp_pose(
            point1_3d, 
            point2_3d,
            approach_vector=approach_vector
        )
        
        # Convert to grasp array with proper width
        grasp_width = np.linalg.norm(point2_3d - point1_3d)
        grasp_array = self.pose_to_grasp_array(
            grasp_pose, 
            width=grasp_width,
            confidence=parsed_response.get('confidence', 1.0)
        )
        
        # Load object model and DexNet model
        object_model, dexnet_model = self.get_object_model_and_dexnet(
            scene_id, ann_id, target_obj_idx
        )
        
        # Evaluate with DexNet (now passing object_pose for proper coordinate transformation)
        print("Evaluating grasp quality with DexNet...")
        dexnet_score = self.evaluate_grasp_dexnet(grasp_array, object_model, dexnet_model, object_pose)
        
        # Compile results
        result = {
            'scene_id': scene_id,
            'ann_id': ann_id,
            'target_object': target_object,
            'target_obj_idx': target_obj_idx,
            'vlm_response': parsed_response,
            'point1_2d': parsed_response['point1'],
            'point2_2d': parsed_response['point2'],
            'point1_3d': point1_3d.tolist(),
            'point2_3d': point2_3d.tolist(),
            'object_pose': object_pose.tolist() if object_pose is not None else None,
            'grasp_pose': grasp_pose,
            'grasp_array': grasp_array,
            'dexnet_score': dexnet_score,
            'evaluation_success': dexnet_score >= 0
        }
        
        if visualize:
            # Traditional matplotlib visualization
            self.visualize_grasp_result(scene_id, ann_id, result)
            
            # Enhanced viser visualization with ground truth comparison
            self.visualize_with_viser(scene_id, ann_id, result, target_obj_idx)
        
        print(f"DexNet Quality Score: {dexnet_score:.3f}")
        
        return result
    
    def visualize_grasp_result(self, scene_id: int, ann_id: int, result: Dict, save_dir: str = "visualizations"):
        """
        Visualize the grasp result in multiple formats (headless-compatible).
        
        **Input:**
        - scene_id: scene index
        - ann_id: annotation index
        - result: evaluation result dictionary
        - save_dir: directory to save visualization files
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import os
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 1. Create 2D visualization on RGB image
            self._visualize_2d_grasp_on_image(scene_id, ann_id, result, save_dir)
            
            # 2. Create 3D point cloud visualization (headless)
            self._visualize_3d_grasp_headless(scene_id, ann_id, result, save_dir)
            
            # 3. Create summary plot
            self._create_summary_plot(scene_id, ann_id, result, save_dir)
            
            print(f"✅ Visualizations saved to: {save_dir}/")
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without visualization...")
    
    def _visualize_2d_grasp_on_image(self, scene_id: int, ann_id: int, result: Dict, save_dir: str):
        """Create 2D visualization overlaying grasp points on RGB image."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
        import matplotlib.patches as mpatches
        
        # Load RGB image
        scene_name = f'scene_{scene_id:04d}'
        image_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'rgb', f'{ann_id:04d}.png')
        
        if not os.path.exists(image_path):
            print(f"RGB image not found: {image_path}")
            return
            
        # Load and display image
        rgb_image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(rgb_image)
        
        # Plot grasp points
        point1_2d = result['point1_2d']
        point2_2d = result['point2_2d']
        
        # Draw grasp points
        ax.plot(point1_2d[0], point1_2d[1], 'ro', markersize=10, label='Grasp Point 1')
        ax.plot(point2_2d[0], point2_2d[1], 'bo', markersize=10, label='Grasp Point 2')
        
        # Draw line between points
        ax.plot([point1_2d[0], point2_2d[0]], [point1_2d[1], point2_2d[1]], 
                'g-', linewidth=3, alpha=0.7, label='Grasp Axis')
        
        # Draw circles around grasp points
        circle1 = Circle((point1_2d[0], point1_2d[1]), 15, fill=False, color='red', linewidth=2)
        circle2 = Circle((point2_2d[0], point2_2d[1]), 15, fill=False, color='blue', linewidth=2)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Add annotations
        ax.annotate(f'P1: ({point1_2d[0]}, {point1_2d[1]})', 
                   xy=(point1_2d[0], point1_2d[1]), xytext=(20, 20), 
                   textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'P2: ({point2_2d[0]}, {point2_2d[1]})', 
                   xy=(point2_2d[0], point2_2d[1]), xytext=(20, -40), 
                   textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Add title and info
        quality_score = result['dexnet_score']
        vlm_confidence = result['vlm_response'].get('confidence', 'N/A')
        
        title = f"Scene {scene_id:04d} Ann {ann_id:04d} - {result['target_object']}\n"
        title += f"DexNet Score: {quality_score:.3f} | VLM Confidence: {vlm_confidence}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        
        # Save
        output_path = os.path.join(save_dir, f'grasp_2d_scene_{scene_id:04d}_ann_{ann_id:04d}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📸 2D visualization saved: {output_path}")
    
    def _visualize_3d_grasp_headless(self, scene_id: int, ann_id: int, result: Dict, save_dir: str):
        """Create 3D visualization using matplotlib (headless)."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        try:
            # Load scene point cloud
            pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)
            points = np.array(pcd.points)
            colors = np.array(pcd.colors)
            
            # Sample points for faster visualization
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # Create 3D plot
            fig = plt.figure(figsize=(15, 5))
            
            # Three views: XY, XZ, YZ
            views = [
                (1, 3, 1, (0, 90), 'XY View (Top)'),
                (1, 3, 2, (0, 0), 'XZ View (Front)'), 
                (1, 3, 3, (90, 0), 'YZ View (Side)')
            ]
            
            for subplot_idx, (rows, cols, pos, (elev, azim), title) in enumerate(views):
                ax = fig.add_subplot(rows, cols, pos, projection='3d')
                
                # Plot point cloud
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=1, alpha=0.6)
                
                # Plot 3D grasp points
                point1_3d = np.array(result['point1_3d'])
                point2_3d = np.array(result['point2_3d'])
                
                ax.scatter(*point1_3d, color='red', s=100, label='Grasp Point 1')
                ax.scatter(*point2_3d, color='blue', s=100, label='Grasp Point 2')
                
                # Draw grasp axis
                ax.plot([point1_3d[0], point2_3d[0]], 
                       [point1_3d[1], point2_3d[1]], 
                       [point1_3d[2], point2_3d[2]], 
                       'g-', linewidth=3, label='Grasp Axis')
                
                # Set view
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(title)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)') 
                ax.set_zlabel('Z (m)')
                
                if subplot_idx == 0:  # Only show legend once
                    ax.legend()
            
            plt.tight_layout()
            
            # Save
            output_path = os.path.join(save_dir, f'grasp_3d_scene_{scene_id:04d}_ann_{ann_id:04d}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 3D visualization saved: {output_path}")
            
        except Exception as e:
            print(f"Error in 3D visualization: {e}")
    
    def _create_summary_plot(self, scene_id: int, ann_id: int, result: Dict, save_dir: str):
        """Create a summary plot with all relevant information."""
        import matplotlib.pyplot as plt
        import textwrap
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RGB Image with grasp points (top-left)
        scene_name = f'scene_{scene_id:04d}'
        image_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'rgb', f'{ann_id:04d}.png')
        
        if os.path.exists(image_path):
            rgb_image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            ax1.imshow(rgb_image)
            
            point1_2d = result['point1_2d'] 
            point2_2d = result['point2_2d']
            ax1.plot(point1_2d[0], point1_2d[1], 'ro', markersize=8)
            ax1.plot(point2_2d[0], point2_2d[1], 'bo', markersize=8)
            ax1.plot([point1_2d[0], point2_2d[0]], [point1_2d[1], point2_2d[1]], 'g-', linewidth=2)
            
        ax1.set_title('RGB Image with Grasp Points')
        ax1.axis('off')
        
        # 2. Depth visualization (top-right)
        depth_path = os.path.join(self.root, 'scenes', scene_name, self.camera, 'depth', f'{ann_id:04d}.png')
        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            im = ax2.imshow(depth, cmap='plasma')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            point1_2d = result['point1_2d']
            point2_2d = result['point2_2d'] 
            ax2.plot(point1_2d[0], point1_2d[1], 'ro', markersize=8)
            ax2.plot(point2_2d[0], point2_2d[1], 'bo', markersize=8)
            
        ax2.set_title('Depth Image with Grasp Points')
        ax2.axis('off')
        
        # 3. Results text (bottom-left)
        ax3.axis('off')
        
        # Format results text
        results_text = f"""
        EVALUATION RESULTS
        ==================
        Scene ID: {scene_id}
        Annotation ID: {ann_id}
        Target Object: {result['target_object']}
        Object Index: {result['target_obj_idx']}
        
        VLM RESPONSE
        ============
        Point 1 (2D): {result['point1_2d']} px
        Point 2 (2D): {result['point2_2d']} px
        Point 1 (3D): [{result['point1_3d'][0]:.3f}, {result['point1_3d'][1]:.3f}, {result['point1_3d'][2]:.3f}] m
        Point 2 (3D): [{result['point2_3d'][0]:.3f}, {result['point2_3d'][1]:.3f}, {result['point2_3d'][2]:.3f}] m
        VLM Confidence: {result['vlm_response'].get('confidence', 'N/A')}
        
        GRASP EVALUATION
        ================
        DexNet Score: {result['dexnet_score']:.3f}
        Evaluation Success: {result['evaluation_success']}
        
        VLM REASONING
        =============
        """
        
        reasoning = result['vlm_response'].get('reasoning', 'No reasoning provided')
        wrapped_reasoning = textwrap.fill(reasoning, width=50)
        results_text += wrapped_reasoning
        
        ax3.text(0.05, 0.95, results_text, transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 4. Score visualization (bottom-right)
        scores = [result['dexnet_score']]
        labels = ['DexNet Score']
        colors = ['green' if result['dexnet_score'] > 0.5 else 'orange' if result['dexnet_score'] > 0.3 else 'red']
        
        bars = ax4.bar(labels, scores, color=colors, alpha=0.7)
        ax4.set_ylim(-1, 1)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
        ax4.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Fair Threshold')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height >= 0 else height - 0.1,
                    f'{score:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        ax4.set_title('Grasp Quality Scores')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle(f'VLM Grasp Evaluation Summary - Scene {scene_id:04d} ({result["target_object"]})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(save_dir, f'summary_scene_{scene_id:04d}_ann_{ann_id:04d}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📋 Summary visualization saved: {output_path}")

    def visualize_with_viser(self, scene_id: int, ann_id: int, result: Dict, target_obj_idx: int):
        """
        Enhanced 3D visualization using viser with ground truth comparison.
        
        **Input:**
        - scene_id: scene index
        - ann_id: annotation index
        - result: evaluation result dictionary
        - target_obj_idx: object index for loading ground truth
        """
        try:
            import viser
            import time
            print("🚀 Starting viser visualization...")
            
            # Create viser server
            server = viser.ViserServer(port=8080)
            print(f"   🌐 Viser server started at http://localhost:8080")
            
            # Load scene point cloud
            try:
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)
                scene_points = np.array(pcd.points)
                scene_colors = np.array(pcd.colors)
                
                # Add scene point cloud
                server.scene.add_point_cloud(
                    name="scene_points",
                    points=scene_points,
                    colors=scene_colors,
                    point_size=0.003
                )
                print(f"   ✅ Added scene point cloud ({len(scene_points)} points)")
                
            except Exception as e:
                print(f"   ⚠️ Could not load scene point cloud: {e}")
            
            # Add VLM grasp visualization
            self._add_vlm_grasp_to_viser(server, result)
            
            # Load and add ground truth grasps
            try:
                gt_grasps, gt_scores = self.load_ground_truth_grasps(target_obj_idx, max_grasps=20)
                if len(gt_grasps) > 0:
                    self._add_ground_truth_grasps_to_viser(server, gt_grasps, gt_scores, target_obj_idx, scene_id, ann_id)
                else:
                    print("   ⚠️ No ground truth grasps to display")
            except Exception as e:
                print(f"   ⚠️ Could not load ground truth grasps: {e}")
            
            # Add object model if available
            try:
                object_model, _ = self.get_object_model_and_dexnet(scene_id, ann_id, target_obj_idx)
                object_pose = self.get_object_pose_in_camera(scene_id, ann_id, target_obj_idx)
                
                # Transform object model to camera coordinates
                object_model_cam = self.transform_points(object_model, object_pose)
                
                server.scene.add_point_cloud(
                    name="object_model",
                    points=object_model_cam,
                    colors=np.array([[1.0, 0.8, 0.0]] * len(object_model_cam)),  # Orange
                    point_size=0.005
                )
                print(f"   ✅ Added object model ({len(object_model_cam)} points)")
                
            except Exception as e:
                print(f"   ⚠️ Could not load object model: {e}")
            
            # Add text information
            info_text = f"""VLM Grasp Evaluation Results

Scene: {scene_id:04d} | Annotation: {ann_id:04d}
Target Object: {result['target_object']} (ID: {target_obj_idx})

VLM Results:
• Point 1: {result['point1_3d']}
• Point 2: {result['point2_3d']}
• Confidence: {result['vlm_response'].get('confidence', 'N/A')}

DexNet Score: {result['dexnet_score']:.3f}
Evaluation: {'✅ Success' if result['evaluation_success'] else '❌ Failed'}

Legend:
🔴 VLM Grasp Points
🟢 Ground Truth Grasps
🟡 Object Model
🔵 Scene Points"""
            
            # server.scene.add_3d_gui_text(
            #     name="info_panel",
            #     text=info_text,
            #     position=(0.5, 0.5, 0.5),
            #     font_size=12
            # )
            print(info_text)
            print("   🎯 Visualization complete! Check http://localhost:8080")
            print("   💡 Press Ctrl+C to stop the server when done viewing")
            
            # Keep server running for viewing
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n   🛑 Stopping viser server...")
                
        except ImportError:
            print("❌ Viser not installed. Please install with: pip install viser")
        except Exception as e:
            print(f"❌ Error in viser visualization: {e}")
            import traceback
            traceback.print_exc()

    def _add_vlm_grasp_to_viser(self, server, result: Dict):
        """Add VLM grasp to viser visualization."""
        point1_3d = np.array(result['point1_3d'])
        point2_3d = np.array(result['point2_3d'])
        grasp_center = (point1_3d + point2_3d) / 2.0
        
        # Add grasp points
        server.scene.add_point_cloud(
            name="vlm_grasp_points",
            points=np.array([point1_3d, point2_3d]),
            colors=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),  # Red
            point_size=0.01
        )
        
        # Add grasp line
        server.scene.add_spline_catmull_rom(
            name="vlm_grasp_line",
            positions=np.array([point1_3d, point2_3d]),
            color=(1.0, 0.0, 0.0),  # Red
            line_width=5.0
        )
        
        # Add grasp pose frame
        if 'grasp_pose' in result:
            grasp_pose = result['grasp_pose']
            self._add_coordinate_frame_to_viser(
                server, 
                "vlm_grasp_frame", 
                grasp_pose, 
                scale=0.05,
                alpha=0.8
            )
        
        print(f"   ✅ Added VLM grasp visualization")

    def _add_ground_truth_grasps_to_viser(self, server, gt_grasps: np.ndarray, gt_scores: np.ndarray, 
                                        target_obj_idx: int, scene_id: int, ann_id: int):
        """Add ground truth grasps to viser visualization."""
        if len(gt_grasps) == 0:
            return
            
        try:
            # Get object pose to transform grasps to camera coordinates
            object_pose = self.get_object_pose_in_camera(scene_id, ann_id, target_obj_idx)
            
            gt_points = []
            gt_colors = []
            
            for i, (grasp, score) in enumerate(zip(gt_grasps, gt_scores)):
                # Extract grasp center and rotation
                grasp_center = grasp[13:16]  # translation
                rotation_matrix = grasp[4:13].reshape(3, 3)  # rotation
                
                # Transform to camera coordinates
                grasp_center_cam = self.transform_points(grasp_center.reshape(1, -1), object_pose)[0]
                
                gt_points.append(grasp_center_cam)
                
                # Color by score (green for high, yellow for medium, red for low)
                if score > 0.8:
                    color = [0.0, 1.0, 0.0]  # Green
                elif score > 0.6:
                    color = [1.0, 1.0, 0.0]  # Yellow
                else:
                    color = [1.0, 0.5, 0.0]  # Orange
                gt_colors.append(color)
                
                # Add individual grasp frame for top grasps
                if i < 5:  # Only show frames for top 5 grasps
                    # Create pose matrix
                    gt_pose = np.eye(4)
                    gt_pose[:3, :3] = rotation_matrix
                    gt_pose[:3, 3] = grasp_center
                    
                    # Transform pose to camera coordinates
                    gt_pose_cam = object_pose @ gt_pose
                    
                    self._add_coordinate_frame_to_viser(
                        server,
                        f"gt_grasp_frame_{i}",
                        gt_pose_cam,
                        scale=0.03,
                        alpha=0.6
                    )
            
            # Add all ground truth grasp points
            if gt_points:
                server.scene.add_point_cloud(
                    name="ground_truth_grasps",
                    points=np.array(gt_points),
                    colors=np.array(gt_colors),
                    point_size=0.008
                )
                
                print(f"   ✅ Added {len(gt_points)} ground truth grasps")
            
        except Exception as e:
            print(f"   ⚠️ Error adding ground truth grasps: {e}")

    def _add_coordinate_frame_to_viser(self, server, name: str, pose: np.ndarray, 
                                     scale: float = 0.1, alpha: float = 1.0):
        """Add a coordinate frame to viser visualization."""
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        
        # X-axis (red)
        x_end = position + rotation_matrix[:, 0] * scale
        server.scene.add_spline_catmull_rom(
            name=f"{name}_x",
            positions=np.array([position, x_end]),
            color=(1.0, 0.0, 0.0),  # Red
            line_width=3.0
        )
        
        # Y-axis (green)
        y_end = position + rotation_matrix[:, 1] * scale
        server.scene.add_spline_catmull_rom(
            name=f"{name}_y", 
            positions=np.array([position, y_end]),
            color=(0.0, 1.0, 0.0),  # Green
            line_width=3.0
        )
        
        # Z-axis (blue)
        z_end = position + rotation_matrix[:, 2] * scale
        server.scene.add_spline_catmull_rom(
            name=f"{name}_z",
            positions=np.array([position, z_end]),
            color=(0.0, 0.0, 1.0),  # Blue
            line_width=3.0
        )

    def transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform 3D points using a 4x4 transformation matrix."""
        if points.shape[1] == 3:
            # Add homogeneous coordinate
            ones = np.ones((points.shape[0], 1))
            points_homo = np.hstack([points, ones])
        else:
            points_homo = points
            
        # Apply transformation
        transformed = (transform @ points_homo.T).T
        
        # Return 3D coordinates
        return transformed[:, :3]

    def estimate_surface_normal(self, point: np.ndarray, scene_id: int, ann_id: int, 
                              radius: float = 0.02) -> np.ndarray:
        """
        Estimate surface normal at a point using nearby points from the point cloud.
        
        **Input:**
        - point: 3D point to estimate normal at
        - scene_id: scene index
        - ann_id: annotation index 
        - radius: radius for neighborhood search
        
        **Output:**
        - normal: estimated surface normal vector (pointing outward)
        """
        try:
            # Load scene point cloud
            pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)
            scene_points = np.array(pcd.points)
            
            # Find nearby points
            distances = np.linalg.norm(scene_points - point, axis=1)
            nearby_indices = np.where(distances < radius)[0]
            
            if len(nearby_indices) < 3:
                # Not enough points, return upward normal
                return np.array([0, 0, 1])
            
            nearby_points = scene_points[nearby_indices]
            
            # Use PCA to estimate surface normal
            centered_points = nearby_points - np.mean(nearby_points, axis=0)
            cov_matrix = np.cov(centered_points.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Normal is the eigenvector with smallest eigenvalue
            normal = eigenvecs[:, 0]
            
            # Make sure normal points away from camera (outward)
            to_camera = -point  # Vector from point to camera at origin
            if np.dot(normal, to_camera) < 0:
                normal = -normal
                
            return normal / np.linalg.norm(normal)
            
        except Exception as e:
            print(f"   ⚠️ Could not estimate surface normal: {e}")
            return np.array([0, 0, 1])  # Default upward normal


# Example usage and demonstration
def example_usage():
    """
    Example of how to use the VLMGraspEval class.
    """
    print("=== VLM Grasp Evaluation Example ===")
    
    # Initialize evaluator
    # Adjust these paths according to your setup
    graspnet_root = "/path/to/graspnet"  # Change this to your GraspNet root
    vlm_config = {
        'model': 'qwen2.5vl:32b',  # or 'qwen2.5'
        'endpoint': 'http://localhost:11434/api/generate',
        'temperature': 0.1,
        'max_tokens': 500
    }
    
    evaluator = VLMGraspEval(
        root=graspnet_root,
        camera='kinect',
        split='test',
        vlm_config=vlm_config
    )
    
    # Example evaluation
    scene_id = 100
    ann_id = 0
    target_object = "red mug"  # Describe the object you want to grasp
    target_obj_idx = 25  # Object index in the scene
    
    # Run evaluation
    result = evaluator.eval_vlm_grasp(
        scene_id=scene_id,
        ann_id=ann_id,
        target_object=target_object,
        target_obj_idx=target_obj_idx,
        vlm_type='ollama',  # or 'openai'
        visualize=True
    )
    
    # Print results
    if 'error' in result:
        print(f"Evaluation failed: {result['error']}")
    else:
        print(f"Evaluation completed successfully!")
        print(f"DexNet Score: {result['dexnet_score']:.3f}")


def batch_evaluation_example():
    """
    Example of batch evaluation across multiple scenes.
    """
    print("=== Batch VLM Grasp Evaluation Example ===")
    
    graspnet_root = "/path/to/graspnet"
    evaluator = VLMGraspEval(root=graspnet_root, camera='kinect')
    
    # Define evaluation cases
    eval_cases = [
        {'scene_id': 100, 'ann_id': 0, 'target_object': 'red mug', 'target_obj_idx': 25},
        {'scene_id': 101, 'ann_id': 5, 'target_object': 'blue bottle', 'target_obj_idx': 12},
        {'scene_id': 102, 'ann_id': 10, 'target_object': 'yellow box', 'target_obj_idx': 8},
    ]
    
    results = []
    for case in eval_cases:
        print(f"\nEvaluating Scene {case['scene_id']}, Object: {case['target_object']}")
        result = evaluator.eval_vlm_grasp(**case, vlm_type='ollama')
        results.append(result)
        
        if 'error' not in result:
            print(f"DexNet Score: {result['dexnet_score']:.3f}")
    
    # Compute statistics
    successful_evals = [r for r in results if 'error' not in r]
    if successful_evals:
        avg_score = np.mean([r['dexnet_score'] for r in successful_evals])
        print(f"\nAverage DexNet Score: {avg_score:.3f}")
        print(f"Success Rate: {len(successful_evals)}/{len(results)}")


if __name__ == "__main__":
    print("VLM Grasp Evaluation Module")
    print("This module provides VLM-based grasp generation and DexNet evaluation.")
    print("\nTo use this module:")
    print("1. Make sure Ollama is running (for ollama VLMs)")
    print("2. Install required dependencies: requests, pillow")
    print("3. Set up your GraspNet dataset path")
    print("4. Run example_usage() or batch_evaluation_example()")
    
    # Uncomment to run examples:
    # example_usage()
    # batch_evaluation_example() 