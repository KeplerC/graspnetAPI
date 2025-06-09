#!/usr/bin/env python3
"""
Demo script for VLM-based grasp evaluation using GraspNet dataset.

This script demonstrates how to use the VLMGraspEval class to:
1. Connect to VLMs (Ollama or OpenAI)
2. Generate grasp poses from images
3. Evaluate grasp quality using DexNet
4. Visualize results

Prerequisites:
- GraspNet dataset downloaded and set up
- Ollama running locally (for Ollama models) OR OpenAI API key (for GPT-4V)
- Required Python packages: requests, pillow, open3d, numpy

Usage:
    python demo_vlm_grasp_eval.py --graspnet_root /path/to/graspnet --camera kinect
"""

import argparse
import os
import sys
import json
import time
from typing import Dict, List

# Add the graspnetAPI to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graspnetAPI.vlm_grasp_eval import VLMGraspEval, example_usage, batch_evaluation_example


def check_ollama_connection(endpoint: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    import requests
    try:
        response = requests.get(f"{endpoint}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_ollama_models(endpoint: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    import requests
    try:
        response = requests.get(f"{endpoint}/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []


def setup_vlm_config(vlm_type: str = 'ollama', api_key: str = None) -> Dict:
    """Set up VLM configuration based on the selected type."""
    if vlm_type == 'ollama':
        # Check Ollama connection
        if not check_ollama_connection():
            print("❌ Ollama is not running or not accessible at http://localhost:11434")
            print("Please start Ollama by running: ollama serve")
            return None
        
        # List available models
        models = list_ollama_models()
        print(f"✅ Ollama is running. Available models: {models}")
        
        # Recommend vision models
        vision_models = [m for m in models if any(vm in m.lower() for vm in ['vision', 'llava', 'llama3.2', 'qwen2'])]
        
        if not vision_models:
            print("⚠️  No vision models found. You may need to pull a vision model:")
            print("ollama pull llama3.2-vision")
            print("ollama pull qwen2-vl")
            model_name = input("Enter model name to use: ").strip() or "llama3.2-vision"
        else:
            print(f"Recommended vision models: {vision_models}")
            model_name = vision_models[0]  # Use first available vision model
        
        config = {
            'model': model_name,
            'endpoint': 'http://localhost:11434/api/generate',
            'temperature': 0.1,
            'max_tokens': 500
        }
        
    elif vlm_type == 'openai':
        if not api_key:
            print("❌ OpenAI API key is required for GPT-4V")
            return None
        
        config = {
            'model': 'gpt-4-vision-preview',
            'api_key': api_key,
            'temperature': 0.1,
            'max_tokens': 500
        }
    
    else:
        print(f"❌ Unsupported VLM type: {vlm_type}")
        return None
    
    return config


def single_grasp_demo(evaluator: VLMGraspEval, scene_id: int, ann_id: int, 
                     target_object: str, target_obj_idx: int, vlm_type: str, 
                     api_key: str = None):
    """Demo for evaluating a single grasp."""
    print(f"\n{'='*60}")
    print(f"🎯 Single Grasp Evaluation Demo")
    print(f"{'='*60}")
    print(f"Scene ID: {scene_id}")
    print(f"Annotation ID: {ann_id}")
    print(f"Target Object: {target_object}")
    print(f"Object Index: {target_obj_idx}")
    print(f"VLM Type: {vlm_type}")
    
    # Run evaluation
    start_time = time.time()
    result = evaluator.eval_vlm_grasp(
        scene_id=scene_id,
        ann_id=ann_id,
        target_object=target_object,
        target_obj_idx=target_obj_idx,
        vlm_type=vlm_type,
        api_key=api_key,
        visualize=True  # Set to False to disable 3D visualization
    )
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n📊 Evaluation Results:")
    print(f"{'─'*40}")
    
    if 'error' in result:
        print(f"❌ Evaluation failed: {result['error']}")
        return None
    
    print(f"✅ Evaluation completed successfully!")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"🤖 VLM Response:")
    print(f"   - Point 1 (2D): {result['point1_2d']} pixels")
    print(f"   - Point 2 (2D): {result['point2_2d']} pixels") 
    print(f"   - Point 1 (3D): {result['point1_3d']}")
    print(f"   - Point 2 (3D): {result['point2_3d']}")
    print(f"   - VLM Confidence: {result['vlm_response'].get('confidence', 'N/A')}")
    print(f"   - Reasoning: {result['vlm_response'].get('reasoning', 'N/A')}")
    print(f"🔬 DexNet Quality Score: {result['dexnet_score']:.3f}")
    
    # Interpret the score
    if result['dexnet_score'] >= 0.7:
        quality = "Excellent 🟢"
    elif result['dexnet_score'] >= 0.5:
        quality = "Good 🟡"
    elif result['dexnet_score'] >= 0.3:
        quality = "Fair 🟠"
    else:
        quality = "Poor 🔴"
    
    print(f"📈 Grasp Quality: {quality}")
    
    return result


def batch_grasp_demo(evaluator: VLMGraspEval, vlm_type: str, api_key: str = None):
    """Demo for batch evaluation of multiple grasps."""
    print(f"\n{'='*60}")
    print(f"📦 Batch Grasp Evaluation Demo")
    print(f"{'='*60}")
    
    # Define test cases - adjust these based on your dataset
    test_cases = [
        {
            'scene_id': 5, 
            'ann_id': 0, 
            'target_object': 'red mug', 
            'target_obj_idx': 0,
            'description': 'Cylindrical mug with handle'
        },
        {
            'scene_id': 10, 
            'ann_id': 5, 
            'target_object': 'plastic bottle', 
            'target_obj_idx': 1,
            'description': 'Tall cylindrical bottle'
        },
        {
            'scene_id': 11, 
            'ann_id': 10, 
            'target_object': 'small box', 
            'target_obj_idx': 2,
            'description': 'Rectangular cardboard box'
        }
    ]
    
    results = []
    successful_evaluations = 0
    total_time = 0
    
    for i, case in enumerate(test_cases):
        print(f"\n📋 Test Case {i+1}/{len(test_cases)}: {case['description']}")
        print(f"   Scene: {case['scene_id']}, Object: {case['target_object']}")
        
        start_time = time.time()
        result = evaluator.eval_vlm_grasp(
            scene_id=case['scene_id'],
            ann_id=case['ann_id'],
            target_object=case['target_object'],
            target_obj_idx=case['target_obj_idx'],
            vlm_type=vlm_type,
            api_key=api_key,
            visualize=False  # Disable visualization for batch processing
        )
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        
        if 'error' not in result:
            successful_evaluations += 1
            print(f"   ✅ DexNet Score: {result['dexnet_score']:.3f} (Time: {elapsed_time:.1f}s)")
        else:
            print(f"   ❌ Failed: {result['error']}")
        
        results.append(result)
    
    # Summary statistics
    print(f"\n📊 Batch Evaluation Summary:")
    print(f"{'─'*40}")
    print(f"Total Cases: {len(test_cases)}")
    print(f"Successful: {successful_evaluations}")
    print(f"Success Rate: {successful_evaluations/len(test_cases)*100:.1f}%")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Average Time per Case: {total_time/len(test_cases):.1f}s")
    
    if successful_evaluations > 0:
        scores = [r['dexnet_score'] for r in results if 'error' not in r]
        print(f"Average DexNet Score: {sum(scores)/len(scores):.3f}")
        print(f"Best Score: {max(scores):.3f}")
        print(f"Worst Score: {min(scores):.3f}")
    
    return results


def interactive_demo(evaluator: VLMGraspEval, vlm_type: str, api_key: str = None):
    """Interactive demo where user can input custom parameters."""
    print(f"\n{'='*60}")
    print(f"🎮 Interactive Grasp Evaluation Demo")
    print(f"{'='*60}")
    
    while True:
        print(f"\nEnter grasp evaluation parameters (or 'quit' to exit):")
        
        # Get user input
        scene_id = input("Scene ID (e.g., 100): ").strip()
        if scene_id.lower() == 'quit':
            break
        
        try:
            scene_id = int(scene_id)
        except ValueError:
            print("❌ Invalid scene ID. Please enter a number.")
            continue
        
        ann_id = input("Annotation ID (e.g., 0): ").strip()
        try:
            ann_id = int(ann_id)
        except ValueError:
            print("❌ Invalid annotation ID. Please enter a number.")
            continue
        
        target_object = input("Target object description (e.g., 'red mug'): ").strip()
        if not target_object:
            print("❌ Please provide an object description.")
            continue
        
        target_obj_idx = input("Target object index (e.g., 25): ").strip()
        try:
            target_obj_idx = int(target_obj_idx)
        except ValueError:
            print("❌ Invalid object index. Please enter a number.")
            continue
        
        # Run evaluation
        result = single_grasp_demo(
            evaluator, scene_id, ann_id, target_object, 
            target_obj_idx, vlm_type, api_key
        )
        
        # Ask if user wants to continue
        continue_demo = input("\nContinue with another evaluation? (y/n): ").strip().lower()
        if continue_demo != 'y':
            break


def main():
    parser = argparse.ArgumentParser(description="VLM Grasp Evaluation Demo")
    parser.add_argument('--graspnet_root', type=str, required=True,
                       help='Path to GraspNet dataset root directory')
    parser.add_argument('--camera', type=str, default='kinect',
                       choices=['kinect', 'realsense'],
                       help='Camera type (default: kinect)')
    parser.add_argument('--vlm_type', type=str, default='ollama',
                       choices=['ollama', 'openai'],
                       help='VLM type to use (default: ollama)')
    parser.add_argument('--openai_api_key', type=str,
                       help='OpenAI API key (required for openai VLM type)')
    parser.add_argument('--demo_type', type=str, default='single',
                       choices=['single', 'batch', 'interactive'],
                       help='Type of demo to run (default: single)')
    parser.add_argument('--scene_id', type=int, default=5,
                       help='Scene ID for single demo (default: 5)')
    parser.add_argument('--ann_id', type=int, default=0,
                       help='Annotation ID for single demo (default: 0)')
    parser.add_argument('--target_object', type=str, default='mug',
                       help='Target object description for single demo (default: mug)')
    parser.add_argument('--target_obj_idx', type=int, default=0,
                       help='Target object index for single demo (default: 0)')
    
    args = parser.parse_args()
    
    print("🚀 VLM Grasp Evaluation Demo")
    print("="*50)
    
    # Validate GraspNet root
    if not os.path.exists(args.graspnet_root):
        print(f"❌ GraspNet root directory not found: {args.graspnet_root}")
        return
    
    print(f"📁 GraspNet Root: {args.graspnet_root}")
    print(f"📷 Camera: {args.camera}")
    print(f"🤖 VLM Type: {args.vlm_type}")
    
    # Set up VLM configuration
    vlm_config = setup_vlm_config(args.vlm_type, args.openai_api_key)
    if vlm_config is None:
        return
    
    print(f"⚙️  VLM Config: {vlm_config}")
    
    # Initialize evaluator
    try:
        evaluator = VLMGraspEval(
            root=args.graspnet_root,
            camera=args.camera,
            split='test',
            vlm_config=vlm_config
        )
        print("✅ VLMGraspEval initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize VLMGraspEval: {e}")
        return
    
    # Run selected demo
    if args.demo_type == 'single':
        single_grasp_demo(
            evaluator, args.scene_id, args.ann_id, 
            args.target_object, args.target_obj_idx, 
            args.vlm_type, args.openai_api_key
        )
    elif args.demo_type == 'batch':
        batch_grasp_demo(evaluator, args.vlm_type, args.openai_api_key)
    elif args.demo_type == 'interactive':
        interactive_demo(evaluator, args.vlm_type, args.openai_api_key)
    
    print(f"\n🎉 Demo completed!")


if __name__ == "__main__":
    main() 