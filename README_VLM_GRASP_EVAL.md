# VLM Grasp Evaluation System

A novel approach to grasp generation and evaluation that combines Vision Language Models (VLMs) with DexNet for robotic grasping on the GraspNet dataset.

## 🎯 Overview

This system allows you to:
1. **Input**: Scene image + target object description
2. **VLM Processing**: Get antipodal grasp points from VLMs (Ollama LLaMA, Qwen2.5, GPT-4V)
3. **Pose Generation**: Convert antipodal points to 6DOF grasp poses
4. **Quality Evaluation**: Assess grasp quality using DexNet
5. **Visualization**: View results in 3D

## 🚀 Quick Start

### Prerequisites

1. **GraspNet Dataset**: Download and set up the GraspNet-1Billion dataset
2. **VLM Backend**: Choose one:
   - **Ollama** (Local, Free): Install and run Ollama with vision models
   - **OpenAI** (Cloud, Paid): Get GPT-4V API access

### Installation

```bash
# Install dependencies
pip install -r requirements_vlm.txt

# For Ollama (recommended for local deployment)
# Install Ollama: https://ollama.ai/
ollama pull llama3.2-vision  # or qwen2-vl
ollama serve  # Start Ollama server
```

### Basic Usage

```bash
# Single grasp evaluation
python demo_vlm_grasp_eval.py \
    --graspnet_root /home/kych/graspnet \
    --camera kinect \
    --vlm_type ollama \
    --demo_type single \
    --scene_id 0 \
    --target_object "cheezit box" \
    --target_obj_idx 0

# Batch evaluation
python demo_vlm_grasp_eval.py \
    --graspnet_root /path/to/graspnet \
    --demo_type batch \
    --vlm_type ollama

# Interactive mode
python demo_vlm_grasp_eval.py \
    --graspnet_root /path/to/graspnet \
    --demo_type interactive
```

## 📚 API Reference

### VLMGraspEval Class

```python
from graspnetAPI.vlm_grasp_eval import VLMGraspEval

# Initialize
evaluator = VLMGraspEval(
    root="/path/to/graspnet",
    camera="kinect",
    split="test",
    vlm_config={
        'model': 'llama3.2-vision',
        'endpoint': 'http://localhost:11434/api/generate',
        'temperature': 0.1,
        'max_tokens': 500
    }
)

# Evaluate a grasp
result = evaluator.eval_vlm_grasp(
    scene_id=100,
    ann_id=0,
    target_object="red mug",
    target_obj_idx=25,
    vlm_type='ollama',
    visualize=True
)

print(f"DexNet Score: {result['dexnet_score']:.3f}")
```

### Key Methods

#### `eval_vlm_grasp()`
Complete evaluation pipeline from image to DexNet score.

**Parameters:**
- `scene_id`: Scene index in GraspNet
- `ann_id`: Annotation index (0-255)
- `target_object`: Natural language description of target object
- `target_obj_idx`: Object index in the scene
- `vlm_type`: 'ollama' or 'openai'
- `api_key`: OpenAI API key (if using OpenAI)
- `visualize`: Show 3D visualization

**Returns:**
```python
{
    'scene_id': 100,
    'ann_id': 0,
    'target_object': 'red mug',
    'target_obj_idx': 25,
    'vlm_response': {
        'point1': [x1, y1, z1],
        'point2': [x2, y2, z2],
        'confidence': 0.85,
        'reasoning': 'These points provide stable grip...'
    },
    'grasp_pose': 4x4_transformation_matrix,
    'grasp_array': graspnet_format_array,
    'dexnet_score': 0.742,
    'evaluation_success': True
}
```

#### `antipodal_points_to_grasp_pose()`
Convert two 3D points to a 6DOF grasp pose.

#### `evaluate_grasp_dexnet()`
Evaluate grasp quality using DexNet metrics.

## 🔧 VLM Backends

### Ollama (Recommended)

**Advantages:**
- ✅ Free and open source
- ✅ Runs locally (privacy)
- ✅ No API costs
- ✅ Multiple vision models available

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull vision models
ollama pull llama3.2-vision    # 11B parameters
ollama pull qwen2-vl          # Alternative model
ollama pull llava             # Another option

# Start server
ollama serve
```

**Supported Models:**
- `llama3.2-vision` (Recommended)
- `qwen2-vl`
- `llava:latest`
- `bakllava`

### OpenAI GPT-4V

**Advantages:**
- ✅ High accuracy
- ✅ Reliable responses
- ✅ No local setup required

**Setup:**
```python
# Usage with API key
result = evaluator.eval_vlm_grasp(
    scene_id=100,
    ann_id=0,
    target_object="red mug",
    target_obj_idx=25,
    vlm_type='openai',
    api_key='your-openai-api-key'
)
```

## 📊 Evaluation Metrics

### DexNet Score
- **Range**: [-1, 1]
- **Interpretation**:
  - `> 0.7`: Excellent grasp 🟢
  - `0.5-0.7`: Good grasp 🟡  
  - `0.3-0.5`: Fair grasp 🟠
  - `< 0.3`: Poor grasp 🔴

### VLM Confidence
- **Range**: [0, 1]
- **Meaning**: VLM's confidence in the suggested grasp points

## 🎨 Examples

### Example 1: Single Object Evaluation

```python
# Evaluate grasping a mug
result = evaluator.eval_vlm_grasp(
    scene_id=100,
    ann_id=0,
    target_object="ceramic mug with handle",
    target_obj_idx=25,
    vlm_type='ollama'
)

if result['evaluation_success']:
    print(f"✅ DexNet Score: {result['dexnet_score']:.3f}")
    print(f"🤖 VLM Reasoning: {result['vlm_response']['reasoning']}")
```

### Example 2: Batch Evaluation

```python
test_cases = [
    {'scene_id': 100, 'target_object': 'red mug', 'target_obj_idx': 25},
    {'scene_id': 101, 'target_object': 'blue bottle', 'target_obj_idx': 12},
    {'scene_id': 102, 'target_object': 'small box', 'target_obj_idx': 8},
]

results = []
for case in test_cases:
    result = evaluator.eval_vlm_grasp(**case, vlm_type='ollama')
    results.append(result)

# Analyze results
scores = [r['dexnet_score'] for r in results if 'error' not in r]
print(f"Average DexNet Score: {np.mean(scores):.3f}")
```

### Example 3: Custom VLM Prompt

```python
custom_prompt = """
Look at this image and find the {target_object}. 
I need two grasp points that would allow a parallel-jaw gripper 
to pick up this object safely without causing damage.

Consider:
- Object material and fragility
- Weight distribution
- Surface texture
- Optimal approach angle

Return exactly two 3D points in JSON format:
{{"point1": [x1, y1, z1], "point2": [x2, y2, z2], "confidence": 0.9}}
"""

result = evaluator.query_vlm_ollama(
    image_path="scene_image.png",
    target_object="glass bottle",
    prompt=custom_prompt
)
```

## 🔬 Advanced Features

### Custom Grasp Parameters

```python
# Convert with custom approach vector
grasp_pose = evaluator.antipodal_points_to_grasp_pose(
    point1=[0.1, 0.2, 0.3],
    point2=[0.15, 0.2, 0.3],
    approach_vector=[0, 0, -1]  # Top-down approach
)

# Create grasp with custom width
grasp_array = evaluator.pose_to_grasp_array(
    grasp_pose=grasp_pose,
    width=0.06,  # 6cm gripper width
    confidence=0.9
)
```

### Collision Detection

The system automatically checks for collisions with:
- Other objects in the scene
- Table surface
- Robot workspace constraints

### Visualization Options

```python
# Enable/disable visualization
result = evaluator.eval_vlm_grasp(
    scene_id=100,
    ann_id=0,
    target_object="mug",
    target_obj_idx=25,
    visualize=True  # Shows 3D scene with grasp
)

# Manual visualization
evaluator.visualize_grasp_result(scene_id, ann_id, result)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama not responding**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Restart if needed
   ollama serve
   ```

2. **Model not found**
   ```bash
   # List available models
   ollama list
   
   # Pull required model
   ollama pull llama3.2-vision
   ```

3. **JSON parsing errors**
   - VLM responses may be inconsistent
   - Try adjusting temperature (lower = more consistent)
   - Use custom prompts for better formatting

4. **DexNet evaluation fails**
   - Check if object models exist in GraspNet dataset
   - Verify object index is correct for the scene

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check VLM response before parsing
vlm_response = evaluator.query_vlm_ollama(image_path, target_object)
print("Raw VLM Response:", vlm_response)
```

## 📈 Performance Tips

1. **VLM Selection**: Ollama models are faster for batch processing
2. **Batch Processing**: Disable visualization for faster evaluation
3. **Caching**: VLM responses can be cached to avoid repeated calls
4. **Parallel Processing**: Evaluate multiple scenes in parallel

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Additional VLM backends (Anthropic Claude, Google Gemini)
- [ ] Improved prompt engineering
- [ ] Better coordinate system handling
- [ ] Real robot integration
- [ ] Performance optimizations

## 📄 License

This project extends the GraspNet-API and follows the same license terms.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{vlm_grasp_eval_2024,
  title={VLM-Based Grasp Evaluation: Integrating Vision Language Models with DexNet for Robotic Grasping},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## 🔗 Related Work

- [GraspNet-1Billion](https://graspnet.net/): Original dataset and evaluation
- [DexNet](https://berkeleyautomation.github.io/dex-net/): Grasp quality evaluation
- [Ollama](https://ollama.ai/): Local LLM inference
- [OpenAI GPT-4V](https://openai.com/research/gpt-4v-system-card): Vision language model 