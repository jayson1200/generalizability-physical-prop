# Scaffolding Dexterous Manipulation with Vision-Language Models

[Website](https://sites.google.com/view/dexterous-vlm-scaffolding) | [ArXiv](https://arxiv.org/abs/2506.19212)

This is the official codebase for "Scaffolding Dexterous Manipulation with Vision-Language Models" by Vincent de Bakker, Joey Hejna, Tyler Ga Wei Lum, Onur Celik, Aleksandar Taranovic, Denis Blessing, Gerhard Neumann, Jeannette Bohg, and Dorsa Sadigh. 

The framework enables training and deployment of dexterous manipulation policies using vision-language models for various manipulation tasks.

If you find our paper or code insightful, feel free to cite us with the following bibtex:

```
@misc{debakker2025scaffoldingdexterousmanipulationvisionlanguage,
      title={Scaffolding Dexterous Manipulation with Vision-Language Models}, 
      author={Vincent de Bakker and Joey Hejna and Tyler Ga Wei Lum and Onur Celik and Aleksandar Taranovic and Denis Blessing and Gerhard Neumann and Jeannette Bohg and Dorsa Sadigh},
      year={2025},
      eprint={2506.19212},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.19212}, 
}
```

## Setup

### Environment Setup

Create and activate a conda environment:

```bash
conda create --name=vlm-scaffolding --python=3.10
conda activate vlm-scaffolding
pip install -r requirements.txt
```

### Gemini Setup

We use [**Gemini 2.5 Flash Thinking**](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash) as our Vision-Language Model (VLM).

To use Gemini, you must first authenticate through the **Google Cloud Platform (GCP)**. Run the following command in your terminal to log in:

```bash
gcloud auth login
```

## Usage

### Supported Tasks and Environments

The framework supports the following tasks in simulation:

| Task Name (CLI) | Environment Name (Training) |
|----------------|----------------------------|
| `apple`        | `EnvApple-v0`             |
| `bottle`       | `EnvBottle-v0`            |
| `hammer`       | `EnvHammer-v0`            |
| `drawer`       | `EnvDrawer-v0`            |
| `fridge`       | `EnvFridge-v0`            |
| `sponge`       | `EnvSponge-v0`            |
| `pliers`       | `EnvPliers-v0`            |
| `scissors`     | `EnvScissors-v0`          |

Additionally, we support training for three real-world tasks in simulation:

| Task Name (CLI) | Environment Name (Training) |
|----------------|----------------------------|
| `box_arm`      | `EnvBoxArm-v0`            |
| `bottle_arm`   | `EnvBottleArm-v0`         |
| `hammer_arm`   | `EnvHammerArm-v0`         |

### Training Pipeline

The training process consists of two main steps:

1. **Dataset Generation**: Create a dataset of keypoints and trajectories using various methods
2. **RL Training**: Train an RL agent using the generated dataset

#### Dataset Generation

We support multiple methods for creating keypoints and trajectories:

**Methods:**
- `gemini`: Default Gemini-based method
- `gemini_few_shot_oracle`: Gemini with few-shot examples
- `gemini_iteration_2_oracle`: Gemini with few-shot examples (2. iteration)
- `gemini_iteration_3_oracle`: Gemini with few-shot examples (3. iteration)
- `gemini_3`: Gemini with 3 waypoints
- `gemini_5`: Gemini with 5 waypoints
- `gemini_10`: Gemini with 10 waypoints
- `gemini_40`: Gemini with 40 waypoints
- `gemini_keypoint_oracle`: Gemini with keypoint oracle
- `gemini_trajectory_oracle`: Gemini with trajectory oracle
- `scripted`: Scripted demonstrations

To generate a dataset:

```bash
python -m build_dataset.build_dataset \
    --task {task} \
    --num-samples=100 \
    --method {method} \
    --split {split} \
    --test
```

The `split` parameter accepts the following values:

**Training Splits:**
- `train1`: First training set
- `train2`: Second training set  
- `train3`: Third training set

**Test Splits:**
- `test`: Standard test set
- `test1`, `test2`, `test3`: Extended test sets (only supported for specific methods)

For few-shot methods, you first need to generate few-shot examples:

```bash
python -m build_dataset.get_fewshot_examples --task {task} --split {split} --method {method}
```

#### Model Training

To train an RL agent, use the following command:

```bash
python -m train.train \
    --env-name {env_name} \
    --num-iterations 2000 \
    --method {method} \
    --split {split}
```

Example: 
```bash
python -m train.train \
    --env-name EnvApple-v0 \
    --num-iterations 2000 \
    --method gemini \
    --split train1
```

## Real-World Experiments

### Camera Setup

On the machine with ZED camera connected and ZED ROS installed:

```bash
roslaunch zed_wrapper zed.launch
```

This will be the master node. Get the hostname with:

```bash
echo $(hostname)
```

We will refer to this as `${MASTER_HOSTNAME}`.

### ROS Configuration

On every terminal, set the ROS master URI:

```bash
export ROS_MASTER_URI=${MASTER_HOSTNAME}:11311
```

Set object parameters (run once):

```bash
# Set object files
rosparam set /mesh_file <path_to_obj_file>  # e.g., .../starbucks_bottle/starbucks_bottle.obj
rosparam set /mesh_file_2 <path_to_obj_file_2>  # e.g., .../plate/plate.obj

# Set text prompts
rosparam set /text_prompt <text_prompt>  # e.g., "white bottle"
rosparam set /text_prompt_2 <text_prompt_2>  # e.g., "blue plate"
```

### Component Setup

#### Segment Anything 2
Run on machine with SAM2 ROS installed:
```bash
python sam2_ros_node.py
python sam2_ros_node_2.py
```

#### FoundationPose
Run on machine with FoundationPose ROS installed:
```bash
# From docker container
apt-get update && apt-get install -y \
    libosmesa6 libosmesa6-dev libgl1-mesa-glx

python fp_ros_node.py
python fp_ros_node_2.py
PYOPENGL_PLATFORM=osmesa python fp_evaluator_ros_node.py
PYOPENGL_PLATFORM=osmesa python fp_evaluator_ros_node_2.py
```

#### Keypoint Tracker
```bash
python -m real_world.keypoint_ros_node --keypoint_x_idx ... --keypoint_y_idx ...
python -m real_world.keypoint_ros_node_2 --keypoint_x_idx ... --keypoint_y_idx ...
```

#### Robot Control
Robot arm:
```bash
roslaunch iiwa_control joint_position_control.launch
```

Hand:
```bash
roslaunch allegro_hand allegro_hand.launch HAND:=right AUTO_CAN:=true CONTROLLER:=pd
```

### Trajectory Preparation

```bash
python -m real_world.prepare_real_trajectories \
    --task_name <task_name>  # e.g., bottle_arm3, box_arm
```

### Policy Execution

```bash
python -m real_world.policy_ros_node
```

### Additional Tools

#### Scene Visualization
```bash
python -m real_world.visualization_ros_node
```

#### Robot Positioning
Move the arm to start position:
```bash
python -m real_world.move_arm_to_start
```

Move the hand to start position:
```bash
rostopic pub /allegroHand_0/joint_cmd sensor_msgs/JointState "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
position: [-0.4700, -0.1960, -0.1740, -0.2270,
          -0.4700, -0.1960, -0.1740, -0.2270,
          -0.4700, -0.1960, -0.1740, -0.2270,
           0.7, -0.1050, -0.1890, -0.1620]
velocity: []
effort: []"
```

#### Testing
Simulate a fake robot for testing:
```bash
python -m real_world.fake_robot_ros_node
```

## BibTeX

```
@misc{debakker2025scaffoldingdexterousmanipulationvisionlanguage,
      title={Scaffolding Dexterous Manipulation with Vision-Language Models}, 
      author={Vincent de Bakker and Joey Hejna and Tyler Ga Wei Lum and Onur Celik and Aleksandar Taranovic and Denis Blessing and Gerhard Neumann and Jeannette Bohg and Dorsa Sadigh},
      year={2025},
      eprint={2506.19212},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.19212}, 
}
```

## License

This code has an MIT license, found in the [LICENSE](LICENSE) file.
