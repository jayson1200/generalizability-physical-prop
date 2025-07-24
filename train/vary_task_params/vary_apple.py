
#!/usr/bin/env python3
import os
import shutil
import argparse
import xml.etree.ElementTree as ET
from .utils import set_mass_distribution

import pandas as pd
import tqdm
import yaml
import mujoco

import numpy as np
import torch

import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper
from tasks import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on a ManiSkill environment"
    )
    parser.add_argument("--method", type=str, default="scripted",
                        help="Method to evaluate on")
    parser.add_argument("--split", type=str, default="test", help="Dataset to test on")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--friction", type=str,
                        help="(linear friction, z-torque, xy-torque)")
    parser.add_argument("--density", type=str, help="density")
    parser.add_argument("--center_of_mass", type=str, help="pos x y z for new COM")
    parser.add_argument("--scale", type=str, help="uniform scale factor")
    return parser.parse_args()

args = parse_args()
flags = {
    "friction":  args.friction,
    "density":   args.density,
    "center_of_mass": args.center_of_mass,
    "scale":     args.scale,
}

if sum(v is not None for v in flags.values()) != 1:
    raise ValueError(
        "Exactly one of --solimp, --solref, --friction, "
        "--density, --center_of_mass, or --scale must be provided."
    )

columns=[
    "friction",
    "density",
    "center_of_mass",
    "scale",
    "max_contact_force",
    "success_rate",
]

# 3) Prepare results DataFrame
results_file = "varying_apple_params.csv"
results_df = pd.read_csv(results_file) if os.path.exists(results_file) else  pd.DataFrame(columns=columns)

# 4) Backup original XML
object_location = "assets/objects/apple/model.xml"
tmp_location    = "assets/objects/apple/model_backup.xml"
shutil.copyfile(object_location, tmp_location)

# 5) Parse & modify
tree = ET.parse(object_location)
root = tree.getroot()

for geom in root.findall(".//geom"):
    if args.friction:
        geom.set("friction", args.friction)
    elif args.density:
        geom.set("density", args.density)

if args.center_of_mass:
    set_mass_distribution(object_location,
                          np.array(args.center_of_mass.split(), dtype=float))
else:
    tree.write(object_location)

# ——— now load & run your eval ———
method     = args.method
split      = args.split
env_name   = "EnvApple-v0-var-scale"
model_path = args.model_path
log_dir    = os.path.dirname(model_path)

with open("rsl_code/config.yaml") as f:
    cfg = yaml.safe_load(f)
# flatten runner settings
runner_cfg = cfg.pop("runner", {})
runner_cfg["run_name"] = runner_cfg.get("run_name", "irrelevant")
cfg.update(runner_cfg)
cfg["env_name"]    = env_name
cfg["gemini_traj"] = True

# create envs
num_envs = 1000
envs = gym.make(
    env_name,
    num_envs=num_envs,
    render_mode="rgb_array",
    method=method,
    use_wrist=True,
    start_limit=10,
    control_mode="pd_joint_pos",
    split=split,
    model_scale=float(args.scale) if args.scale else 1.5
)

envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True)
vec  = RslRlVecEnvWrapper(envs)

scene = envs.base_env.scene.actors["apple_0"]

runner = OnPolicyRunner(vec, cfg, log_dir=log_dir, device="cuda")
runner.load(model_path)

# seeds
torch.manual_seed(0)
np.random.seed(0)

policy = runner.get_noisy_inference_policy()
obs, _ = vec.reset(seed=0)
success = np.zeros(num_envs, dtype=bool)
max_net_contact_force_on_object = np.array([0, 0, 0], dtype=np.float32)
max_net_contact_force_norm = 0.0

with torch.inference_mode():
    for _ in tqdm.trange(envs.unwrapped.max_episode_steps):
        act = policy(obs)
        obs, _, _, _ = vec.step(act)
        success |= vec.unwrapped.is_success().cpu().numpy()

        current_bottle_net_contact_forces = envs.base_env.scene.actors["apple_0"].get_net_contact_forces().cpu().numpy()
        net_contact_forces_norms = np.linalg.norm(current_bottle_net_contact_forces, axis=-1)

        largest_net_force_idx = np.argmax(net_contact_forces_norms)

        if net_contact_forces_norms[largest_net_force_idx] > max_net_contact_force_norm:
            max_net_contact_force_on_object = current_bottle_net_contact_forces[largest_net_force_idx]
            max_net_contact_force_norm = net_contact_forces_norms[largest_net_force_idx]


success_rate = success.mean()

# record result
row = {
    **flags,
    "max_contact_force": np.array2string(max_net_contact_force_on_object, separator=" ")[1:-1],
    "success_rate": success_rate,
}
results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
results_df.to_csv(results_file, index=False)

# restore original
shutil.copyfile(tmp_location, object_location)
os.remove(tmp_location)
