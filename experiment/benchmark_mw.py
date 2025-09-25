from mypolicy import MyPolicy_CL
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames
import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
from flowdiffusion.inference_utils import get_video_model, pred_video
import random
import torch
import wandb
import time
from argparse import ArgumentParser

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def get_policy(env_name):
    name = "".join(" ".join(get_task_text(env_name)).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

    
def run(args):
    start_time = time.time()

    wandb.init(
        project="Draft-and-Target-MetaWorld",  # Feel free to change the project name
        name=f"benchmark_{args.env_name}",
        config={
            "env_name": args.env_name,
            "n_exps": args.n_exps,
            "ckpt_dir": args.ckpt_dir,
            "milestone": args.milestone,
            "task": "manipulation"
        },
        tags=["metaworld", "AVDC"]
    )
    

    result_root = args.result_root
    os.makedirs(result_root, exist_ok=True)

    n_exps = args.n_exps
    resolution = (320, 240)
    cameras = ['corner', 'corner2', 'corner3']
    max_replans = 5

    video_model = get_video_model(ckpts_dir=args.ckpt_dir, milestone=args.milestone)

    flow_model = get_flow_model()
    
    try:
        with open(f"{result_root}/result_dict.json", "r") as f:
            result_dict = json.load(f)
    except:
        result_dict = {}


    env_name = args.env_name
    print(env_name)
    seg_ids = name2maskid[env_name]
    benchmark_env = env_dict[env_name]

    succes_rates = []
    reward_means = []
    reward_stds = []
    replans_counters = []
    
    total_experiments = len(cameras) * n_exps
    experiment_count = 0
    
    all_episode_results = []
    total_successful_episodes = 0
    
    for camera_idx, camera in enumerate(cameras):
        success = 0
        rewards = []
        replans_counter = {i: 0 for i in range(max_replans + 1)}
        
        for seed in tqdm(range(n_exps), desc=f"Camera {camera}"):
            experiment_count += 1
            episode_start_time = time.time()
            episode_success = False
            episode_length = 0
            used_replans = 0
            episode_reward = 0
            
            try: 
                env = benchmark_env(seed=seed)
                obs = env.reset()
                policy = MyPolicy_CL(env, env_name, camera, video_model, flow_model, max_replans=max_replans)

                images, _, episode_return = collect_video(obs, env, policy, camera_name=camera, resolution=resolution)
                
                episode_reward = episode_return / len(images)
                rewards.append(episode_reward)

                used_replans = max_replans - policy.replans
                episode_length = len(images)
                
                ### save sample video 
                # save video from the simulator
                os.makedirs(f'{result_root}/videos/{env_name}', exist_ok=True)
                imageio.mimsave(f'{result_root}/videos/{env_name}/{camera}_{seed}.mp4', images)

                print(f"Episode {experiment_count}: len={episode_length}, replans={used_replans}")
                if episode_length <= 500:
                    success += 1
                    episode_success = True
                    total_successful_episodes += 1
                    replans_counter[used_replans] += 1
                    print(f"✅ Success! Used replans: {used_replans}")
                else:
                    print("❌ Failed (episode too long)")
                
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                
                episode_result = {
                    'seed': seed,
                    'camera': camera,
                    'camera_idx': camera_idx,
                    'experiment_count': experiment_count,
                    'success': episode_success,
                    'length': episode_length,
                    'reward': episode_reward,
                    'replans_used': used_replans,
                    'duration_seconds': episode_duration,
                    'duration_minutes': episode_duration / 60
                }
                all_episode_results.append(episode_result)
                

            except Exception as e:
                print(f"❌ Episode {experiment_count} failed: {e}")
                print("Skipping this seed")
                
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                
                failed_episode_result = {
                    'seed': seed,
                    'camera': camera,
                    'camera_idx': camera_idx,
                    'experiment_count': experiment_count,
                    'success': False,
                    'length': 0,
                    'reward': 0,
                    'replans_used': 0,
                    'duration_seconds': episode_duration,
                    'duration_minutes': episode_duration / 60,
                    'failed': True
                }
                all_episode_results.append(failed_episode_result)
                continue

        rewards = rewards + [0] * (n_exps - len(rewards))
        reward_means.append(np.mean(rewards))
        reward_stds.append(np.std(rewards))

        success_rate = success / n_exps
        succes_rates.append(success_rate)

        
        replans_counters.append(replans_counter)
    
    end_time = time.time()
    total_duration = end_time - start_time
        
    episode_results_table = wandb.Table(columns=[
        "experiment_count", "seed", "camera", "camera_idx", "success", 
        "episode_length", "reward", "replans_used", "duration_minutes", "failed"
    ])
    
    for result in all_episode_results:
        episode_results_table.add_data(
            result['experiment_count'],
            result['seed'],
            result['camera'],
            result['camera_idx'],
            result['success'],
            result['length'],
            result['reward'],
            result['replans_used'],
            result['duration_minutes'],
            result.get('failed', False)
        )

    wandb.log({
        "episode_results_detailed": episode_results_table
    })
    

    print(f"\n=== Experiment Summary ===")
    print(f"Total duration: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
    print(f"Average time per experiment: {total_duration/total_experiments/60:.2f} minutes")
    print(f"Success rates for {env_name}: {succes_rates}")
    print(f"\n📁 Saved files:")
    print(f"  Full videos: {result_root}/videos/{env_name}/")
    print(f"  Results JSON: {result_root}/result_dict.json")
    
    result_dict[env_name] = {
        "success_rates": succes_rates,
        "reward_means": reward_means,
        "reward_stds": reward_stds,
        "replans_counts": replans_counters,
    }
    with open(f"{result_root}/result_dict.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env_name", type=str, default="door-open-v2-goal-observable")
    parser.add_argument("--n_exps", type=int, default=25)
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/metaworld")
    parser.add_argument("--milestone", type=int, default=24)
    parser.add_argument("--result_root", type=str, default="../results/results_AVDC_full")
    args = parser.parse_args()

    try:
        with open(f"{args.result_root}/result_dict.json", "r") as f:
            result_dict = json.load(f)
    except:
        result_dict = {}

    assert args.env_name in name2maskid.keys()
    if args.env_name in result_dict.keys():
        print("already done")
    else:
        run(args)
        