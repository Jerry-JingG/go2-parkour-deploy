import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import parkour_isaaclab
from scripts.utils import load_local_cfg
from core.deployment_player import DeploymentPlayer
# import multiprocessing as mp

def main(args):
    """Play with RSL-RL agent."""
    logs_path = '/'
    for path in parkour_isaaclab.__path__[0].split('/')[1:-1]:
        logs_path = os.path.join(logs_path, path)
    logs_path = os.path.join(logs_path,'logs',args.rl_lib,args.task, args.expid)
    # model_path = os.path.join(logs_path, f'{args.model_id}.pt')
    cfgs_path = os.path.join(logs_path, 'params')
    env_cfg = load_local_cfg(cfgs_path, 'env')
    agent_cfg = load_local_cfg(cfgs_path, 'agent')
    env_cfg.scene.num_envs = 1
    
    player = DeploymentPlayer(
        env_cfg=env_cfg,
        agent_cfg = agent_cfg, 
        network_interface= args.interface,
        logs_path = logs_path, 
        use_joystick=args.use_joystick,
        use_keyboard=args.keyboard_control,
        command_x=args.command_x,
        show_depth=args.show_depth,
        manual_start=args.keyboard_control,
        record_csv=args.record_csv,
        record_depth_dir=args.record_depth_dir,
        record_depth_every=args.record_depth_every,
        record_foot_force_threshold=args.record_foot_force_threshold,
        action_lpf_alpha=args.action_lpf_alpha,
        action_delta_limit=args.action_delta_limit,
        action_clip=args.action_clip,
        stand_record_seconds=args.stand_record_seconds,
    )
    
    player.reset(maximum_iteration = args.n_eval)
    while player.alive():
        _, terminated, timeout, extras = player.play()
        if terminated or timeout:
           player.reset(extras = extras)
    print('Eval Done')
    
    sys.exit()

if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-09-03_12-07-56')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=False)
    parser.add_argument("--keyboard_control", action='store_true', default=False)
    parser.add_argument("--command_x", type=float, default=None)
    parser.add_argument("--show_depth", action='store_true', default=False)
    parser.add_argument("--n_eval", type=int, default=10)
    parser.add_argument("--record_csv", type=str, default=None)
    parser.add_argument("--record_depth_dir", type=str, default=None)
    parser.add_argument("--record_depth_every", type=int, default=5)
    parser.add_argument("--record_foot_force_threshold", type=float, default=2.0)
    parser.add_argument("--action_lpf_alpha", type=float, default=0.5)
    parser.add_argument("--action_delta_limit", type=float, default=0.25)
    parser.add_argument("--action_clip", type=float, default=4.8)
    parser.add_argument("--stand_record_seconds", type=float, default=5.0)
    args = parser.parse_args()
    main(args)
