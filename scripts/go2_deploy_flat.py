import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import core
import parkour_isaaclab
from core.deployment_player import DeploymentPlayer
from scripts.utils import load_local_cfg


def main(args):
    logs_path = '/'
    for path in parkour_isaaclab.__path__[0].split('/')[1:-1]:
        logs_path = os.path.join(logs_path, path)
    logs_path = os.path.join(logs_path, 'logs', args.rl_lib, args.task, args.expid)
    cfgs_path = os.path.join(logs_path, 'params')
    env_cfg = load_local_cfg(cfgs_path, 'env')
    agent_cfg = load_local_cfg(cfgs_path, 'agent')
    env_cfg.scene.num_envs = 1

    scene_xml = os.path.join(core.__path__[0], 'go2/scene_flat.xml')
    player = DeploymentPlayer(
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        network_interface=args.interface,
        logs_path=logs_path,
        use_joystick=args.use_joystick,
        use_keyboard=args.keyboard_control,
        command_x=args.command_x,
        show_depth=args.show_depth,
        model_xml_path=scene_xml,
        manual_start=args.keyboard_control,
    )

    player.reset(maximum_iteration=args.n_eval)
    while player.alive():
        _, terminated, timeout, extras = player.play()
        if terminated or timeout:
            player.reset(extras=extras)
    print('Eval Done')
    sys.exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='sim_2_sim_flat')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-09-03_12-07-56')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=False)
    parser.add_argument("--keyboard_control", action='store_true', default=False)
    parser.add_argument("--command_x", type=float, default=None)
    parser.add_argument("--show_depth", action='store_true', default=False)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()
    main(args)
