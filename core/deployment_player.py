import torch as th
from mujoco_deploy.mujoco_wrapper import MujocoWrapper
import os
import core
from typing import Dict

class DeploymentPlayer:
    def __init__(
        self,
        env_cfg,
        agent_cfg, 
        network_interface,
        logs_path,
        use_joystick: bool = False,
        use_keyboard: bool = False,
        command_x: float | None = None,
        show_depth: bool = False,
        model_xml_path: str | None = None,
        manual_start: bool = False,
    ):
        try: 
            env_cfg.scene.depth_camera
            use_camera = True 
        except: 
            use_camera = False 

        if network_interface.lower() =='lo':
            if model_xml_path is None:
                model_xml_path = os.path.join(core.__path__[0], 'go2/scene_parkour.xml')
            self.env = MujocoWrapper(
                env_cfg,
                agent_cfg,
                model_xml_path,
                use_camera,
                use_joystick=use_joystick,
                use_keyboard=use_keyboard,
                command_x=command_x,
                show_depth=show_depth,
            )
        self._use_camera = use_camera
        self._logs_path = logs_path
        self._manual_start = manual_start
        self._mode = "auto"
        self.policy = None
        self.depth_encoder = None

        self._clip_actions = agent_cfg['clip_actions']
        estimator_paras = agent_cfg["estimator"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.num_priv_explicit = estimator_paras["num_priv_explicit"]
        self.history_len = 10 
        self.cnt = 0 
        self._call_cnt = 0
        self._maximum_iteration = float('inf')
        if not self._manual_start:
            self._load_policy()
        else:
            self._mode = "passive"
            print("[INFO] Manual FSM: press 0=FixStand, 1=Policy, p/9=Passive")

    def _load_policy(self):
        if self.policy is not None:
            return
        if self._use_camera:
            policy_path = os.path.join(self._logs_path, 'exported_deploy', 'policy.pt')
            depth_path = os.path.join(self._logs_path, 'exported_deploy', 'depth_latest.pt')
            print(f"[INFO] Loading deploy policy: {policy_path}")
            self.policy = th.jit.load(policy_path, map_location=self.env.device)
            self.policy.eval()
            print(f"[INFO] Loading depth encoder: {depth_path}")
            self.depth_encoder = th.jit.load(depth_path, map_location=self.env.device)
            self.depth_encoder.eval()
        else:
            policy_path = os.path.join(self._logs_path, 'exported_teacher', 'policy.pt')
            print(f"[INFO] Loading teacher policy: {policy_path}")
            self.policy = th.jit.load(policy_path, map_location=self.env.device)
            self.policy.eval()
            self.depth_encoder = None
        
    def play(self):
        """Advances the environment one time step after generating observations"""
        if self._manual_start:
            return self._play_manual()
        return self._play_policy()

    def _play_manual(self):
        self._handle_manual_keys()
        if self._mode == "passive":
            self.env.hold_passive_step()
            return None, False, False, {}
        if self._mode == "fixstand":
            self.env.hold_stand_step()
            return None, False, False, {}
        return self._play_policy()

    def _handle_manual_keys(self):
        while True:
            key = self.env.pop_command_key()
            if key is None:
                return
            if key == "0":
                if self._mode != "fixstand":
                    self.env.stand_up()
                    self._mode = "fixstand"
                print("[FSM] FixStand")
            elif key == "1":
                if self._mode == "passive":
                    print("[FSM] Press 0 to stand before entering policy")
                    continue
                self._load_policy()
                self._mode = "policy"
                print("[FSM] Policy")
            elif key in ("p", "P", "9"):
                if self._mode != "passive":
                    self.env.stand_down()
                self._mode = "passive"
                print("[FSM] Passive")

    def _play_policy(self):
        self._load_policy()
        obs, extras = self.env.get_observations()
        with th.inference_mode():
            if not self._use_camera:
                actions = self.policy(obs , hist_encoding=True)
            else:
                if self.env.common_step_counter %5 == 0:
                    depth_image = extras["observations"]['depth_camera']
                    proprioception = obs[:, :self.num_prop].clone()
                    proprioception[:, 6:8] = 0
                    depth_latent_and_yaw = self.depth_encoder(depth_image , proprioception )
                    self.depth_latent = depth_latent_and_yaw[:, :-2]
                    self.yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*self.yaw
                actions = self.policy(obs , scandots_latent=self.depth_latent)
        if self._clip_actions is not None:
            actions = th.clamp(actions, -self._clip_actions, self._clip_actions)
        obs, terminated, timeout, extras = self.env.step(actions)  # For HW, this internally just does forward

        self.cnt += 1
        return obs, terminated, timeout, extras
    
    def reset(self, maximum_iteration: int |None = None, extras: Dict[str, str] | None = None):
        self._call_cnt +=1 
        if type(maximum_iteration) == int:
            self.maximum_iteration = maximum_iteration
        if self.alive():
            if self._manual_start:
                self._mode = "passive"
                self.env.reset_passive()
            else:
                self.env.reset()
            print('[Current eval iter]: ', self._call_cnt, '[Left]: ', self.maximum_iteration-self._call_cnt)

    def alive(self):
        if self._call_cnt <= self.maximum_iteration:
            return True
        else:
            self.env.close()
            return False 
        
