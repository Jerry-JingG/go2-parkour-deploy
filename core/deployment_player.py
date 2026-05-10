import torch as th
from mujoco_deploy.mujoco_wrapper import MujocoWrapper
from mujoco_deploy.mujoco_trace_recorder import MujocoTraceRecorder
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
        record_csv: str | None = None,
        record_depth_dir: str | None = None,
        record_depth_every: int = 5,
        record_foot_force_threshold: float = 2.0,
        action_lpf_alpha: float = 0.5,
        action_delta_limit: float = 0.25,
        action_clip: float = 4.8,
        stand_record_seconds: float = 5.0,
    ):
        if not (0.0 < action_lpf_alpha <= 1.0):
            raise ValueError("action_lpf_alpha must be in (0, 1]")
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
        self._recorder = None
        self._prev_policy_action = None
        self._action_lpf_alpha = float(action_lpf_alpha)
        self._action_delta_limit = float(action_delta_limit)
        self._action_clip = float(action_clip)
        self._policy_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
        self._stand_record_seconds = max(0.0, float(stand_record_seconds))
        self._stand_record_remaining = 0
        self._stand_record_total = 0

        self._clip_actions = agent_cfg['clip_actions']
        estimator_paras = agent_cfg["estimator"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.num_priv_explicit = estimator_paras["num_priv_explicit"]
        self.history_len = 10 
        self.cnt = 0 
        self._call_cnt = 0
        self._maximum_iteration = float('inf')
        if record_csv:
            depth_max_distance = 2.0
            if self._use_camera:
                depth_max_distance = float(env_cfg.scene.depth_camera.max_distance)
            self._recorder = MujocoTraceRecorder(
                record_csv,
                depth_dir=record_depth_dir,
                depth_every=record_depth_every,
                foot_force_threshold=record_foot_force_threshold,
                depth_max_distance=depth_max_distance,
            )
        if not self._manual_start:
            self._load_policy()
        else:
            self._mode = "passive"
            print("[INFO] Manual FSM: press 0=FixStand, 1=Policy, r=record stand burst, p/9=Passive")

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
            self._record_manual_burst_if_active()
            return None, False, False, {}
        if self._mode == "fixstand":
            self.env.hold_stand_step()
            self._record_manual_burst_if_active()
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
            elif key in ("r", "R"):
                self._start_stand_record_burst()

    def _start_stand_record_burst(self):
        if self._stand_record_seconds <= 0.0:
            print("[Record] Stand burst disabled (--stand_record_seconds <= 0)")
            return
        if self._recorder is None:
            print("[Record] Start with --record before using r=record stand burst")
            return
        if not self._use_camera:
            print("[Record] Stand burst needs the depth-camera policy config")
            return
        steps = max(1, int(round(self._stand_record_seconds / self._policy_dt)))
        self._stand_record_remaining = steps
        self._stand_record_total = steps
        print(
            f"[Record] Stand burst armed: {self._stand_record_seconds:.2f}s "
            f"({steps} policy steps) in mode={self._mode}"
        )

    def _record_manual_burst_if_active(self):
        if self._stand_record_remaining <= 0:
            return
        self._record_manual_sample()
        self._stand_record_remaining -= 1
        recorded = self._stand_record_total - self._stand_record_remaining
        if recorded == 1 or recorded % 50 == 0:
            print(f"[Record] Stand burst {recorded}/{self._stand_record_total}")
        if self._stand_record_remaining == 0:
            self.env.common_step_counter = self.cnt
            print("[Record] Stand burst complete")

    def _record_manual_sample(self):
        self._load_policy()
        self.env.common_step_counter = self.cnt
        obs, extras = self.env.get_observations()
        proprio_obs = obs[:, :self.num_prop].clone()
        zero_action = th.zeros_like(self.env._actions, dtype=obs.dtype, device=obs.device)
        depth_yaw = th.zeros((1, 2), dtype=obs.dtype, device=obs.device)
        policy_obs_6_8 = th.zeros((1, 2), dtype=obs.dtype, device=obs.device)
        if self._use_camera:
            with th.inference_mode():
                depth_image = extras["observations"]["depth_camera"]
                proprioception = proprio_obs.clone()
                proprioception[:, 6:8] = 0
                depth_latent_and_yaw = self.depth_encoder(depth_image, proprioception)
                self.depth_latent = depth_latent_and_yaw[:, :-2]
                self.yaw = depth_latent_and_yaw[:, -2:]
                depth_yaw = self.yaw.clone()
                policy_obs_6_8 = 1.5 * self.yaw
        self._recorder.write(
            policy_step_index=self.cnt,
            proprio_obs=proprio_obs,
            policy_action=zero_action,
            depth_yaw=depth_yaw,
            policy_obs_6_8=policy_obs_6_8,
            foot_force_isaac=self.env.get_isaac_foot_force_norms(),
            depth_trace=self.env.get_depth_trace() if self._use_camera else None,
        )
        self.cnt += 1

    def _play_policy(self):
        self._load_policy()
        obs, extras = self.env.get_observations()
        proprio_obs = obs[:, :self.num_prop].clone()
        with th.inference_mode():
            if not self._use_camera:
                actions = self.policy(obs , hist_encoding=True)
                depth_yaw = th.zeros((1, 2), dtype=obs.dtype, device=obs.device)
            else:
                if self.env.common_step_counter %5 == 0:
                    depth_image = extras["observations"]['depth_camera']
                    proprioception = proprio_obs.clone()
                    proprioception[:, 6:8] = 0
                    depth_latent_and_yaw = self.depth_encoder(depth_image , proprioception )
                    self.depth_latent = depth_latent_and_yaw[:, :-2]
                    self.yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*self.yaw
                depth_yaw = self.yaw.clone()
                actions = self.policy(obs , scandots_latent=self.depth_latent)
        if self._clip_actions is not None:
            actions = th.clamp(actions, -self._clip_actions, self._clip_actions)
        actions = self._filter_policy_action(actions)
        if self._recorder is not None:
            self._recorder.write(
                policy_step_index=self.cnt,
                proprio_obs=proprio_obs,
                policy_action=actions,
                depth_yaw=depth_yaw,
                policy_obs_6_8=obs[:, 6:8],
                foot_force_isaac=self.env.get_isaac_foot_force_norms(),
                depth_trace=self.env.get_depth_trace() if self._use_camera else None,
            )
        obs, terminated, timeout, extras = self.env.step(actions)  # For HW, this internally just does forward

        self.cnt += 1
        return obs, terminated, timeout, extras

    def _filter_policy_action(self, action: th.Tensor) -> th.Tensor:
        original_shape = action.shape
        filtered = action.reshape(-1)
        if self._prev_policy_action is None or self._prev_policy_action.shape != filtered.shape:
            self._prev_policy_action = th.zeros_like(filtered)

        alpha = self._action_lpf_alpha
        if alpha < 1.0:
            filtered = alpha * filtered + (1.0 - alpha) * self._prev_policy_action

        if self._action_delta_limit > 0.0:
            delta = th.clamp(
                filtered - self._prev_policy_action,
                -self._action_delta_limit,
                self._action_delta_limit,
            )
            filtered = self._prev_policy_action + delta

        if self._action_clip > 0.0:
            filtered = th.clamp(filtered, -self._action_clip, self._action_clip)

        self._prev_policy_action = filtered.detach()
        return filtered.reshape(original_shape)
    
    def reset(self, maximum_iteration: int |None = None, extras: Dict[str, str] | None = None):
        self._call_cnt +=1 
        if type(maximum_iteration) == int:
            self.maximum_iteration = maximum_iteration
        if self.alive():
            self._prev_policy_action = None
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
            if self._recorder is not None:
                self._recorder.close()
            self.env.close()
            return False 
        
