import numpy as np
import atexit
import select
import sys
import termios
import tty
from threading import Lock, Thread
from collections import deque
import torch as th 

class FixedVelocityCommand:
    def __init__(self, env_cfg, device, x_vel: float | None = None):
        self._device = device
        commands_cfg = env_cfg.commands.base_velocity
        lin_vel_x_range = commands_cfg.ranges.lin_vel_x
        if x_vel is None:
            x_vel = 0.5 * (lin_vel_x_range[0] + lin_vel_x_range[1])
        self._velocity_cmd = th.tensor([[float(x_vel), 0.0, 0.0]], device=self._device)
        print(f"[INFO] No joystick mode: fixed velocity command x={float(x_vel):.2f} m/s")

    def start_listening(self):
        return None

    def reset(self):
        return None

    @property
    def velocity_cmd(self):
        return self._velocity_cmd

    def close(self):
        return None

    def pop_key(self):
        return None


class KeyboardVelocityCommand:
    def __init__(self, device, initial_x: float | None = None):
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard control needs an interactive terminal.")
        self._device = device
        self._step = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        self._limit = np.array([0.4, 0.3, 0.6], dtype=np.float32)
        self._velocity_cmd = np.array([[float(initial_x or 0.0), 0.0, 0.0]], dtype=np.float32)
        self._velocity_cmd[0] = np.clip(self._velocity_cmd[0], -self._limit, self._limit)
        self._lock = Lock()
        self._stopping = False
        self._listening_thread = None
        self._last_key = ""
        self._key_events = deque(maxlen=20)
        self._fd = sys.stdin.fileno()
        self._old_terminal_settings = termios.tcgetattr(self._fd)
        atexit.register(self.close)
        print("[INFO] Keyboard control enabled: W/S or Up/Down=x, A/D or Left/Right=y, Q/E=yaw, Space=zero")
        self._print_command()

    def start_listening(self):
        tty.setcbreak(self._fd)
        self._listening_thread = Thread(target=self.listen, daemon=True)
        self._listening_thread.start()

    def listen(self):
        while not self._stopping:
            key = self._read_key()
            if not key:
                self._last_key = ""
                continue
            if key == self._last_key:
                continue
            self._last_key = key
            with self._lock:
                self._key_events.append(key)
            self._apply_key(key)

    def _read_key(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.08)
        if not readable:
            return ""
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return ch
        readable, _, _ = select.select([sys.stdin], [], [], 0.01)
        if not readable:
            return ""
        ch = sys.stdin.read(1)
        if ch != "[":
            return ""
        readable, _, _ = select.select([sys.stdin], [], [], 0.01)
        if not readable:
            return ""
        ch = sys.stdin.read(1)
        return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch, "")

    def _apply_key(self, key):
        delta = np.zeros(3, dtype=np.float32)
        if key in ("w", "W", "up"):
            delta[0] = self._step[0]
        elif key in ("s", "S", "down"):
            delta[0] = -self._step[0]
        elif key in ("a", "A", "left"):
            delta[1] = self._step[1]
        elif key in ("d", "D", "right"):
            delta[1] = -self._step[1]
        elif key in ("q", "Q"):
            delta[2] = self._step[2]
        elif key in ("e", "E"):
            delta[2] = -self._step[2]
        elif key == " ":
            with self._lock:
                self._velocity_cmd[:] = 0.0
            self._print_command()
            return
        else:
            return

        with self._lock:
            self._velocity_cmd[0] = np.clip(self._velocity_cmd[0] + delta, -self._limit, self._limit)
        self._print_command()

    def _print_command(self):
        with self._lock:
            cmd = self._velocity_cmd[0].copy()
        print(f"[Keyboard velocity] x={cmd[0]:.2f}, y={cmd[1]:.2f}, yaw={cmd[2]:.2f}")

    def reset(self):
        return None

    @property
    def velocity_cmd(self):
        with self._lock:
            command = self._velocity_cmd.copy()
        return th.tensor(command, device=self._device)

    def pop_key(self):
        with self._lock:
            if not self._key_events:
                return None
            return self._key_events.popleft()

    def close(self):
        if getattr(self, "_stopping", True):
            return
        self._stopping = True
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, self._old_terminal_settings)
        except termios.error:
            pass


class MujocoJoystick:
    """
    Reference from 
    """
    def __init__(self, env_cfg, device):
        self._init_joystick()
        self.x_vel = 0
        self.y_vel = 0
        self.yaw = 0
        self._device = device
        commands_cfg = env_cfg.commands.base_velocity
        self._resampling_time_range = commands_cfg.resampling_time_range[-1]
        self._small_commands_to_zero = commands_cfg.small_commands_to_zero
        self._lin_vel_x_range = commands_cfg.ranges.lin_vel_x
        self._heading = commands_cfg.ranges.heading
        self._lin_vel_clip = commands_cfg.clips.lin_vel_clip
        self._ang_vel_clip = commands_cfg.clips.ang_vel_clip
        self._stopping = False
        self._listening_thread = None



    def _init_joystick(self, device_id=0):
        """
        We are only support gamepad joystick type
        """
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            raise RuntimeError("No gamepad detected. Run without --use_joystick or connect a gamepad.")
        print(f"[INFO] Initialized {self.joystick.get_name()}")
        print(f"[INFO] Joystick power level {self.joystick.get_power_level()}")
        buffer_length = 10
        self._x_buffer = deque([0] * buffer_length, buffer_length)
        self._velocity_cmd = np.zeros((1,3))

    def start_listening(self):
        self._listening_thread = Thread(target=self.listen, daemon=True)
        self._listening_thread.start()

    def listen(self):
        while not self._stopping:
            pygame = self._pygame
            pygame.event.pump()
            x_input = (self.joystick.get_axis(0))  * (self._lin_vel_x_range[1] - self._lin_vel_x_range[0]) + self._lin_vel_x_range[0]
            self._x_buffer.append(x_input)
            self.x_vel = np.median(self._x_buffer)
            self._velocity_cmd[:] = np.array([[self.x_vel, self.y_vel, self.yaw]])
            pygame.time.wait(10)


    def reset(self):
        self._x_buffer.clear()

    @property 
    def velocity_cmd(self): 
        if self._small_commands_to_zero:
            self._velocity_cmd[:,:2] *= np.abs(self._velocity_cmd[:, 0:1]) \
                                            > self._lin_vel_clip
        return th.tensor(self._velocity_cmd).to(self._device)

    def close(self):
        self._stopping = True

    def pop_key(self):
        return None
