"""Gate crossing navigation environment."""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dva_quadrotor_mjx.envs.hover import HoveringStateEnv
from dva_quadrotor_mjx.utils.math import smooth_l1


class GateCrossingEnv(HoveringStateEnv):
    """Gate crossing navigation task.

    Drone must fly through a sequence of gates.
    Success: crossing gate plane from entry to exit side.
    """

    def __init__(
        self,
        *,
        max_steps_in_episode=2000,
        dt=0.02,
        delay=0.02,
        yaw_scale=0.1,
        pitch_roll_scale=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        reward_sharpness=1.0,
        action_penalty_weight=1.0,
        margin=0.5,
        num_gates=3,
        **kwargs
    ):
        super().__init__(
            max_steps_in_episode=max_steps_in_episode,
            dt=dt,
            delay=delay,
            yaw_scale=yaw_scale,
            pitch_roll_scale=pitch_roll_scale,
            velocity_std=velocity_std,
            omega_std=omega_std,
            reward_sharpness=reward_sharpness,
            action_penalty_weight=action_penalty_weight,
            margin=margin,
            **kwargs
        )
        self.num_gates = num_gates
        # Gate positions from XML
        self.gate_positions = jnp.array([
            [2.0, 0.0, 1.0],
            [4.0, 2.0, 1.0],
            [6.0, 0.0, 1.0],
        ])
        self.gate_width = 1.0
        self.gate_height = 1.0

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info["current_gate"] = jnp.array(0, dtype=jnp.int32)
        info["gates_crossed"] = jnp.array(0, dtype=jnp.int32)
        return state.replace(info=info)

    def step(self, state, action):
        next_state = super().step(state, action)

        p = next_state.pipeline_state.qpos[0:3]
        current_gate = state.info["current_gate"]

        # Check if crossed current gate
        gate_pos = self.gate_positions[current_gate]
        # Gate plane is at x = gate_pos[0]
        prev_p = state.pipeline_state.qpos[0:3]
        crossed = jnp.logical_and(
            prev_p[0] < gate_pos[0],
            p[0] >= gate_pos[0]
        )
        # Check within gate bounds
        in_bounds = jnp.logical_and(
            jnp.abs(p[1] - gate_pos[1]) < self.gate_width / 2,
            jnp.abs(p[2] - gate_pos[2]) < self.gate_height / 2
        )
        gate_success = jnp.logical_and(crossed, in_bounds)

        # Reward for approaching next gate
        target = jax.lax.select(
            current_gate < self.num_gates,
            self.gate_positions[jnp.minimum(current_gate, self.num_gates - 1)],
            self.gate_positions[-1]
        )
        dist_to_target = jnp.linalg.norm(p - target)
        approach_reward = -0.01 * dist_to_target

        # Gate crossing bonus
        gate_bonus = jax.lax.select(gate_success, 10.0, 0.0)

        new_gate = jnp.minimum(
            current_gate + jnp.where(gate_success, 1, 0),
            self.num_gates
        )

        reward = next_state.reward + approach_reward + gate_bonus

        info = dict(next_state.info)
        info["current_gate"] = new_gate
        info["gates_crossed"] = state.info["gates_crossed"] + jnp.where(gate_success, 1, 0)
        info["gate_success"] = gate_success

        # Done if all gates crossed
        all_crossed = new_gate >= self.num_gates
        done = jnp.where(all_crossed, 1.0, next_state.done)

        return next_state.replace(
            reward=reward,
            done=done,
            info=info
        )
