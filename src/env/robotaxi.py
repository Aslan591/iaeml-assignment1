from functools import partial

import jax
from jax import numpy as jnp

from env.base import BaseEnv, BaseEnvObservation, BaseEnvState


MAX_ACCELERATION = 3.0
MAX_STEERING = 0.6
MAX_SPEED = 4.0
WHEELBASE = 0.6
FRICTION = 0.5


class RobotaxiState(BaseEnvState):
    speed: jax.Array


class RobotaxiObservation(BaseEnvObservation):
    speed: jax.Array
    goal_local: jax.Array
    path_local: jax.Array
    ray_distances: jax.Array

    def as_vector(self):
        return jnp.concatenate((
            jnp.atleast_1d(self.speed),
            self.goal_local,
            self.path_local,
            jnp.atleast_1d(self.distance_to_path),
            self.ray_distances,
        ))


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def init_params(*args, **kwargs):
        params, state = BaseEnv.init_params(*args, **kwargs)
        return params, RobotaxiState(**vars(state), speed=jnp.asarray(0.0))

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        return RobotaxiEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def get_observation(env_state, env_params):
        base = BaseEnv.get_observation(env_state, env_params)
        forward = env_state.agent_forward_dir
        left = jnp.array([-forward[1], forward[0]])
        axes = jnp.stack((forward, left))
        goal = env_state.goal_pos - env_state.agent_pos
        return RobotaxiObservation(
            distance_to_path=base.distance_to_path,
            direction_of_path=base.direction_of_path,
            collision_rays=base.collision_rays,
            speed=env_state.speed,
            goal_local=axes @ goal,
            path_local=axes @ base.direction_of_path,
            ray_distances=jnp.linalg.norm(base.collision_rays, axis=1),
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        dt = env_params.step_size
        acceleration = jnp.clip(action[0], -MAX_ACCELERATION, MAX_ACCELERATION)
        steering = jnp.clip(action[1], -MAX_STEERING, MAX_STEERING)
        speed = env_state.speed + dt * (acceleration - FRICTION * env_state.speed)
        speed = jnp.clip(speed, 0.0, MAX_SPEED)

        forward = env_state.agent_forward_dir
        heading = jnp.arctan2(forward[1], forward[0])
        heading += dt * speed * jnp.tan(steering) / WHEELBASE
        forward = jnp.array([jnp.cos(heading), jnp.sin(heading)])
        position = env_state.agent_pos + dt * speed * forward

        moving = BaseEnv._move_kinematic_obstacles(env_state, env_params)
        obstacles = jnp.concatenate((env_state.static_obstacles, moving))
        time = env_state.time + 1
        path = jax.lax.cond(
            time % env_params.fps == 0,
            lambda: BaseEnv._find_path(position, env_state.goal_pos, obstacles, env_params),
            lambda: env_state.path_array,
        )
        state = RobotaxiState(
            time=time,
            goal_pos=env_state.goal_pos,
            agent_pos=position,
            agent_forward_dir=forward,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=moving,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path,
            speed=speed,
        )

        height, width = env_params.map_height_width
        outside = jnp.any(position < 0) | jnp.any(position > jnp.array([width - 1, height - 1]))
        collision = outside | BaseEnv._check_collisions(position, obstacles)
        reached_goal = BaseEnv._check_goal(position, state.goal_pos) & ~collision
        timeout = time >= env_params.max_steps_in_episode
        done = collision | reached_goal | timeout

        old_distance = jnp.linalg.norm(env_state.agent_pos - state.goal_pos)
        distance = jnp.linalg.norm(position - state.goal_pos)
        reward = old_distance - distance - 0.01
        reward = jnp.where(collision, -10.0, jnp.where(reached_goal, 10.0, reward))
        obs = RobotaxiEnv.get_observation(state, env_params)
        info = {
            "time": time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "speed": speed,
            "reached_goal": reached_goal,
            "collision": collision,
            "timeout": timeout,
        }
        return obs, state, reward, done, info
