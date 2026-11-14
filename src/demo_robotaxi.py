import argparse
import math
import os
from pathlib import Path

import jax
from jax import numpy as jnp
import pygame

from env.robotaxi import FRICTION, MAX_ACCELERATION, MAX_STEERING, RobotaxiEnv
from utils.renderer import PygameFrontend


class RobotaxiFrontend(PygameFrontend):
    def handle_keys(self):
        keys = pygame.key.get_pressed()
        throttle = int(keys[pygame.K_UP] or keys[pygame.K_w])
        brake = int(keys[pygame.K_DOWN] or keys[pygame.K_s])
        left = int(keys[pygame.K_LEFT] or keys[pygame.K_a])
        right = int(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        return jnp.array([
            (throttle - brake) * MAX_ACCELERATION,
            (left - right) * MAX_STEERING,
        ])


def route_action(state, target):
    dx, dy = target - state.agent_pos
    fx, fy = state.agent_forward_dir
    error = math.atan2(float(fx * dy - fy * dx), float(fx * dx + fy * dy))
    target_speed = 2.2 / (1 + abs(error))
    speed = float(state.speed)
    acceleration = 3 * (target_speed - speed) + FRICTION * speed
    return jnp.array([1.2 * acceleration, 1.5 * error], dtype=jnp.float32)


def record(frontend, output):
    import imageio_ffmpeg

    video = imageio_ffmpeg.write_frames(
        str(output), frontend.screen.get_size(), fps=30, codec="libx264",
        macro_block_size=1,
    )
    targets = jnp.array([[2, 6.5], [7, 6.5], [9, 3], [11, 4], [15, 3], [16, 1]])
    waypoint = 0
    video.send(None)
    try:
        while True:
            pygame.event.pump()
            distance = jnp.linalg.norm(targets[waypoint] - frontend.state.agent_pos)
            if waypoint < len(targets) - 1 and float(distance) < 0.45:
                waypoint += 1
            action = route_action(frontend.state, targets[waypoint])
            obs, state, reward, done, info = frontend.env.step(
                frontend.key, frontend.state, action, frontend.params
            )
            frontend.obs, frontend.state, frontend.info = obs, state, info
            if int(state.time) % 2 == 0 or bool(done):
                frontend.draw()
                video.send(pygame.image.tobytes(frontend.screen, "RGB"))
            if bool(done):
                print(f"Steps: {state.time}, goal: {info['reached_goal']}, reward: {reward}")
                if not bool(info["reached_goal"]):
                    raise RuntimeError(f"Demo ended at {state.agent_pos}")
                break
    finally:
        video.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    key = jax.random.PRNGKey(0)
    params, state = RobotaxiEnv.init_params(
        key=key, map_id=1, max_steps=1200, path_length=100, num_ray_sensors=32,
    )
    frontend = RobotaxiFrontend(RobotaxiEnv, params, state)
    try:
        if args.record:
            record(frontend, args.record)
        else:
            while frontend.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        frontend.running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        frontend.obs, frontend.state = RobotaxiEnv.reset(key, params, state)
                action = frontend.handle_keys()
                frontend.obs, frontend.state, _, done, frontend.info = RobotaxiEnv.step(
                    key, frontend.state, action, params
                )
                frontend.draw()
                if bool(done):
                    frontend.obs, frontend.state = RobotaxiEnv.reset(key, params, state)
                frontend.clock.tick(params.fps)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
