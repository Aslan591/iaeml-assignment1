import argparse
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import FRICTION, MAX_ACCELERATION, MAX_STEERING, RobotaxiEnv


TARGETS = jnp.array([[2, 6.5], [7, 6.5], [9, 3], [11, 4], [15, 3], [16, 1]])
BASELINE = jnp.zeros((2, 6), dtype=jnp.float32)
POPULATION = 32
ELITES = 8
ITERATIONS = 12


def policy(weights, obs, state, waypoint):
    distance = jnp.linalg.norm(TARGETS[waypoint] - state.agent_pos)
    advance = (waypoint < len(TARGETS) - 1) & (distance < 0.45)
    waypoint = waypoint + advance.astype(jnp.int32)
    offset = TARGETS[waypoint] - state.agent_pos
    fx, fy = state.agent_forward_dir
    dx, dy = offset
    angle = jnp.arctan2(fx * dy - fy * dx, fx * dx + fy * dy)
    target_speed = 2.2 / (1 + jnp.abs(angle))
    acceleration = 3 * (target_speed - obs.speed) + FRICTION * obs.speed

    half = len(obs.ray_distances) // 2
    left = jnp.min(obs.ray_distances[half + 1:half + 5])
    right = jnp.min(obs.ray_distances[half - 4:half])
    features = jnp.array([
        1.0, obs.speed / 4, jnp.sin(angle),
        1 - obs.ray_distances[half] / 5, 1 - left / 5, 1 - right / 5,
    ])
    base_action = jnp.array([1.2 * acceleration, 1.5 * angle])
    scales = jnp.array([MAX_ACCELERATION, MAX_STEERING])
    action = base_action + scales * jnp.tanh(weights @ features)
    return action, waypoint


def setup():
    key = jax.random.PRNGKey(0)
    params, start = RobotaxiEnv.init_params(
        key=key, map_id=1, max_steps=1200, path_length=100, num_ray_sensors=32,
    )
    obs, start = RobotaxiEnv.reset(key, params, start)
    return key, params, obs, start


def simulator(key, params, first_obs, start):
    def run(weights):
        def step(carry, _):
            obs, state, waypoint, total, finished, reached, steps = carry
            action, waypoint = policy(weights, obs, state, waypoint)
            obs, state, reward, done, info = RobotaxiEnv.step(key, state, action, params)
            total += jnp.where(finished, 0.0, reward)
            reached |= ~finished & info["reached_goal"]
            steps += (~finished).astype(jnp.int32)
            return (obs, state, waypoint, total, finished | done, reached, steps), None

        initial = (first_obs, start, 0, 0.0, False, False, 0)
        result, _ = jax.lax.scan(step, initial, None, length=params.max_steps_in_episode)
        return result[3], result[5], result[6]

    return jax.jit(jax.vmap(run))


def train(evaluate, checkpoint):
    rng = jax.random.PRNGKey(42)
    mean = BASELINE
    std = jnp.full_like(mean, 0.12)
    best = BASELINE
    best_score = -float("inf")
    for iteration in range(ITERATIONS):
        rng, sample_key = jax.random.split(rng)
        noise = jax.random.normal(sample_key, (POPULATION - 2, *mean.shape))
        candidates = jnp.concatenate((best[None], mean[None], mean + std * noise))
        returns, goals, steps = evaluate(candidates)
        elite_indices = jnp.argsort(returns)[-ELITES:]
        elite = candidates[elite_indices]
        mean = jnp.mean(elite, axis=0)
        std = jnp.maximum(jnp.std(elite, axis=0), 0.01)
        winner = int(elite_indices[-1])
        if iteration == 0:
            print(f"Baseline: return={float(returns[0]):.3f}, goal={bool(goals[0])}, steps={int(steps[0])}")
        if float(returns[winner]) > best_score:
            best = candidates[winner]
            best_score = float(returns[winner])
        print(f"Iteration {iteration + 1}: best return={best_score:.3f}, goals={int(goals.sum())}/{POPULATION}, best steps={int(steps[winner])}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint, weights=np.asarray(best), seed=42, map_id=1)


def replay(checkpoint, output):
    with np.load(checkpoint) as saved:
        weights = jnp.asarray(saved["weights"])
    key, params, obs, state = setup()
    action_fn = jax.jit(partial(policy, weights))
    waypoint = jnp.asarray(0)
    total = 0.0

    if output:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import imageio_ffmpeg
        import pygame
        from utils.renderer import PygameFrontend

        frontend = PygameFrontend(RobotaxiEnv, params, state)
        video = imageio_ffmpeg.write_frames(
            str(output), frontend.screen.get_size(), fps=30,
            codec="libx264", macro_block_size=1,
        )
        video.send(None)
    try:
        while True:
            action, waypoint = action_fn(obs, state, waypoint)
            obs, state, reward, done, info = RobotaxiEnv.step(key, state, action, params)
            total += float(reward)
            if output and (int(state.time) % 2 == 0 or bool(done)):
                pygame.event.pump()
                frontend.obs, frontend.state, frontend.info = obs, state, info
                frontend.draw()
                video.send(pygame.image.tobytes(frontend.screen, "RGB"))
            if bool(done):
                print(f"Replay: return={total:.3f}, goal={bool(info['reached_goal'])}, steps={int(state.time)}")
                break
    finally:
        if output:
            video.close()
            pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("stage2-policy.npz"))
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.replay or args.record:
        replay(args.checkpoint, args.record)
    else:
        train(simulator(*setup()), args.checkpoint)


if __name__ == "__main__":
    main()
