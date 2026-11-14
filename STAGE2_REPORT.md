# Assignment 2 — Reinforcement Learning

Author: Aslan Iakupov

## Experimental setup

The experiment uses the JAX robotaxi environment from Assignment 1. Each episode starts on map 1 with the same initial car position and moving obstacles. An episode runs for at most 1,200 steps and ends earlier on a goal, a collision, or an exit from the map. The objective is to maximize the sum of the environment rewards over an episode.

The policy starts with the fixed waypoint controller from Assignment 1. A trainable correction adjusts its acceleration and steering. The correction uses six inputs: a constant bias, normalized speed, the sine of the heading error, and obstacle proximity ahead, to the left, and to the right. A 2-by-6 weight matrix maps these features to the two corrections. The hyperbolic tangent bounds the corrections, and the environment clips the final actions to its limits. Zero weights reproduce the baseline.

## Learning algorithm

The cross-entropy method (CEM) trains the policy from episode returns. Each iteration evaluates 32 policies: the best policy found so far, the current mean policy, and 30 samples from a Gaussian distribution over the weights. The eight policies with the highest returns set the mean and standard deviation for the next iteration. A minimum standard deviation of 0.01 keeps some exploration. Training uses 12 iterations and random seed 42. The checkpoint stores the weights with the best observed return.

This is a derivative-free policy search method. The environment supplies the reward after each action, and the total return determines which policies influence the next update.

## Computational optimizations

JAX batches the candidate policies with `vmap`, compiles the batched evaluator with `jit`, and runs each rollout with `lax.scan`. This keeps the per-step loop out of Python. All rollouts have the same static length. A mask excludes rewards and step counts after the first terminal transition, although the simulator still executes the remaining steps in the batch. Rendering runs separately when replaying the saved policy.

## Results

| Policy | Reached the goal | Steps | Episode return |
| --- | --- | ---: | ---: |
| Fixed waypoint baseline | Yes | 698 | 16.021 |
| CEM policy | Yes | 553 | 17.478 |

A separate replay from the saved checkpoint reached the goal in 553 steps with a return of 17.479. Batched training and single-episode replay differ slightly in floating-point accumulation.

The learned correction reduced the number of steps by 145, about 21%, on this scenario. The comparison uses the training map and its fixed initial state. It measures improvement on that route, and generalization to other maps and starting states remains untested.

## Reflection

The difficult part for me to understand was why a small change in steering or speed could cause a collision even when the car followed the same waypoints. The first batch showed this clearly: only 13 of the 32 candidates reached the goal. Starting with a controller that already completed the route gave me a baseline for judging whether the learned correction helped. I also found it important to stop counting rewards after termination. The compiled loop has a fixed length, so without the mask a policy could receive credit for transitions that happened after the episode had already ended.

## Checkpoint and reproduction

[Demo video](stage2-demo.mp4) · [Saved policy](stage2-policy.npz)

From the repository root, install the dependencies in an active Python environment. The experiment was run with Python 3.12 and CPU JAX.

```bash
python -m pip install jax==0.12.0.1 equinox==0.13.8 chex==0.1.92 pygame==2.6.1 imageio-ffmpeg==0.8.1
python src/train_robotaxi.py --replay
python src/train_robotaxi.py --record stage2-demo.mp4
```

To rerun training and save a separate checkpoint:

```bash
python src/train_robotaxi.py --checkpoint stage2-retrained.npz
python src/train_robotaxi.py --replay --checkpoint stage2-retrained.npz
```

