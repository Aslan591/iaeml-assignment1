# Assignment 1 — Environment

Author: Aslan Iakupov

## Vehicle dynamics

The car's state includes its position, heading, and forward speed. The two action values are acceleration, limited to [-3, 3], and steering angle, limited to [-0.6, 0.6] radians.

The environment uses a kinematic bicycle model with a wheelbase of 0.6 map units. Each step updates the speed by `dt * (acceleration - 0.5 * speed)` and clips it to [0, 4]. The term `0.5 * speed` models resistance, so the car gradually slows down when the accelerator is released. The heading changes by `dt * speed * tan(steering) / wheelbase`, and the car then moves `dt * speed` in the new direction. This model gives the car a turning radius and lets it build up speed gradually with only a few state variables. It supports forward motion and braking.

## Observation and reward

The agent sees its speed, the goal position and path direction in the car's coordinate frame, the distance to the path, and one obstacle distance per ray. The demo uses 32 rays. The local coordinates show where to turn relative to the current heading, and the rays detect both static and moving obstacles.

On a regular step, the reward is the decrease in straight-line distance to the goal minus 0.01. The distance term gives feedback at each step, and the step cost favors shorter episodes. Reaching the goal gives +10. A collision or leaving the map gives -10. The episode also ends when it reaches the step limit.

## Reflection

The trickiest part for me to understand was how the moving obstacles affect the route. A short route can cross an obstacle's path at the wrong moment. The demo handles this with intermediate waypoints that give the car room to turn and let it cross at a different point. That made the limits of the reward clearer to me. A turn around an obstacle can briefly take the car farther from the goal, even when it is the right move. The distance term gives useful feedback on progress, while the collision penalty makes that shortcut costly.

## Demo and reproduction

The [demo video](stage1-demo.mp4) shows a controller following fixed waypoints on map 1. The car reaches the goal in 698 steps with a terminal reward of +10.

From the repository root, with a Python environment activated:

```bash
python -m pip install jax equinox chex pygame imageio-ffmpeg
python src/demo_robotaxi.py --record stage1-demo.mp4
```

Run `python src/demo_robotaxi.py` to drive manually. Use the arrow keys or WASD to accelerate, brake, and steer. Press R to reset the episode.
