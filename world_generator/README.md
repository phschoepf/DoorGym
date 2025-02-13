## Basic steps to make door worlds.

## Step0: Download the doorknob data set
#### [Pull knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/pullknobs.tar.gz) (0.75 GB)
#### [Lever knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/leverknobs.tar.gz) (0.77 GB)
#### [Round knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/roundknobs.tar.gz) (1.24 GB)

## Step1: Configure the folder structure.
* Extract and place the downloaded door knob dataset under the `door` folder (or make a symlink).
* Place your favorite robots under the `robot` folder. (Blue robots are there as default)

## Step2: Run following (e.g. lever knob and hook arm combination.)
`cd path/to/DoorGym/`

`python3 ./world_generator/world_generator.py --knob-type lever --robot-type floatinghook`

## Step3: Check the model by running the mujoco simulator
`cd ~/.mujoco/mjpro150/bin`

`./simulate path/to/DoorGym/world_generator/world/lever_floatinghook/1551848929_lever_floatinghook.xml`

## Arguments

| name             | options                                                                              | default        | explaination                                                                        |
|------------------|--------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------------|
| knob-type        | "lever", "round", "pull" or "slider" (Slider is WIP)                                 | ' '            | If no arg, it use all types.                                                        |
| robot-type       | "gripper","hook","floatinghook","floatinggripper", "mobile_gripper" or "mobile_hook" | 'floatinghook' | -                                                                                   |
| one-dir          | True/False                                                                           | False          | Save everything into one dir, or save into separate dir by its robot and knob types |
| pulldoor-ratio   | 0.0~1.0                                                                              | 1.0            | ratio of door that opens by pulling.                                                |
| righthinge-ratio | 0.0~1.0                                                                              | 1.0            | ratio of door that has hinge on right side.                                         |

# Reference
This mjcf generator is powered by the following repo.

[https://github.com/iandanforth/mjcf](https://github.com/iandanforth/mjcf)

# List of worlds

| name                                 | knob type | robot type      | options   | used in                                              | modification                                                              |
|--------------------------------------|-----------|-----------------|-----------|------------------------------------------------------|---------------------------------------------------------------------------|
| lever_blue_floatinggripper           | lever     | floatinggripper |           | ppo-hn<x>, x=9..11, guild optimizations, determinism ||
| lever_blue_floatinggripper_left      | lever     | floatinggripper | left      | ppo-hn<x>, x=9..11, guild optimizations              ||
| lever_blue_floatinghook              | lever     | floatinghook    |           | ppo-hn<x>, x=1..4, guild hnppo series                ||
| lever_blue_floatinghook_left         | lever     | floatinghook    | left      | guild hnppo series                                   ||
| lever_blue_floatinghook_push         | lever     | floatinghook    | push      | guild hnppo series                                   ||
| lever_blue_floatinghook_push_left    | lever     | floatinghook    | push+left | guild hnppo series                                   ||
| lever_blue_gripper                   | lever     | gripper         |           | ppo-hn<x>, x=5..8                                    ||
| pull_blue_floatinggripper            | pull      | floatinggripper |           | ppo-hn<x>, x=9..11, guild optimizations              ||
| pull_blue_floatinggripper_left       | pull      | floatinggripper | left      | ppo-hn<x>, x=9..10, guild optimizations              ||
| pull_blue_floatinggripper_left_fixed | pull      | floatinggripper | left      | ppo-hn11, guild optimizations                        | moved handle further inward to avoid frame collision, no fixed seed (old) |
| pull_blue_floatinghook               | pull      | floatinghook    |           | ppo-hn<x>, x=1..4                                    ||
| pull_blue_floatinghook_left_fixed    | pull      | floatinghook    | left      | guild hnppo series2                                  | moved handle further inward                                               |
| pull_blue_gripper                    | pull      | gripper         |           | ppo-hn<x>, x=5..8                                    ||
| round_blue_floatinggripper           | round     | floatinggripper |           | ppo-hn<x>, x=9..10, guild optimizations              ||
| round_blue_floatinggripper_easy      | round     | floatinggripper |           | ppo-hn11                                             | decrease spring and damping force on handle, no fixed seed (old)          |
| round_blue_floatinghook              | round     | floatinghook    |           | ppo-hn<x>, x=1..4, guild hnppo series                ||
| round_blue_gripper                   | round     | gripper         |           | ppo-hn<x>, x=5..8                                    ||

2022-08-26: remake all worlds with seeded rng they are reproducible (seed=1).