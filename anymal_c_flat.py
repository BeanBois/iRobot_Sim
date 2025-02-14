# anymal_rl_example.py
import numpy as np
from skrl.envs.loaders.torch import load_isaaclab_env

# Load the environment.
# Here, we specify the task name matching the AnyMal C Rough environment.
# You can also override the number of parallel environments and headless mode.
env = load_isaaclab_env(
    task_name="Isaac-Velocity-Flat-Anymal-C-v0",  # Ensure this matches your registered task name.
    num_envs=1,                              # Number of parallel environments (set as desired).
    headless=True,                           # Set to True to run without rendering (or False if you want visualization).
    cli_args=[],
    show_cfg=True
)

# Reset the environment and get the initial observation.
obs = env.reset()
print("Initial observation:", obs)

# Run one episode using a random policy.
done = False
total_reward = 0.0
while not done:
    # Sample a random action from the environment's action space.
    action = env.action_space.sample()
    # Take a step in the environment.
    obs, reward, done, info = env.step(action)
    total_reward += reward
    # Optionally render the environment (if not headless).
    env.render()

print("Episode finished with total reward:", total_reward)

# Always close the environment after running.
env.close()
