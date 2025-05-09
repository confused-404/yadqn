import gymnasium as gym


def doenv():
    env = gym.make("CarRacing-v3", domain_randomize=True, render_mode="human")
    obs, _ = env.reset()
    randomize_obs, _ = env.reset(options={"randomize": True})
    non_random_obs, _ = env.reset(options={"randomize": False})

    while True:
        action = env.action_space.sample()  # Take a random action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            observation, info = env.reset()

    env.close()