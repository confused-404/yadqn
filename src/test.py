import gymnasium as gym
from ai.Middleman import Middleman

def doenv():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    randomize_obs, _ = env.reset(options={"randomize": True})
    non_random_obs, _ = env.reset(options={"randomize": False})

    agent = Middleman(env)
    agent.train(render=True, log_progress=True)

    env.close()