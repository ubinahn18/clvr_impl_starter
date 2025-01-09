from models import *
from sprites_env.envs.sprites import *

 data_spec = AttrDict(
        resolution=64,
        max_ep_len=40,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True,
    )

env = SpritesEnv()


def rollout_trajectory(env, num_steps = 50):
 
    states = []
    rewards = []

    state = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        states.append(next_state)
        rewards.append(reward)

        if done:
            break

    return np.array(states), np.array(rewards)









