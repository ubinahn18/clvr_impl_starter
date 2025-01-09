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


