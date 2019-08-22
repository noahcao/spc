import argparse
from args import init_parser, post_processing
import numpy as np
import torch
from train import train_policy
from evaluate import evaluate_policy
from utils import setup_dirs
import random
from envs import make_env

parser = argparse.ArgumentParser(description='SPC')
init_parser(parser)  # See `args.py` for default arguments
args = parser.parse_args()
args = post_processing(args)

if __name__ == '__main__':
    setup_dirs(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if 'carla' in args.env:
        if args.env == "carla8":
            # run spc on carlav0.8 simulator, 0.8.4 is recommended
            from carla.client import make_carla_client
            from envs.CARLA.carla_env import CarlaEnv
            clint = make_carla_client('localhost', args.port)
            env = CarlaEnv(client)
        elif args.env == "carla9":
            # run spc on carla0.9 simulator, currently only 0.9.4 is supported
            import glob
            import os
            import sys
            from envs.CARLA.carla_env.world import World
            try:
                sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
            except IndexError:
                pass 
            import carla
            client = carla.Client('localhost', args.port)
            client.set_timeout(20.0)
            carla_world = client.get_world()
            settings = carla_world.get_settings()
            settings.synchronous_mode = True
            client.get_world().apply_settings(settings)
            env = World(args, carla_world)

        if args.eval:
            evaluate_policy(args, env)
        else:
            train_policy(args, env, max_steps=args.max_steps)
    else:
        with make_env(args) as env:
            if args.eval:
                evaluate_policy(args, env)
            else:
                train_policy(args, env, max_steps=args.max_steps)
