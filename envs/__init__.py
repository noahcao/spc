from __future__ import division, print_function

class make_env(object):
    def __init__(self, args):
        super(make_env, self).__init__()
        self.args = args

    def __enter__(self):
        if 'torcs' in self.args.env:
            import gym
            from .TORCS.torcs_wrapper import TorcsWrapper
            self.env = gym.make('TORCS-v0')
            self.env.init(isServer=self.args.server,
                          continuous=True,
                          resize=not self.args.eval,
                          ID=self.args.id)
            return TorcsWrapper(self.env)

        elif 'carla8' in self.args.env:
            # run spc on carla0.9 simulator, currently only 0.9.4 is supported
            from carla.client import make_carla_client
            from .CARLA.carla_env import CarlaEnv
            with make_carla_client('localhost', self.args.port) as client:
                return CarlaEnv(client)

        elif 'carla9' in self.args.env:
            # run spc on carla0.9 simulator, currently only 0.9.4 is supported
            import glob
            import os
            import sys
            from .CARLA.carla_env.world import World
            try:
                sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
            except IndexError:
                pass
            import carla
            client = carla.Client('localhost', self.args.port)
            client.set_timeout(20.0)
            carla_world = client.get_world()
            settings = carla_world.get_settings()
            settings.synchronous_mode = True
            client.get_world().apply_settings(settings)
            env = World(self.args, carla_world)
            return env


        elif 'gta' in self.args.env:
            from .GTAV.gta_env import GtaEnv
            from .GTAV.gta_wrapper import GTAWrapper
            return GTAWrapper(GtaEnv(autodrive=None))

        else:
            assert(0)


    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'torcs' in self.args.env:
            self.env.close()

        elif 'carla8' in self.args.env:
            pass  # self.client.__exit__(exc_type, exc_val, exc_tb)

        elif 'carla9' in self.args.env:
            pass

        elif 'gta' in self.args.env:
            pass

