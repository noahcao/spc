from bbox import ClientSideBoundingBoxes
import carla
import numpy as np
import math

VIEW_FOV = 90

class Agent(object):
    def __init__(self, args, world):
        self.args = args
        self.world = world

        self.camera = None
        self.car = None
        self.display = None
        self.image = None
        self.capture = True
        self.col_sensor = None
        self.offlane_detector = None

        self.collision = False
        self.offlane = False

        setup_vehicle()
        setup_camera()
        setup_collision_sensor()
        setup_offlane_detector()

    def setup_vehicle(self):
        # set up the agent vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle = random.choice(blueprint_library.filter('vehicle.bmw.*')) # randomly choose a bmw vehicle
        self.spawn_points = world.get_map().get_spawn_points() # where to spawn the vehicle
        self.vehicle = world.spawn_actor(vehicle, spawn_points)

    def setup_camera(self):
        # set up the camera to observe surrounding
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = self.args.width / 2.0
        calibration[1, 2] = self.args.height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.args.width / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def setup_collision_sensor(self):
        collision_sensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = world.spawn_actor(collision_sensor, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.col_sensor.listen(lambda event: self.on_collision(weak_self, event))

    def setup_offlane_detector(self):
        offlane_bp = self.world.get_blueprint_library().find('sensor.other.lane_detector')
        self.offlane_detector = world.spawn_actor(bp, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.offlane_detector.listen(lambda event: on_invasion(weak_self, event))

    @staticmethod
    def on_collision(weak_self, event):
        self.collision = True
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    @staticmethod
    def on_invasion(weak_self, event):
        self.offlane = True
        self = weak_self()
        if not self:
            return

    def info(self):
        v = self.car.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        return [self.offlane, self.collision, speed]

    def observe(self):
        self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        return self.display

    def bbox(slef):
        bounding_box = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
        return bounding_box

    def camera_blueprint(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.args.frame_width))
        camera_bp.set_attribute('image_size_y', str(self.args.frame_height))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def step(self, throttle, reverse, steer, hand_brake):
        control = self.car.get_control()
        control.throttle = throttle
        control.reverse = reverse
        control.steer = steer
        control.hand_brake = hand_brake
        self.car.apply_control(control)

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)
    