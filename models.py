import json
import numpy as np

from typing import List, Dict

from settings import Configuration
from utils import Vector, clamp


class Coordinate(Vector):
    def __init__(self, x: float, y: float, z: float, momentum=None):
        super().__init__(x, y, z)
        assert momentum is None or isinstance(momentum, Mobility)
        self.momentum = momentum if momentum else Mobility.random()

    def copy(self):
        return Coordinate(self.x, self.y, self.z, self.momentum)

    def move(self, env, new_coordinate):
        assert isinstance(new_coordinate, Coordinate)

        while env.block(self, new_coordinate):
            distance = abs(np.random.normal(0, (new_coordinate - self).size()))
            angle = np.random.uniform(0, 2 * np.pi)
            new_coordinate = Coordinate(
                x=self.x + distance * np.cos(angle), y=self.y + distance * np.sin(angle), z=self.z
            )
            self.momentum = Mobility(distance, angle)

        self.update(
            x=clamp(new_coordinate.x, 0, env.width),
            y=clamp(new_coordinate.y, 0, env.height),
            z=clamp(new_coordinate.z, 0, env.depth)
        )

    def random_walk(self, env, distance: float):
        angle = np.random.uniform(0, 2 * np.pi)
        new_coordinate = Coordinate(
            x=self.x + distance * np.cos(angle), y=self.y + distance * np.sin(angle), z=self.z
        )
        self.move(env, new_coordinate)

    def move_toward(self, env, destination, max_distance: float):
        assert isinstance(destination, Coordinate)
        distance = np.random.uniform(0, max_distance)
        if self.horizontal_distance(destination) <= distance:
            new_coordinate = Coordinate(x=destination.x, y=destination.y, z=self.z)
        else:
            angle = np.arctan((destination.y - self.y) / (destination.x - self.x))
            new_coordinate = Coordinate(
                x=self.x + distance * np.cos(angle), y=self.y + distance * np.sin(angle), z=self.z
            )
        self.move(env, new_coordinate)

    def move_pattern(self, env, speed_scale: float, momentum_ratio: float):
        patterns = [
            env.mobility_map[x][y]
            for x, y in [
                (int(np.trunc(self.x)), int(np.trunc(self.y))),
                (int(np.trunc(self.x)), int(np.ceil(self.y))),
                (int(np.ceil(self.x)), int(np.trunc(self.y))),
                (int(np.ceil(self.x)), int(np.ceil(self.y)))
            ]
            if not env.block(Coordinate(self.x, self.y, 0), Coordinate(x, y, 0))
        ]

        distance = abs(np.random.normal(0, speed_scale))

        if patterns:
            mobility = sum(patterns) / len(patterns)

            if mobility.is_opposite(self.momentum):
                mobility = -mobility

            self.momentum = self.momentum * momentum_ratio + mobility * (1 - momentum_ratio)

        self.move(env, self.momentum.sample(self, distance))


class Mobility:
    def __init__(self, radius: float, theta: float):
        assert 0 <= radius
        self.radius = radius
        self.theta = theta % (2 * np.pi)

    @staticmethod
    def random():
        return Mobility(np.random.uniform(0.5, 1), np.random.uniform(0, 2 * np.pi))

    @property
    def x(self):
        return self.radius * np.cos(self.theta)

    @property
    def y(self):
        return self.radius * np.sin(self.theta)

    def is_opposite(self, mobility) -> bool:
        assert isinstance(mobility, Mobility)
        return np.pi / 2 <= abs(self.theta - mobility.theta) < 3 * np.pi / 2

    def sample(self, coordinate: Coordinate, distance: float) -> Coordinate:
        radius = abs(np.random.normal(0, self.radius))
        theta = np.random.normal(self.theta)
        return Coordinate(
            x=coordinate.x + radius * np.cos(theta) * distance,
            y=coordinate.y + radius * np.sin(theta) * distance,
            z=coordinate.z
        )

    def __neg__(self):
        return Mobility(self.radius, self.theta + np.pi)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Mobility(self.radius + other, self.theta)
        return Mobility(self.radius + other.radius, self.theta + other.theta)

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Mobility(other + self.radius, self.theta)
        return Mobility(other.radius + self.radius, other.theta + self.theta)

    def __sub__(self, other):
        return Mobility(self.radius - other.radius, self.theta - other.theta)

    def __rsub__(self, other):
        return Mobility(other.radius - self.radius, other.theta - self.theta)

    def __mul__(self, other):
        return Mobility(self.radius * other, self.theta * other)

    def __rmul__(self, other):
        return Mobility(other * self.radius, other * self.theta)

    def __truediv__(self, other):
        return Mobility(self.radius / other, self.theta / other)

    def __eq__(self, other):
        return isinstance(other, Mobility) and self.radius == other.radius and self.theta == other.theta


class Service:
    def __init__(self, sid: int, coordinate: Coordinate, configuration: Configuration):
        self.sid = sid
        assert isinstance(coordinate, Coordinate)
        self.coordinate = coordinate
        self.agent = None

        # Pattern
        self.intensity_range = [0, np.random.randint(*configuration.service_maximum_intensity_range)]
        self.adjust_step = configuration.service_adjust_step

        # State
        self.user = None
        self.intensity = 0  # 1m reference
        self.duration = 0
        self.count = 0  # Number of provision, count after release

    def set(self, intensity: int, duration: int):
        self.set_intensity(intensity)
        self.duration = duration

    def set_intensity(self, intensity: int):
        self.intensity = clamp(value=intensity, lower=self.intensity_range[0], upper=self.intensity_range[1])

    def set_agent(self, agent):
        self.agent = agent
        self.agent.set_service(self)

    def increase(self):
        self.set_intensity(self.intensity + self.adjust_step)

    def decrease(self):
        self.set_intensity(self.intensity - self.adjust_step)

    def reset(self):
        assert not self.user
        self.set_intensity(0)
        self.duration = 0
        self.count = 0

    def prediction(self, samples):
        return self.agent.prediction(samples, training=False)

    def add(self, observation, context, value):
        self.agent.memory.learn(observation, context, value)

    def learn(self):
        return self.agent.learn()

    def is_full(self):
        return self.agent.memory.is_full()

    def is_empty(self):
        return self.agent.memory.is_empty()

    def is_ready(self):
        return self.agent.is_ready()

    def distance(self, observation: List[float]):
        return self.agent.memory.distance(observation)

    def state(self):
        return [self.intensity, self.duration]

    def freeze(self) -> dict:
        return {
            "coordinate": self.coordinate.copy(),
        }

    def resume(self, history: dict):
        self.coordinate = history["coordinate"].copy()

    def summary(self) -> str:
        return json.dumps({
            "memory": self.agent.memory.summary()
        }, indent=4)

    def __str__(self):
        return "Service %d" % self.sid


class User:
    def __init__(self, uid: int, coordinate: Coordinate, configuration: Configuration):
        self.uid = uid
        assert isinstance(coordinate, Coordinate)
        assert configuration.user_height_range[0] <= coordinate.z <= configuration.user_height_range[1]
        self.coordinate = coordinate

        # Action
        self.enter_probability = configuration.user_enter_probability
        self.request_probability = configuration.user_request_probability
        self.exit_probability = configuration.user_exit_probability
        self.intensity_range = configuration.user_intensity_range
        self.duration_range = configuration.user_duration_range
        self.speed_scale = configuration.user_speed_scale
        self.momentum_ratio = configuration.momentum_ratio

        # Feedback
        self.feedback_probability = configuration.user_feedback_probability
        self.satisfaction_scale = configuration.user_satisfaction_scale
        self.interference_scale = configuration.user_interference_scale
        self.interference_threshold = configuration.user_interference_threshold

        # State
        self.active = False
        self.request = None
        self.service = None
        self.episode = None
        self.releasing = False

    def reset(self):
        assert not self.service
        self.active = False
        self.request = None
        self.service = None
        self.episode = None
        self.releasing = False

    def selection(self, candidates: List[Service], context: Dict[str, Dict], exploration: bool, train: bool) -> Service:
        assert self.request and len(candidates) > 0

        Q = [service.prediction([{"observation": self.state(), "context": context}]) for service in candidates]
        assert len(Q) == len(candidates)

        # Exploration
        if exploration and train and all([q <= 0 for q in Q]):
            selection = []
            min_distance = float('inf')
            for service in candidates:
                if service.agent.memory.length() < min_distance:
                    min_distance = service.agent.memory.length()
                    selection = [service]
                elif service.agent.memory.length() == min_distance:
                    selection.append(service)

            assert selection
            return np.random.choice(selection)

        # Greedy
        return candidates[np.argmax(Q)]

    def step(self, env, minute: int, max_minute: int):
        if not self.active and np.random.random() < self.enter_probability:
            self.coordinate = env.get_random_user_coordinate()
            self.active = True
        elif self.active and np.random.random() < self.exit_probability and not self.service:
            self.active = False

        # Mobility pattern
        self.coordinate.move_pattern(env, self.speed_scale, self.momentum_ratio)
        if env.border(self.coordinate):
            self.active = False

        if not self.active:
            return

        # Request pattern
        if not self.service and not self.request:
            if np.random.random() < self.request_probability:
                request = Request(
                    intensity=np.random.randint(low=self.intensity_range[0], high=self.intensity_range[1]),
                    duration=np.random.randint(low=self.duration_range[0], high=self.duration_range[1])
                )
                if minute + request.duration < max_minute:
                    # Do not allow duration to exceed num_steps
                    self.request = request

        # Release pattern
        if self.service:
            assert 0 < self.service.duration == self.request.duration
            self.service.duration -= 1
            self.request.duration -= 1
            if self.service.duration <= 0:
                self.releasing = True

    def state(self) -> List[float]:
        return self.coordinate.vectorize()

    def feedback(self) -> bool:
        return np.random.random() < self.feedback_probability

    def satisfaction(self, env) -> float:
        if not self.feedback():
            return 0

        assert self.service and self.episode

        intensity = self.perceived_intensity(env, self.service)

        return self.satisfaction_scale if (
                intensity >= self.perceived_interference(env) and intensity >= self.episode.request.intensity
        ) else -self.satisfaction_scale

    def interference(self, env) -> float:
        if not self.feedback():
            return 0
        return self.interference_scale if self.perceived_interference(env) > self.interference_threshold else 0

    def perceived_intensity(self, env, service: Service) -> float:
        if service.intensity <= 0:
            return 0

        intensity = service.intensity - 20 * np.log10(self.distance(service))

        for wall, intersection in env.block(service.coordinate, self.coordinate):
            intensity += 20 * np.log10(1 - wall.absorption_rate)

        return intensity

    def perceived_interference(self, env) -> float:
        sounds = [env.background_noise] + [
            self.perceived_intensity(env, service) for service in env.services if service is not self.service
        ]
        return 10 * np.log10(sum([pow(10, n / 10) for n in sounds if n > 0]))

    def acquire(self, service: Service):
        assert self.request and not self.service and not service.user
        service.user = self

        service.set(intensity=self.request.intensity, duration=self.request.duration)

        self.service = service

        self.episode = Episode(self, self.request, service)

    def release(self):
        assert self.service and self.episode
        self.service.user = None
        self.service.set_intensity(0)
        self.service.duration = 0
        self.service.count += 1

        self.service = None

        episode = self.episode
        self.episode = None
        self.request = None

        return episode

    def adjust_intensity(self, env):
        assert self.service

        while (self.satisfaction(env) == self.satisfaction_scale
               and self.service.intensity > self.service.intensity_range[0]):
            self.service.decrease()

        while (self.satisfaction(env) == -self.satisfaction_scale
               and self.service.intensity < self.service.intensity_range[1]):
            self.service.increase()

    def distance(self, other):
        assert isinstance(other, User) or isinstance(other, Service)
        return self.coordinate.distance(other.coordinate)

    def freeze(self) -> dict:
        return {
            "active": self.active,
            "coordinate": self.coordinate.copy(),
            "request": self.request.copy() if self.request else None,
            "releasing": self.releasing,
        }

    def resume(self, history: dict):
        self.active = history["active"]
        self.coordinate = history["coordinate"].copy()
        self.request = history["request"].copy() if history["request"] else None
        self.releasing = history["releasing"]

    def __str__(self):
        return "User %d" % self.uid


class Wall:
    def __init__(self, a: Coordinate, b: Coordinate, configuration: Configuration):
        # wall is from a-endpoint to b-endpoint, and orthogonal to ground
        self.a = a
        self.b = b

        sub = (b - a)
        self.normal = Vector(x=1, y=(-sub.x / sub.y), z=0) if sub.y != 0 else Vector(x=(-sub.y / sub.x), y=1, z=0)

        self.absorption_rate = np.random.uniform(*configuration.wall_absorption_rate_range)

    def intersect(self, start: Coordinate, end: Coordinate):
        direction = end - start
        denominator = self.normal.dot(direction)

        # Parallel, ignore contained case
        if denominator == 0:
            return None

        # Intersect
        intersection = start + (self.normal.dot(self.a - start) / denominator) * direction

        if intersection.between(start, end) and intersection.between(self.a, self.b):
            return intersection
        return None


class Request:
    def __init__(self, intensity: float, duration: int):
        self.intensity = intensity
        self.duration = duration

    def copy(self):
        return Request(self.intensity, self.duration)


class Reward:
    def __init__(self, satisfaction: float, interference: float):
        self.satisfaction = satisfaction
        self.interference = interference

    def __float__(self):
        return float(self.satisfaction - self.interference)

    def __add__(self, other):
        return float(self) + float(other)

    def __radd__(self, other):
        return float(other) + float(self)

    def __sub__(self, other):
        return float(self) - float(other)

    def __rsub__(self, other):
        return float(other) - float(self)


class Episode:
    def __init__(self, requester: User, request: Request, service: Service):
        self.requester = requester
        self.request = request
        self.service = service

        self.total_duration = request.duration

        self.observations = []
        self.contexts = []
        self.rewards = []

    def add_observation(self, observation: List[float], context: Dict[str, Dict], reward: Reward):
        self.observations.append(observation)
        self.contexts.append(context)
        self.rewards.append(reward)

    def reward_sum(self):
        return sum(self.rewards)

    def learn(self):
        assert len(self.observations) == len(self.contexts) == len(self.rewards)
        assert len(self.rewards) <= self.total_duration + 1

        return self.service.agent.learn([{
            "observation": self.observations[i],
            "context": self.contexts[i],
            "satisfaction": sum([r.satisfaction for r in self.rewards[i:]]) / len(self.rewards[i:]),
            "interference": sum([r.interference for r in self.rewards[i:]]) / len(self.rewards[i:])
        } for i in range(len(self.observations))])
