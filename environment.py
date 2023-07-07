import numpy as np

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict

from settings import record, record_detail, num_services_variable
from models import User, Service, Wall, Episode, Coordinate, Reward, Request, Mobility
from utils import clamp, get_summary_path, save_gif
from agent import ServiceAgent


class Environment:
    def __init__(self, index: int, configuration):
        self.index = index
        self.configuration = configuration

        # Settings
        self.width = configuration.width
        self.height = configuration.height
        self.depth = configuration.depth

        self.background_noise = configuration.background_noise

        # States
        self.users = []
        self.services = []
        self.walls = []

        # Mobility pattern
        self.mobility_map = []

        # Trace
        self.training_history = {}
        self.testing_history = {}
        self.testing_num_requests = {}

        # Evaluation
        self.worst_reward = {}
        self.benchmark_reward = {}

        # Rendering
        self.scale = 50
        self.user_icon = Image.open('rendering/user.png', 'r').convert("RGBA").resize(
            (int(self.scale), int(2 * self.scale))
        )
        self.service_icon = Image.open('rendering/speaker.png', 'r').convert("RGBA").resize(
            (int(self.scale / 2), int(self.scale * 3 / 4))
        )
        self.font = ImageFont.truetype("arial.ttf", int(self.scale / 2)) if record else None
        self.small_font = ImageFont.truetype("arial.ttf", int(self.scale / 4)) if record else None

        self.field = None

        # Services
        assert configuration.num_services >= 1
        self.set_services(configuration.num_services)

        # Walls
        self.walls = []
        for i in range(configuration.num_walls):
            a = self.get_random_coordinate()
            a.x = clamp(int(np.trunc(a.x)) + 0.5, 0, self.width)
            a.y = clamp(int(np.trunc(a.y)) + 0.5, 0, self.height)
            a.z = 0
            direction = np.random.randint(0, 4)
            if direction == 0:  # Up
                length = 1 + np.round(np.random.random() * self.width)
                b = Coordinate(a.x + length, a.y, self.depth)
            elif direction == 1:  # Down
                length = 1 + np.round(np.random.random() * self.width)
                b = Coordinate(a.x - length, a.y, self.depth)
            elif direction == 2:  # Right
                length = 1 + np.round(np.random.random() * self.height)
                b = Coordinate(a.x, a.y + length, self.depth)
            else:  # Left
                length = 1 + np.round(np.random.random() * self.height)
                b = Coordinate(a.x, a.y - length, self.depth)
            self.walls.append(Wall(a=a, b=b, configuration=configuration))

        # Mobility pattern
        self.mobility_map = []
        for x in range(self.width + 1):
            line = []
            for y in range(self.height + 1):
                mobility = Mobility.random()
                while self.block(
                    Coordinate(x - mobility.x, y - mobility.y, 0),
                    Coordinate(x + mobility.x, y + mobility.y, 0)
                ):
                    mobility = Mobility.random()
                line.append(mobility)
            self.mobility_map.append(line)

        # Smooth mobility pattern
        mobility_map = []
        for x in range(self.width + 1):
            line = []
            for y in range(self.height + 1):
                vicinity = [self.mobility_map[x][y]]
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if 0 <= x + i <= self.width and 0 <= y + j <= self.height and not self.block(
                            Coordinate(x, y, 0), Coordinate(x + i, y + j, 0)
                        ):
                            vicinity.append(self.mobility_map[x + i][y + j])
                average = sum(vicinity) / len(vicinity)
                if not self.block(
                        Coordinate(x - average.x, y - average.y, 0),
                        Coordinate(x + average.x, y + average.y, 0)
                ):
                    line.append(average)
                else:
                    line.append(self.mobility_map[x][y])
            mobility_map.append(line)

        self.mobility_map = mobility_map

    def installation(self, installation_size: int):
        for service in self.services:
            locations = []
            for x in range(self.width):
                for y in range(self.height):
                    locations.append(Coordinate(x, y, self.depth / 2))

            episodes = []
            for location in locations:
                context = {"services": {}, "users": {}, "fingerprints": {}}

                dummy = User(uid=-1, coordinate=location, configuration=self.configuration)
                dummy.active = True
                dummy.request = Request(intensity=self.configuration.user_intensity_range[1], duration=0)
                dummy.acquire(service)
                for _ in range(10):
                    dummy.adjust_intensity(self)
                dummy.episode.add_observation(
                    observation=dummy.state(),
                    context=context,
                    reward=Reward(satisfaction=dummy.satisfaction(self), interference=self.get_interferences())
                )
                episode = dummy.release()
                episodes.append(episode)

            np.random.shuffle(episodes)

            successful_episodes = []
            failed_episodes = []
            for episode in episodes:
                if episode.reward_sum() > 0 and len(successful_episodes) < installation_size / 2:
                    successful_episodes.append(episode)
                elif episode.reward_sum() < 0 and len(failed_episodes) < installation_size / 2:
                    failed_episodes.append(episode)

            for episode in successful_episodes + failed_episodes:
                episode.learn()

            service.reset()

    def reset(self):
        for user in self.users:
            user.reset()
        for service in self.services:
            service.reset()

    def erase_history(self, iteration: int = None, training: bool = None):
        if iteration is not None and training is not None:
            self.worst_reward.pop(iteration, None)
            self.benchmark_reward.pop(iteration, None)

            if training:
                self.training_history.pop(iteration, None)
            else:
                self.testing_history.pop(iteration, None)
                self.testing_num_requests.pop(iteration, None)
        else:
            self.worst_reward = {}
            self.benchmark_reward = {}

            self.training_history = {}
            self.testing_history = {}
            self.testing_num_requests = {}

    def freeze(self, train: bool, iteration: int, minute: int):
        history = self.training_history if train else self.testing_history
        if iteration not in history:
            history[iteration] = {}
        history[iteration][minute] = {
            e: e.freeze() for e in self.users + self.services
        }

    def resume(self, train: bool, iteration: int, minute: int):
        history = self.training_history[iteration] if train else self.testing_history[iteration]
        for e in self.users + self.services:
            e.resume(history[minute][e])

    def set_num_requests(self, iteration: int, num_requests: int):
        self.testing_num_requests[iteration] = num_requests

    def has_history(self, train: bool, iteration: int, steps: List[int]) -> bool:
        history = self.training_history if train else self.testing_history
        return iteration in history and all([step in history[iteration] for step in steps])

    def get_worst_reward(self, now, name, algorithm, iteration, steps):
        if iteration not in self.worst_reward:
            reward, frames = algorithm.run(self, iteration, steps, self.testing_num_requests[iteration], record)
            self.worst_reward[iteration] = reward
            save_gif(frames, get_summary_path(
                now=now, index=self.index, name=name, filename=f"TestWorst{iteration}.gif"
            ))
        return self.worst_reward[iteration]

    def get_benchmark_reward(self, now, name, algorithm, iteration, steps):
        if iteration not in self.benchmark_reward:
            reward, frames = algorithm.run(self, iteration, steps, self.testing_num_requests[iteration], record)
            self.benchmark_reward[iteration] = reward
            save_gif(frames, get_summary_path(
                now=now, index=self.index, name=name, filename=f"TestBenchmark{iteration}.gif"
            ))
        return self.benchmark_reward[iteration]

    @property
    def num_users(self) -> int:
        return len(self.users)

    def set_users(self, num_users: int):
        self.users = [
            User(uid=i, coordinate=self.get_random_user_coordinate(), configuration=self.configuration)
            for i in range(num_users)
        ]
        self.erase_history()

    @property
    def num_services(self) -> int:
        return len(self.services)

    def set_services(self, num_services: int):
        self.services = [
            Service(sid=i, coordinate=self.get_random_coordinate(), configuration=self.configuration)
            for i in range(num_services)
        ]
        self.erase_history()

    @property
    def num_walls(self) -> int:
        return len(self.walls)

    def set_agents(self, agents: List[ServiceAgent]):
        assert len(self.services) == len(agents)
        for i in range(len(self.services)):
            self.services[i].set_agent(agents[i])

    def observation(self) -> Tuple[List[User], List[Service], Dict[str, Dict]]:
        requesters = [user for user in self.users if user.active and user.request and not user.service]
        assert all([not user.episode for user in requesters])
        assert all([
            service.intensity > 0 or (service.intensity == 0 and service.duration == 0)
            for service in self.services
        ])

        available_services = [service for service in self.services if not service.user]

        return requesters, available_services, {
            "services": {
                service: service.state() for service in self.services
                if service.intensity > 0 or not num_services_variable
            },
            "fingerprints": {
                service: service.agent.fingerprint for service in self.services
            },
        }

    def step(self, minute: int, max_minute: int):
        for user in self.users:
            user.step(self, minute, max_minute)

    def after_step(self):
        for user in self.users:
            if user.service:
                user.adjust_intensity(self)

    def step_summary(self, context) -> Tuple[Reward, List[Episode]]:
        # Collect noise feedback
        interference_sum = self.get_interferences()
        satisfaction_sum = 0

        finished_episodes = []

        # Count active services
        num_active_services = len([service for service in self.services if service.intensity > 0])

        # Summarize reward
        for user in self.users:
            if user.episode:
                satisfaction = user.satisfaction(self)
                user.episode.add_observation(
                    observation=user.state(),
                    context=context,
                    reward=Reward(satisfaction=satisfaction, interference=interference_sum / num_active_services)
                )
                satisfaction_sum += satisfaction

                if user.releasing or not user.active:
                    finished_episodes.append(user.release())
                    user.releasing = False

        return Reward(satisfaction_sum, interference_sum), finished_episodes

    def get_interferences(self):
        return sum([user.interference(self) for user in self.users if user.active])

    def block(self, a: Coordinate, b: Coordinate) -> List[Tuple[Wall, Coordinate]]:
        walls = []
        for wall in self.walls:
            intersection = wall.intersect(a, b)
            if intersection is not None:
                walls.append((wall, intersection))
        return walls

    def border(self, coordinate: Coordinate) -> bool:
        return coordinate.x == 0 or coordinate.x == self.width or coordinate.y == 0 or coordinate.y == self.height

    def render_field(self) -> Image.Image:
        if self.field is None:
            image = Image.new('RGBA', (self.width * self.scale, self.height * self.scale), (250, 250, 250))
            draw = ImageDraw.Draw(image, "RGBA")

            # Services
            for service in self.services:
                x, y = service.coordinate.x * self.scale, service.coordinate.y * self.scale
                image.paste(self.service_icon, (int(x - self.scale / 4), int(y - self.scale / 2)), self.service_icon)

            # Walls
            for wall in self.walls:
                c = int(250 * (1 - wall.absorption_rate))
                draw.line([(wall.a.x * self.scale,
                            wall.a.y * self.scale),
                           (wall.b.x * self.scale,
                            wall.b.y * self.scale)],
                          fill=(c, c, c), width=int(self.scale / 5))

            self.field = image

        return self.field.copy()

    def render(self, minute: int) -> Image.Image:
        assert record

        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        for service in self.services:
            if service.intensity > 0:
                x, y = service.coordinate.x * self.scale, service.coordinate.y * self.scale

                draw.ellipse(xy=(x - self.scale / 8, y - self.scale / 8, x + self.scale / 8, y + self.scale / 8),
                             fill='green')
                radius = np.power(10, (service.intensity - service.user.episode.request.intensity) / 20) * self.scale
                draw.ellipse(xy=(x - radius, y - radius, x + radius, y + radius),
                             fill=None, outline='green')
                draw.text(xy=(x, y + int(self.scale / 2)),
                          text=f"{service.intensity:.2f}", fill='green', font=self.font)

        # Users
        satisfaction_sum = 0

        for user in self.users:
            if user.active:
                x, y = user.coordinate.x * self.scale, user.coordinate.y * self.scale
                image.paste(self.user_icon, (int(x - self.scale/2), int(y - self.scale)), self.user_icon)

                draw.text(xy=(x, y + self.scale),
                          text=f"{user.perceived_interference(self):.2f}",
                          fill='orange', font=self.font)

                if user.interference(self) > 0:
                    draw.ellipse(xy=(x - self.scale / 4, y - self.scale / 4, x + self.scale / 4, y + self.scale / 4),
                                 fill='orange')

                if user.service:
                    satisfaction = user.satisfaction(self)
                    draw.text(xy=(x, y + self.scale),
                              text=f"\n{user.perceived_intensity(self, user.service):.2f}",
                              fill='green', font=self.font)
                    satisfaction_sum += satisfaction
                    if satisfaction > 0:
                        color = 'green'
                    else:
                        color = 'red'
                    draw.ellipse(xy=(x - self.scale/4, y - self.scale/4, x + self.scale/4, y + self.scale/4),
                                 fill=color)

                    draw.line([(user.coordinate.x * self.scale,
                                user.coordinate.y * self.scale),
                               (user.service.coordinate.x * self.scale,
                                user.service.coordinate.y * self.scale)],
                              fill=color, width=int(self.scale / 10))

                    for wall, intersection in self.block(user.coordinate, user.service.coordinate):
                        ix, iy = intersection.x * self.scale, intersection.y * self.scale
                        draw.ellipse(
                            xy=(ix - self.scale/4, iy - self.scale/4, ix + self.scale/4, iy + self.scale/4),
                            fill=color
                        )

        # Text
        draw.multiline_text(
            xy=(self.scale, self.scale),
            text=f"Step {minute}\nSatisfaction {satisfaction_sum:.2f}\nNoise {self.get_interferences():.2f}",
            fill='blue', font=self.font
        )

        return image

    def render_dead_space(self) -> Image.Image:
        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        for x in range(self.width + 1):
            for y in range(self.height + 1):
                location = Coordinate(x, y, self.depth / 2)
                total_blocked = all([self.block(service.coordinate, location) for service in self.services])
                draw.ellipse(xy=(x * self.scale - self.scale / 16, y * self.scale - self.scale / 16,
                                 x * self.scale + self.scale / 16, y * self.scale + self.scale / 16),
                             fill=(255, 0, 0, 255) if total_blocked else (0, 255, 0, 255))
                mobility = self.mobility_map[x][y]
                draw.line([
                    ((x - mobility.x / 4) * self.scale, (y - mobility.y / 4) * self.scale),
                    ((x + mobility.x / 4) * self.scale, (y + mobility.y / 4) * self.scale)
                ], fill='grey', width=1)

        return image

    def render_mobility(self) -> Image.Image:
        image = self.render_field()
        draw = ImageDraw.Draw(image, "RGBA")

        for x in range(self.width + 1):
            for y in range(self.height + 1):
                mobility = self.mobility_map[x][y]
                draw.line([
                    ((x - mobility.x / 4) * self.scale, (y - mobility.y / 4) * self.scale),
                    ((x + mobility.x / 4) * self.scale, (y + mobility.y / 4) * self.scale)
                ], fill='grey', width=1)

        return image

    def render_value_map(self, service: Service):
        field = self.render_field()

        image = field.copy()
        draw = ImageDraw.Draw(image, "RGBA")

        # Marker
        draw.ellipse(xy=(service.coordinate.x * self.scale - self.scale / 8,
                         service.coordinate.y * self.scale - self.scale / 8,
                         service.coordinate.x * self.scale + self.scale / 8,
                         service.coordinate.y * self.scale + self.scale / 8),
                     fill='green')

        # Range
        radius = np.power(
            10, (service.intensity_range[1] - self.configuration.user_intensity_range[0]) / 20
        ) * self.scale
        draw.ellipse(xy=(service.coordinate.x * self.scale - radius,
                         service.coordinate.y * self.scale - radius,
                         service.coordinate.x * self.scale + radius,
                         service.coordinate.y * self.scale + radius),
                     fill=None, outline='green')

        # Memory
        for memory in service.agent.memory.memory:
            draw.ellipse(
                xy=(memory["observation"][0] * self.scale - self.scale / 16,
                    memory["observation"][1] * self.scale - self.scale / 16,
                    memory["observation"][0] * self.scale + self.scale / 16,
                    memory["observation"][1] * self.scale + self.scale / 16),
                fill='grey'
            )

        # Q value
        locations = []
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                locations.append(Coordinate(x, y, self.depth / 2))

        Q = service.prediction(
            samples=[{
                "observation": location.vectorize(),
                "context": {"services": {}, "users": {}}
            } for location in locations]
        )

        for i in range(len(locations)):
            heat = int(max(64., min(255., abs(Q[i]) * 5)))
            draw.text(xy=(locations[i].x * self.scale, locations[i].y * self.scale),
                      text=f"{Q[i]:.1f}",
                      fill=(255, 0, 0, heat) if Q[i] < 0 else (0, 255, 0, heat),
                      font=self.small_font)

        return image

    def save_preview(self, now: str):
        self.render_mobility().save(get_summary_path(
            now, index=self.index, filename=f"Env{self.index:02}.png"
        ))

    def get_random_coordinate(self) -> Coordinate:
        return Coordinate(
            x=float(np.around(np.random.random() * self.width, decimals=3)),
            y=float(np.around(np.random.random() * self.height, decimals=3)),
            z=float(np.around(np.random.random() * self.depth, decimals=3))
        )

    def get_random_user_coordinate(self) -> Coordinate:
        direction = np.random.random() * 4
        if direction <= 1:  # Top
            x = float(np.around(np.random.random() * self.width, decimals=3))
            y = float(np.around(np.random.random(), decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi)+np.pi)
        elif direction <= 2:  # Right
            x = float(np.around(np.random.random() + self.width - 1, decimals=3))
            y = float(np.around(np.random.random() * self.height, decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi)+np.pi/2)
        elif direction <= 3:  # Bottom
            x = float(np.around(np.random.random() * self.width, decimals=3))
            y = float(np.around(np.random.random() + self.height - 1, decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi))
        else:  # Left
            x = float(np.around(np.random.random(), decimals=3))
            y = float(np.around(np.random.random() * self.height, decimals=3))
            momentum = Mobility(radius=1, theta=np.random.uniform(0, np.pi)-np.pi/2)

        coordinate = Coordinate(
            x=x, y=y,
            z=float(np.around(clamp(np.random.normal(
                self.configuration.user_height_mean, self.configuration.user_height_std
            ), *self.configuration.user_height_range), decimals=3))
        )
        coordinate.momentum = momentum

        return coordinate
