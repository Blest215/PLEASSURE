import tensorflow as tf
import warnings
import json

from baseline import WallNearestGreedyAlgorithm, FarthestAlgorithm, NearestGreedyAlgorithm
from agent import ServiceAgent, IndependentAgent, EqualAttentionAgent, FingerprintAttentionAgent


warnings.filterwarnings('ignore')

"""
units
distance = m
sound_pressure = dB
time = minute
"""

record = False
record_detail = False

num_services_variable = False
num_environments = 25


class Configuration:
    def __init__(self, difference_dict: dict = None):
        # Experiment
        self.num_training_days = 5
        self.num_testing_days = 1
        self.num_day_steps = 1440
        self.num_testing_iterations = 25

        # Evaluation
        self.worst = FarthestAlgorithm()
        self.baseline = NearestGreedyAlgorithm()
        self.benchmark = WallNearestGreedyAlgorithm()

        # Environment
        self.width = 10
        self.height = 10
        self.depth = 3

        self.num_users_training = 5
        self.num_users_testing = [1, 3, 7, 9]
        self.num_services = 20
        self.num_walls = 5

        self.background_noise = 30

        # Service model
        self.service_maximum_intensity_range = [40, 100]
        self.service_adjust_step = 1

        # User model
        self.user_height_range = [1.5, 1.9]
        self.user_height_mean = 1.7
        self.user_height_std = 0.1
        self.user_intensity_range = [50, 60]  # Normal conversation 40-60, maximum safe 70
        self.user_duration_range = [5, 15]
        self.user_enter_probability = 0.1
        self.user_exit_probability = 0.01
        self.user_request_probability = 0.01
        self.user_feedback_probability = 1
        self.user_interference_threshold = 60
        self.user_satisfaction_scale = 1
        self.user_interference_scale = 1
        self.user_speed_scale = 1
        self.momentum_ratio = 0.2

        # Wall model
        self.wall_absorption_rate_range = [0.9, 1.0]

        # Training
        self.exploration = True
        self.installation_size = 0

        # Agent
        self.agents = [
            IndependentAgent,
            EqualAttentionAgent,
            FingerprintAttentionAgent,
        ]
        self.agent_spec = Spec(
            embedding_size=256, embedding_activation='relu',
            layer_specs=[
                Spec(
                    tf.keras.layers.Dense,
                    units=512, activation='relu',
                ),
                Spec(
                    tf.keras.layers.Dense,
                    units=512, activation='relu',
                ),
                Spec(
                    tf.keras.layers.Dense,
                    units=512, activation='relu',
                ),
                Spec(
                    tf.keras.layers.Dense,
                    units=512, activation='relu',
                ),
                Spec(
                    tf.keras.layers.Dense,
                    units=512, activation='relu',
                ),
            ],
            memory_size=4096, learning_rate=1e-4,
            epochs=100, patience=5, validation_fold=5, batch_size=4096,
            ragged=num_services_variable,
            user_state_size=3, service_state_size=2,
        )

        if difference_dict:
            for key, value in difference_dict.items():
                self.__dict__[key] = value

        # Save settings
        self.__setting__ = self.__dict__.copy()

    def update(self, difference_dict: dict):
        diff = self.__dict__.copy()
        del diff['__setting__']
        for key, value in difference_dict.items():
            diff[key] = value
        return Configuration(difference_dict=diff)

    def construct_agents(self, agent_class: ServiceAgent):
        return [self.agent_spec.instantiate(agent_class) for _ in range(self.num_services)]

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            try:
                return json.JSONEncoder.default(self, obj)
            except TypeError:
                try:
                    return obj.__setting__
                except (TypeError, AttributeError):
                    return str(obj)

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            f.write(json.dumps(self.__setting__, indent=4, cls=self.CustomJSONEncoder))
            f.close()

    def save_difference(self, file_path: str):
        def compare(a, b):
            if isinstance(a, int) or isinstance(a, str):
                return a if a != b else None
            a_dict = a.__setting__ if hasattr(a, "__setting__") else a
            b_dict = b.__setting__ if hasattr(b, "__setting__") else b

            difference = {}
            for key in a_dict:
                if key not in b_dict:
                    difference[key] = a_dict[key]
                else:
                    if hasattr(a_dict[key], "__setting__") or isinstance(a_dict[key], dict):
                        result = compare(a_dict[key], b_dict[key])
                        if result:
                            difference[key] = result
                    elif isinstance(a_dict[key], list):
                        result = [
                            compare(a_dict[key][i], b_dict[key][i]) for i in range(len(a_dict[key]))
                        ] if len(a_dict[key]) == len(b_dict[key]) else a_dict[key]
                        if any(result):
                            difference[key] = result
                    elif a_dict[key] != b_dict[key]:
                        difference[key] = a_dict[key]
            return difference

        default = Configuration()
        with open(file_path, 'w') as f:
            f.write(json.dumps(compare(self, default), indent=4, cls=self.CustomJSONEncoder))
            f.close()

    # Compare in terms of environmental settings
    def __eq__(self, other):
        return isinstance(other, Configuration) and all([
            self.width == other.width,
            self.height == other.height,
            self.depth == other.depth,
            self.num_services == other.num_services,
            self.num_walls == other.num_walls,
            self.background_noise == other.background_noise,
        ])


class Spec:
    def __init__(self, constructor=None, **kwargs):
        self.constructor = constructor
        self.kwargs = kwargs

        # Save settings
        self.__setting__ = self.__dict__.copy()

    def update(self, constructor=None, difference_dict: dict = None):
        kwargs = self.kwargs.copy()
        if difference_dict:
            for key, value in difference_dict.items():
                kwargs[key] = value
        return Spec(constructor if constructor else self.constructor, **kwargs)

    def instantiate(self, constructor=None):
        assert self.constructor or constructor
        return self.constructor(**self.kwargs) if self.constructor else constructor(**self.kwargs)
