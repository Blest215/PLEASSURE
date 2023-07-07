import os
import errno
import datetime
import numpy as np
import logging

from typing import List, Dict


def get_summary_path(now: str, index: int = None, name: str = None, filename: str = None):
    """ get_summary_path: returns a file path for collecting summary for each experiment """
    path_list = [os.path.dirname(__file__), "summary", now]
    if index is not None:
        path_list.append(f"Env{index:02}")
    if name is not None:
        path_list.append(name)
    if filename is not None:
        path_list.append(filename)
    file_path = os.path.join(*path_list)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return file_path


def info(text):
    t = f"[{datetime.datetime.now()}] {str(text)}"
    logging.info(t)
    print(t)


def debug(text):
    logging.debug(f"[{datetime.datetime.now()}] {str(text)}")


def compute_value(rewards: List[float], discount_factor: float) -> List[float]:
    values = [rewards[-1]]
    for i in reversed(range(len(rewards) - 1)):
        values.append(rewards[i] + discount_factor * values[-1])
    values.reverse()
    assert len(rewards) == len(values)
    return values


def save_gif(frames, path: str):
    assert ".gif" in path
    if len(frames) > 0:
        frames[0].save(path, append_images=frames[1:], format='GIF', save_all=True, duration=100)


def clamp(value: float, lower: float, upper: float):
    assert lower < upper
    return min(upper, max(lower, value))


class ExperienceMemory:
    def __init__(self, size: int):
        self.name = self.__class__.__name__

        assert size > 0
        self.size = size

        # Save settings
        self.__setting__ = self.__dict__.copy()

        self.memory = []

    def add(self, observation: List[float], context: Dict[str, Dict], satisfaction: float, interference: float):
        while self.is_full():
            self.memory.pop(0)
        self.memory.append({
            "observation": observation,
            "context": context,
            "satisfaction": satisfaction,
            "interference": interference,
        })

    def sample(self, size=None):
        if size is None or 0 <= self.length() <= size:
            return self.memory
        return np.random.choice(self.memory, size)

    def split(self, validation_fold: int):
        copied = self.memory.copy()
        np.random.shuffle(copied)

        validation_size = int(self.length() / validation_fold)
        assert validation_size >= 1

        folds = [
            copied[i * validation_size: (i + 1) * validation_size]
            for i in range(validation_fold)
        ]
        if [] in folds:
            folds.remove([])
        folds[-1] += copied[validation_fold * validation_size:]

        return folds

    def length(self) -> int:
        return len(self.memory)

    def is_empty(self) -> bool:
        return self.length() <= 0

    def is_full(self) -> bool:
        return self.length() >= self.size

    def load(self) -> float:
        return self.length() / self.size

    def flush(self):
        self.memory = []

    def distance(self, observation: List[float]) -> float:
        """ Measure the distance between recorded observations and a new observation """
        if self.is_empty():
            return float('inf')
        observations = np.array([sample["observation"] for sample in self.memory])

        return np.average(np.sqrt(np.abs(
            np.sum(np.square(observations), axis=1)
            + np.tile(np.sum(np.square(observation)), len(observations))
            - 2 * observations.dot(observation)
        )))

    def summary(self) -> List[Dict]:
        return [{
            "observation": memory["observation"],
            "context": {
                "services": {
                    str(service): memory["context"]["services"][service]
                    for service in memory["context"]["services"]
                },
                "users": {
                    str(user): memory["context"]["users"][user]
                    for user in memory["context"]["users"]
                }
            },
            "satisfaction": memory["satisfaction"],
            "interference": memory["interference"],
        } for memory in self.memory]


class Vector:
    """ Vector: class of 3-dimensional vector for Coordinate and Direction """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def update(self, x: float, y: float, z: float):
        """ update: updates the elements of the vector """
        self.x = x
        self.y = y
        self.z = z

    def vectorize(self):
        """ vectorize: returns list form of the vector, for concatenation with other lists """
        return [self.x, self.y, self.z]

    def size(self) -> float:
        """ size: returns the size of the vector """
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dot(self, other) -> float:
        """ dot: performs dot product of vectors """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """ cross: performs cross product of vectors """
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def distance(self, other) -> float:
        """ get_distance: get distance between to vectors """
        assert isinstance(other, Vector)
        return (other - self).size()

    def horizontal_distance(self, other) -> float:
        return Vector(other.x - self.x, other.y - self.y, 0).size()

    def unit(self):
        """ to_unit: converts the vector to a unit vector that is parallel to the vector but size 1 """
        denominator = self.size()
        u = Vector(x=self.x/denominator, y=self.y/denominator, z=self.z/denominator)
        assert 1 - 1e-3 <= u.size() <= 1 + 1e-3
        return u

    def between(self, a, b) -> bool:
        """ between: determine whether the vector is between a and b """
        return (self.x - a.x) * (self.x - b.x) <= 0 and (self.y - a.y) * (self.y - b.y) <= 0

    def copy(self):
        return Vector(self.x, self.y, self.z)

    def __str__(self):
        return "(X:{x}, Y:{y}, Z:{z})".format(x=self.x, y=self.y, z=self.z)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.x + other, self.y + other, self.z + other)
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(other + self.x, other + self.y, other + self.z)
        return Vector(other.x + self.x, other.y + self.y, other.z + self.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __rsub__(self, other):
        return Vector(other.x - self.x, other.y - self.y, other.z - self.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Vector(other * self.x, other * self.y, other * self.z)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.x == other.x and self.y == other.y and self.z == other.z
