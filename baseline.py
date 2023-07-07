import numpy as np

from typing import List

from utils import info


class BaselineAlgorithm:
    def __init__(self):
        self.name = self.__class__.__name__

        # Save settings
        self.__setting__ = self.__dict__.copy()

    def run(self, env, iteration: int, steps: List[int], num_requests: int, record: bool):
        rewards = []
        frames = []

        env.reset()

        for step in steps:
            env.resume(train=False, iteration=iteration, minute=step)

            requesters, available_services, context = env.observation()

            for requester in requesters:
                if len(available_services) > 0:
                    action = self.selection(env, requester, available_services)
                    requester.acquire(action)
                    available_services.remove(action)

            env.after_step()

            reward, finished_episodes = env.step_summary(context)
            rewards.append(reward)

            if record:
                frames.append(env.render(step))

        return sum(rewards), frames

    def selection(self, env, user, services):
        pass


# Baseline

class RandomAlgorithm(BaselineAlgorithm):
    def selection(self, env, user, services):
        return np.random.choice(services)


class FIFOAlgorithm(BaselineAlgorithm):
    def selection(self, env, user, services):
        return services[0]


class NearestGreedyAlgorithm(BaselineAlgorithm):
    def selection(self, env, user, services):
        nearest = None
        distance = float('inf')
        for service in services:
            if not service.user:
                service_distance = user.distance(service)
                if distance > service_distance:
                    distance = service_distance
                    nearest = service
        return nearest


class FarthestAlgorithm(BaselineAlgorithm):
    def selection(self, env, user, services):
        farthest = None
        distance = 0
        for service in services:
            if not service.user:
                service_distance = user.distance(service)
                if distance < service_distance:
                    distance = service_distance
                    farthest = service
        return farthest


class WallFarthestAlgorithm(BaselineAlgorithm):
    def selection(self, env, user, services):
        farthest = None
        distance = 0

        blocked = [service for service in services if env.block(service.coordinate, user.coordinate)]
        services = blocked if blocked else services

        for service in services:
            if not service.user:
                service_distance = user.distance(service)
                if distance < service_distance:
                    distance = service_distance
                    farthest = service
        return farthest


# Benchmark

class WallNearestGreedyAlgorithm(BaselineAlgorithm):
    def selection(self, env, user, services):
        nearest = None
        distance = float('inf')
        for service in services:
            if not service.user and not env.block(service.coordinate, user.coordinate):
                service_distance = user.distance(service)
                if distance > service_distance:
                    distance = service_distance
                    nearest = service

        if not nearest:
            return services[0]
        return nearest


class Chromosome:
    def __init__(self, num_actions, sequence):
        self.num_actions = num_actions
        self.sequence = list(sequence)

        self.score = None

    @staticmethod
    def random(num_actions: int, length: int):
        return Chromosome(num_actions, np.random.choice(range(num_actions), size=length, replace=True))

    def mutation(self, p):
        for i in range(len(self.sequence)):
            if np.random.random() < p:
                self.random_repair(i)
        self.score = None

    def crossover(self, other):
        cut = sorted(np.random.choice(range(1, len(self.sequence)-1), size=2, replace=False))
        child = Chromosome(
            self.num_actions,
            self.sequence[:cut[0]] + other.sequence[cut[0]:cut[1]] + self.sequence[cut[1]:]
        )
        assert len(self.sequence) == len(other.sequence) == len(child.sequence)
        return child

    def random_repair(self, index: int):
        self.sequence[index] = np.random.randint(0, self.num_actions)

    def similarity(self, other) -> float:
        return sum([1 if self[i] == other[i] else 0 for i in range(len(self.sequence))]) / len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]

    def __ge__(self, other):
        return self.score >= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __le__(self, other):
        return self.score <= other.score

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return all([self[i] == other[i] for i in range(len(self.sequence))])

    def __ne__(self, other):
        return any([self[i] != other[i] for i in range(len(self.sequence))])

    def __str__(self):
        return " ".join([str(gene) for gene in self.sequence])


class GeneticAlgorithm(BaselineAlgorithm):
    def __init__(self, population_size: int, patience: int, mutation_probability: float, similarity_threshold: float):
        self.population_size = population_size
        self.patience = patience
        self.mutation_probability = mutation_probability

        self.similarity_threshold = similarity_threshold

        super().__init__()

    def run(self, env, iteration: int, steps: List[int], num_requests: int, record: bool):
        num_actions = env.num_services
        chromosome_length = num_requests
        info("Genetic algorithm %d" % chromosome_length)
        if num_requests <= 0:
            return 0, []

        generation = 0
        rewards = []

        # Initial population
        population = [Chromosome.random(num_actions, chromosome_length) for _ in range(self.population_size)]

        while True:
            assert len(population) == self.population_size

            # Evaluate
            for chromosome in population:
                self.evaluation(env, chromosome, iteration, steps, False)

            # Sort
            population.sort()
            population.reverse()
            reward = population[0].score
            similarity = (sum([
                sum([p.similarity(o) for p in population]) for o in population
            ]) - len(population)) / (len(population) * (len(population) - 1))
            info("Generation %d(%.2f): %.2f" % (generation, similarity, reward))
            if similarity > self.similarity_threshold:
                break

            generation += 1
            rewards.append(reward)
            if len(rewards) >= self.patience and all(x == rewards[-1] for x in rewards[-self.patience:]):
                break

            # Crossover
            fitness = [chromosome.score for chromosome in population]
            if not np.max(fitness) > np.min(fitness):
                break
            probability = (fitness - np.min(fitness)) / np.sum(fitness - np.min(fitness))

            next_population = [population[0]]
            for _ in range(int(self.population_size) - 1):
                parents = np.random.choice(population, size=2, replace=False, p=probability)
                next_population.append(parents[0].crossover(parents[1]))
            population = next_population

            # Mutation
            for chromosome in population[1:]:
                chromosome.mutation(self.mutation_probability)

        # Record the best
        frames = self.evaluation(env, population[0], iteration, steps, record)

        return population[0].score, frames

    @staticmethod
    def evaluation(env, chromosome: Chromosome, iteration: int, steps: List[int], record: bool):
        frames = []
        rewards = []
        counter = 0

        env.reset()

        for step in steps:
            env.resume(train=False, iteration=iteration, minute=step)

            requesters, available_services, context = env.observation()

            for requester in requesters:
                if len(available_services) > 0:
                    action = env.services[chromosome[counter]]
                    while action.user:
                        # Resolve conflict randomly
                        chromosome.random_repair(counter)
                        action = env.services[chromosome[counter]]
                    requester.acquire(action)
                    available_services.remove(action)
                    counter += 1

            env.after_step()

            reward, finished_provisions = env.step_summary(context)
            rewards.append(reward)

            if record:
                frames.append(env.render(step))

        assert counter == len(chromosome.sequence)
        assert chromosome.score is None or chromosome.score == sum(rewards)

        chromosome.score = sum(rewards)

        return frames

