import datetime
import logging
import gc
import shutil

from settings import Configuration, num_environments
from environment import Environment
from experiment import Experiment
from utils import info, get_summary_path
from baseline import FarthestAlgorithm, NearestGreedyAlgorithm, WallNearestGreedyAlgorithm


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    log_path = get_summary_path(now=now, filename=f"log_{now}.txt")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    configuration = Configuration()
    configuration.save(get_summary_path(now=now, filename=f"configuration_{now}.txt"))
    exp = Experiment(now=now, configuration=configuration)

    shutil.copy('plot.ipynb', get_summary_path(now=now, filename=f"plot_{now}.ipynb"))

    for index in range(num_environments):
        info(f">> Construct Environment {index}")
        env = Environment(index=index, configuration=configuration)
        env.save_preview(now=now)

        agents = {
            agent_class: configuration.construct_agents(agent_class=agent_class)
            for agent_class in configuration.agents
        }

        # Training
        for agent_class in configuration.agents:
            exp.run(env, agents[agent_class], num_users=configuration.num_users_training, training=True)

        # Baseline
        exp.test_baseline(env, FarthestAlgorithm(), "Farthest")
        exp.test_baseline(env, NearestGreedyAlgorithm(), "Nearest")
        exp.test_baseline(env, WallNearestGreedyAlgorithm(), "WallNearest")

        # Testing
        for num_users in configuration.num_users_testing:
            for agent_class in configuration.agents:
                exp.run(env, agents[agent_class], num_users=num_users, training=False)

            # Baseline
            exp.test_baseline(env, FarthestAlgorithm(), "Farthest")
            exp.test_baseline(env, NearestGreedyAlgorithm(), "Nearest")
            exp.test_baseline(env, WallNearestGreedyAlgorithm(), "WallNearest")

        # Save results
        exp.save_results()

        del env
        del agents
        gc.collect()
