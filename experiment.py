import time
import tensorflow as tf
import pandas as pd

from typing import List

from settings import record, record_detail, Configuration
from environment import Environment
from utils import info, save_gif, get_summary_path
from agent import ServiceAgent


def variable_summary(summary_writer, scope_name: str, variable_name: str, step: int, value):
    if record_detail:
        with summary_writer.as_default(step=step), tf.name_scope(scope_name):
            if isinstance(value, list) and len(value) > 0:
                values = [float(v) for v in value]
                with tf.name_scope(variable_name):
                    tf.summary.scalar('mean', tf.reduce_mean(values))
                    tf.summary.scalar('stddev', tf.math.reduce_std(values))
                    tf.summary.scalar('max', tf.reduce_max(values))
                    tf.summary.scalar('min', tf.reduce_min(values))
                    tf.summary.scalar('sum', tf.reduce_sum(values))
            elif isinstance(value, (int, float)):
                tf.summary.scalar(variable_name, value)


def calculate_score(lower: float, higher: float, value: float) -> float:
    assert lower <= higher
    if lower == higher:
        return 1.0 if value >= higher else 0.0
    return (value - lower) / (higher - lower)


class Experiment:
    def __init__(self, now: str, configuration: Configuration):
        self.now = now

        # Training
        self.exploration = configuration.exploration
        self.installation_size = configuration.installation_size

        # Iteration settings
        self.num_training_days = configuration.num_training_days
        self.num_testing_days = configuration.num_testing_days
        self.num_day_steps = configuration.num_day_steps
        self.num_testing_iterations = configuration.num_testing_iterations

        # Evaluation settings
        self.worst = configuration.worst
        self.benchmark = configuration.benchmark

        self.evaluation_dataframe = pd.DataFrame(columns=[
            'Environment', 'Day', 'Iteration', 'Agent', 'Users',
            'Performance Score', 'Reward', 'Worst', 'Baseline', 'Benchmark',
        ])
        self.time_dataframe = pd.DataFrame(columns=[
            'Environment', 'Day', 'Iteration', 'Agent', 'Users', 'Train',
            'Simulation Time', 'Mean Selection Time', 'Mean Training Time',
        ])

        # Run settings
        self.num_users = None
        self.agent_name = None

    @property
    def name(self):
        return f"{self.num_users}users{self.agent_name}"

    def run(self, env: Environment, agents: List[ServiceAgent], num_users: int, training: bool):
        self.agent_name = type(agents[0]).__name__.replace("Agent", "")
        self.num_users = num_users

        info(f"<<< {self.name} >>>")

        env.set_agents(agents)
        self.save_agent_model(env=env)

        if num_users != env.num_users:
            info(f"The number of the users {env.num_users} -> {num_users}")
            env.set_users(num_users)

        train_writer = tf.summary.create_file_writer(
            get_summary_path(now=self.now, index=env.index, name=self.name, filename='train')
        ) if record_detail and training else None
        test_writer = tf.summary.create_file_writer(
            get_summary_path(now=self.now, index=env.index, name=self.name, filename='test')
        ) if record_detail else None

        start = time.time()

        num_testing_steps = list(range(self.num_testing_days * self.num_day_steps))

        # Training
        if training:
            if self.installation_size > 0:
                env.installation(self.installation_size)

            for day in range(self.num_training_days):
                # Testing before each training session
                i = 0
                while i < self.num_testing_iterations:
                    if self.test(env=env, summary_writer=test_writer, day=day, iteration=i, steps=num_testing_steps):
                        i += 1

                # Training
                training_steps = list(range(day * self.num_day_steps, (day + 1) * self.num_day_steps))
                self.train(env=env, summary_writer=train_writer, day=day, iteration=0, steps=training_steps)
                self.save_environment_value_map(env=env, index=env.index, day=day)

        # Final testing
        i = 0
        while i < self.num_testing_iterations:
            if self.test(
                env=env, summary_writer=test_writer, day=self.num_training_days, iteration=i, steps=num_testing_steps
            ):
                i += 1

        info(f"-- Total {time.time() - start:.2f} seconds --")

    @staticmethod
    def step_summary(summary_writer, step, env, reward, rewards, available_services, requesters):
        variable_summary(summary_writer, 'Main', 'reward', step, float(reward))
        variable_summary(summary_writer, 'Main', 'reward_mean', step, sum(rewards) / len(rewards))
        variable_summary(
            summary_writer, 'Main', 'reward_moving_average_simple', step,
            sum(rewards[-100:]) / 100 if len(rewards) >= 100 else sum(rewards) / len(rewards),
        )

        variable_summary(summary_writer, 'Detail', 'satisfaction', step, reward.satisfaction)
        variable_summary(summary_writer, 'Detail', 'interference', step, reward.interference)

        variable_summary(summary_writer, 'Stat', 'users', step, len([user for user in env.users if user.active]))
        variable_summary(summary_writer, 'Stat', 'requesters', step, len(requesters))
        variable_summary(summary_writer, 'Stat', 'available_services', step, len(available_services))
        variable_summary(summary_writer, 'Stat', 'provision', step, [service.count for service in env.services])

    def train(self, env: Environment, summary_writer, day: int, iteration: int, steps: List[int]):
        info(f"[Env{env.index} {self.name}] (Training day {day}: iteration {iteration})")
        train_start = time.time()

        rewards = []
        num_requests = 0

        selection_time_list = []
        training_time_list = []

        env.reset()
        has_history = env.has_history(train=True, iteration=iteration, steps=steps)
        info(f">> Reuse history for the train ({steps[0]}-{steps[-1]})" if has_history
             else f">> Record new history for the train ({steps[0]}-{steps[-1]})")

        for step in steps:
            if has_history:
                env.resume(train=True, iteration=iteration, minute=step)
            else:
                env.step(step, steps[-1])
                env.freeze(train=True, iteration=iteration, minute=step)

            # Observation for request
            requesters, available_services, context = env.observation()

            # If request occur, perform selection
            for requester in requesters:
                if len(available_services) > 0:
                    num_requests += 1
                    selection_start = time.time()
                    action = requester.selection(available_services, context, exploration=self.exploration, train=True)
                    selection_time_list.append(time.time() - selection_start)
                    requester.acquire(action)
                    available_services.remove(action)

            env.after_step()

            # Summarize the step
            reward, finished_episodes = env.step_summary(context)
            rewards.append(reward)

            # Training
            loss_list = []
            satisfaction_error_list = []
            interference_error_list = []

            for episode in finished_episodes:
                training_start = time.time()
                losses, satisfaction_errors, interference_errors = episode.learn()
                training_time_list.append(time.time() - training_start)
                loss_list += losses
                satisfaction_error_list += satisfaction_errors
                interference_error_list += interference_errors

            variable_summary(summary_writer, 'Learning', 'loss', step, loss_list)
            variable_summary(
                summary_writer, 'Learning', 'memory', step, [service.agent.memory.length() for service in env.services]
            )

            variable_summary(summary_writer, 'Detail', 'satisfaction_error', step, satisfaction_error_list)
            variable_summary(summary_writer, 'Detail', 'interference_error', step, interference_error_list)

            # Summary
            self.step_summary(summary_writer, step, env, reward, rewards, available_services, requesters)

        variable_summary(summary_writer, 'Evaluation', 'training_time', day, training_time_list)
        self.record_time(
            env.index, day, iteration, self.agent_name,
            True, time.time() - train_start, sum(selection_time_list) / len(selection_time_list),
            sum(training_time_list) / len(training_time_list),
        )

    def test(self, env: Environment, summary_writer, day: int, iteration: int, steps: List[int]) -> bool:
        info(f"[Env{env.index} {self.name}] (Testing day {day}: iteration {iteration})")
        test_start = time.time()

        frames = []
        rewards = []
        num_requests = 0

        selection_time_list = []

        env.reset()
        has_history = env.has_history(train=False, iteration=iteration, steps=steps)
        info(f">> Reuse history for the test ({steps[0]}-{steps[-1]})" if has_history
             else f">> Record new history for the test ({steps[0]}-{steps[-1]})")

        for step in steps:
            if has_history:
                env.resume(train=False, iteration=iteration, minute=step)
            else:
                env.step(step, steps[-1])
                env.freeze(train=False, iteration=iteration, minute=step)

            # Observation for request
            requesters, available_services, context = env.observation()

            # If request occur, perform selection
            for requester in requesters:
                if len(available_services) > 0:
                    num_requests += 1
                    selection_start = time.time()
                    action = requester.selection(available_services, context, exploration=self.exploration, train=False)
                    selection_time_list.append(time.time() - selection_start)
                    requester.acquire(action)
                    available_services.remove(action)

            env.after_step()

            # Summarize the step
            reward, finished_episodes = env.step_summary(context)
            rewards.append(reward)

            if record:
                frames.append(env.render(step))

            # Step summaries
            self.step_summary(summary_writer, step, env, reward, rewards, available_services, requesters)

        save_gif(frames, get_summary_path(
            now=self.now, index=env.index, name=self.name, filename=f"TestMainDay{day:02}Iteration{iteration:02}.gif"
        ))

        reward_sum = sum(rewards)

        self.record_time(
            env.index, day, iteration, self.agent_name,
            False, time.time() - test_start, sum(selection_time_list) / len(selection_time_list), None
        )

        # Evaluation

        env.set_num_requests(iteration, num_requests)
        # Worst
        worst_reward = env.get_worst_reward(self.now, self.name, self.worst, iteration, steps)
        # Benchmark
        benchmark_reward = env.get_benchmark_reward(self.now, self.name, self.benchmark, iteration, steps)

        # Erase trace
        if benchmark_reward < worst_reward:
            info(f"Benchmark {benchmark_reward} <= Worst {worst_reward} failure")
            env.erase_history(iteration=iteration, training=False)
            return False

        # Summary
        assert benchmark_reward >= worst_reward
        score = calculate_score(worst_reward, benchmark_reward, value=reward_sum)

        self.record_evaluation(
            env.index, day, iteration, self.agent_name, score, reward_sum, worst_reward, benchmark_reward
        )

        return True

    def test_baseline(self, env: Environment, baseline, name: str):
        if baseline is None:
            return

        steps = list(range(self.num_testing_days * self.num_day_steps))

        for iteration in range(self.num_testing_iterations):
            worst_reward = env.get_worst_reward(self.now, self.name, self.worst, iteration, steps)
            benchmark_reward = env.get_benchmark_reward(self.now, self.name, self.benchmark, iteration, steps)

            reward, frames = baseline.run(env, iteration, steps, env.testing_num_requests[iteration], record)
            save_gif(frames, get_summary_path(
                now=self.now, index=env.index, name=self.name, filename=f"TestBaseline{iteration}.gif"
            ))

            score_baseline = calculate_score(worst_reward, benchmark_reward, value=reward)
            for day in range(self.num_training_days + 1):
                self.record_evaluation(
                    env.index, day, iteration, name,
                    score_baseline, reward, worst_reward, benchmark_reward
                )

    def record_evaluation(self, index, day, iteration, agent, score, reward_sum, worst, benchmark):
        info(
            f"Rewards: {reward_sum:.2f}, Worst: {worst:.2f}, Benchmark: {benchmark:.2f}, Score: {score * 100:.2f}%"
        )
        self.evaluation_dataframe = self.evaluation_dataframe.append({
            'Environment': index,
            'Day': day,
            'Iteration': iteration,
            'Agent': agent,
            'Users': self.num_users,
            'Performance Score': score,
            'Reward': reward_sum,
            'Worst': worst,
            'Benchmark': benchmark,
        }, ignore_index=True)

    def record_time(self, index, day, iteration, agent, train, simulation_time, selection_time, training_time):
        info(f"-- {'Train' if train else 'Test'} {simulation_time:.2f} seconds --")
        self.time_dataframe = self.time_dataframe.append({
            'Environment': index,
            'Day': day,
            'Iteration': iteration,
            'Agent': agent,
            'Users': self.num_users,
            'Train': train,
            'Simulation Time': simulation_time,
            'Mean Selection Time': selection_time,
            'Mean Training Time': training_time,
        }, ignore_index=True)

    def save_results(self):
        try:
            self.evaluation_dataframe.to_csv(get_summary_path(self.now, filename=f'evaluation_{self.now}.csv'))
            self.time_dataframe.to_csv(get_summary_path(self.now, filename=f'time_{self.now}.csv'))
        except PermissionError:
            print('File not available')

    def save_agent_model(self, env: Environment):
        try:
            tf.keras.utils.plot_model(
                env.services[0].agent.model,
                to_file=get_summary_path(now=self.now, index=env.index, name=self.name, filename="model.png"),
                show_shapes=True, expand_nested=True, show_layer_activations=True, show_layer_names=True,
                dpi=300, rankdir='TB'
            )
        except ImportError as e:
            info(e)

    def save_environment_value_map(self, env: Environment, index: int, day: int):
        for service in env.services:
            value_map = env.render_value_map(service)
            value_map.save(get_summary_path(
                now=self.now, index=index, name=self.name, filename=f"Service{service.sid:02}Day{day:02}.png"
            ))
