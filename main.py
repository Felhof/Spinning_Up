from matplotlib import pyplot as plt
import numpy as np
import torch

from Actor import Actor
from environment_wrapper import EnvironmentWrapper
from vanilla_policy_gradient import VanillaPolicyGradient
from PPO import PPO

def main():
    env = EnvironmentWrapper("Pendulum-v0")
    actor = Actor(env)
    result_plotter = ResultPlotter(unit="Reward")
    ppo = PPO(actor=actor, environment=env, plotter=result_plotter, render=False)
    ppo.proximal_policy_optimisation(epochs=300, episodes=15, max_steps_per_episode=150)

    # plot learning curve
    result_plotter.plot_results()

    # evaluation
    obs = env.reset()
    total_reward = 0
    for _ in range(100):
        done = False
        while not done:
            action = actor.policy.get_greedy_action(torch.tensor(obs))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                obs = env.reset()
    print(total_reward)

    obs = env.reset()
    # show
    for _ in range(1000):
        env.render()
        action = actor.policy.get_greedy_action(torch.tensor(obs))
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    env.close()


class ResultPlotter:

    def __init__(self, unit="Reward"):
        self.episode_results = []
        self.epoch_results_mean = []
        self.epoch_results_worst = []
        self.epoch_results_best = []
        self.epoch_results_lower_q = []
        self.epoch_results_upper_q = []
        self.unit = unit

    def add_episode_result(self, result):
        self.episode_results.append(result)

    def end_epoch(self):
        worst, lower_q, upper_q, best = np.percentile(self.episode_results, [0, 25, 75, 100])
        mean = np.mean(self.episode_results)
        self.epoch_results_worst.append(worst)
        self.epoch_results_lower_q.append(lower_q)
        self.epoch_results_mean.append(mean)
        self.epoch_results_upper_q.append(upper_q)
        self.epoch_results_best.append(best)
        self.episode_results = []

    def plot_results(self, timescale="Epoch", logscale=False):
        fig, ax = plt.subplots()
        ax.set(xlabel=timescale, ylabel=self.unit, title="")
        ax.plot(
            range(len(self.epoch_results_worst)),
            self.epoch_results_worst,
            color='black',
            label="Min"
        )
        ax.plot(
            range(len(self.epoch_results_lower_q)),
            self.epoch_results_lower_q,
            color='green',
            label="Lower Quartile"
        )
        ax.plot(
            range(len(self.epoch_results_mean)),
            self.epoch_results_mean,
            color='blue',
            label="Mean"
        )
        ax.plot(
            range(len(self.epoch_results_upper_q)),
            self.epoch_results_upper_q,
            color='orange',
            label="Upper Quartile"
        )
        ax.plot(
            range(len(self.epoch_results_best)),
            self.epoch_results_best,
            color='red',
            label="Max"
        )
        plt.legend()
        if logscale:
            plt.yscale('log')
        plt.show()

main()

