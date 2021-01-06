import torch


class VanillaPolicyGradient:

    def __init__(self, actor, environment, plotter=None, render=False):
        self.actor = actor
        self.policy_optimiser = torch.optim.Adam(actor.get_policy().parameters(), lr=0.001)
        self.environment = environment
        self.plotter = plotter
        self.render = render

    def vanilla_policy_gradient(self, epochs, episodes, max_steps_per_episode=None):
        for _ in range(epochs):
            self.train_one_epoch(episodes, max_steps_per_episode)

    def train_one_epoch(self, num_episodes, max_steps_per_episode):
        observations = []
        actions = []
        rewards_to_go = []

        for _ in range(num_episodes):
            rewards = []
            obs = self.environment.reset()
            done = False
            step = 1
            while (not done) and (step <= max_steps_per_episode):
                if self.render:
                    self.environment.render()
                action = self.actor.get_policy().get_action(torch.as_tensor(obs, dtype=torch.float32))
                next_obs, reward, done, _ = self.environment.step(action)
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = next_obs
                step += 1

            rewards_to_go += self.compute_rewards_to_go(rewards)
            if self.plotter is not None:
                self.plotter.add_episode_result(sum(rewards))

        if self.plotter is not None:
            self.plotter.end_epoch()

        observations = torch.as_tensor(observations, dtype=torch.float32)
        rewards_to_go = torch.as_tensor(rewards_to_go, dtype=torch.float32)
        value_estimates = self.actor.estimate_value(observations)
        advantage = rewards_to_go - value_estimates

        self.policy_optimiser.zero_grad()
        policy_loss = self.compute_loss(
            observations=observations,
            actions=torch.as_tensor(actions, dtype=torch.long),
            weights=advantage
        )
        policy_loss.backward(retain_graph=True)
        self.policy_optimiser.step()

        self.actor.optimise_critic(rewards_to_go, value_estimates)

    def compute_loss(self, observations, actions, weights):
        log_probabilities = self.actor.get_policy().get_log_probs(observations, actions)
        loss = -(log_probabilities.squeeze(-1) * weights).mean()
        return loss

    def compute_rewards_to_go(self, rewards):
        rewards_to_go = [0] * len(rewards)

        for idx, r in enumerate(reversed(rewards)):
            rewards_to_go[-(idx+1)] = r + rewards_to_go[-idx]

        return rewards_to_go
