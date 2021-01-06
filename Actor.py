import gym
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def mlp(sizes, activations):
    layers = []
    connections = [
        (in_dim, out_dim) for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
    ]
    for connection, activation in zip(connections, activations):
        layers.append(torch.nn.Linear(connection[0], connection[1]))
        layers.append(activation())

    return torch.nn.Sequential(*layers)


class Policy(torch.nn.Module):

    def __init__(self, action_dim, observation_dim, hidden_sizes, activations):
        super(Policy, self).__init__()
        sizes = [observation_dim] + hidden_sizes + [action_dim]
        self.ff_stream = mlp(sizes, activations)

    def get_distribution(self, obs):
        return NotImplementedError

    def get_action(self, obs):
        return self.get_distribution(obs).sample().item()

    def get_greedy_action(self, obs):
        return NotImplementedError

    def get_log_probs(self, obs, actions):
        return NotImplementedError


class CategoricalPolicy(Policy):

    def __init__(self, action_dim, observation_dim, hidden_sizes, activations):
        super(CategoricalPolicy, self).__init__(action_dim, observation_dim, hidden_sizes, activations)

    def get_distribution(self, obs):
        return Categorical(logits=self.ff_stream(obs.float()))

    def get_greedy_action(self, obs):
        probs = self.get_distribution(obs).probs
        return torch.argmax(probs).item()

    def get_log_probs(self, obs, actions):
        probs = self.get_distribution(obs)
        return probs.log_prob(actions)


class GaussianPolicy(Policy):

    def __init__(self, action_dim, observation_dim, hidden_sizes, activations):
        super(GaussianPolicy, self).__init__(action_dim, observation_dim, hidden_sizes, activations)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def get_distribution(self, obs):
        mu = self.ff_stream(obs.float())
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def get_greedy_action(self, obs):
        dist = self.get_distribution(obs)
        return dist.mean.item()

    def get_log_probs(self, obs, actions):
        probs = self.get_distribution(obs)
        return probs.log_prob(actions).sum(axis=-1)


class Actor:

    def __init__(self, environment, hidden_sizes=[128], activations=[torch.nn.ReLU, torch.nn.Identity]):
        observation_dim = self.get_observation_dim(environment.observation_space)
        if isinstance(environment.action_space, gym.spaces.Box):
            self.policy = GaussianPolicy(environment.action_space.shape[0], observation_dim, hidden_sizes, activations)
        elif isinstance(environment.action_space, gym.spaces.Discrete):
            self.policy = CategoricalPolicy(environment.action_space.n, observation_dim, hidden_sizes, activations)

        vale_net_sizes = [observation_dim, 128, 1]
        vale_net_activations = [torch.nn.ReLU, torch.nn.Identity]
        self.value_net = mlp(vale_net_sizes, vale_net_activations)

    def get_observation_dim(self, observation_space):
        if isinstance(observation_space, gym.spaces.Box):
            observation_dim = observation_space.shape[0]
        elif isinstance(observation_space, gym.spaces.Discrete):
            observation_dim = observation_space.n
        elif isinstance(observation_space, gym.spaces.Tuple):
            observation_dim = sum([self.get_observation_dim(subspace) for subspace in observation_space])

        return observation_dim

    def get_policy(self):
        return self.policy

    def estimate_value(self, obs):
        return self.value_net.forward(obs)