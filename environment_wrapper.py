import gym
import numpy as np


def one_hot_encode(scalar, n):
    t = np.zeros(n)
    t[int(scalar)] = 1
    return t


# Wrap gym environment so that step can be called in a consistent manner
class EnvironmentWrapper:

    def __init__(self, environment_id):
        self.env = gym.make(environment_id)
        self.action_transformer = lambda act: act
        if isinstance(self.env.action_space, gym.spaces.Box) \
                and (self.env.action_space.shape[0] == 1):
            self.action_transformer = lambda act: [act]
        self.observation_transformer = self.get_observation_transformer(self.env.observation_space)

    def get_observation_transformer(self, observation_space):
        def observation_transformer(obs): return obs
        if isinstance(observation_space, gym.spaces.Discrete):
            def observation_transformer(obs): return one_hot_encode(obs, self.env.observation_space.n)
        if isinstance(observation_space, gym.spaces.Tuple):
            def observation_transformer(obs):
                encoded_observations = []
                for idx, subspace in enumerate(observation_space):
                    encoded_observations.append(
                        one_hot_encode(obs[idx], subspace.n)
                    )
                return np.concatenate(encoded_observations)

        return observation_transformer

    # transform action based on the environments action space before calling step
    def step(self, action):
        obs, reward, done, info = self.env.step(self.action_transformer(action))
        return self.observation_transformer(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.observation_transformer(obs)

    # Offer every other method and property dynamically.
    def __getattr__(self, name):
        return getattr(self.__dict__['env'], name)

    def __setattr__(self, name, value):
        if name == "env":
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['env'], name, value)

    def __delattr__(self, name):
        delattr(self.__dict__['env'], name)
