import torch
from Algorithm import Algorithm


class PPO(Algorithm):

    def __init__(self, actor, environment, plotter=None, render=False):
        super(PPO, self).__init__(actor, environment, plotter, render)
        self.clip_ratio = 0.2

    def compute_loss(self, observations, actions, advantage):
        log_probabilities = self.actor.get_policy().get_log_probs(observations, actions)
        old_log_probabilities = log_probabilities.clone().detach()
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss = -(torch.min(ratio, clipped_ratio) * advantage).mean()
        return loss
