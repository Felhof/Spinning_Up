from Algorithm import Algorithm


class VanillaPolicyGradient(Algorithm):

    def __init__(self, actor, environment, plotter=None, render=False):
        super(VanillaPolicyGradient, self).__init__(actor, environment, plotter, render)

    def compute_loss(self, observations, actions, weights):
        log_probabilities = self.actor.get_policy().get_log_probs(observations, actions)
        loss = -(log_probabilities.squeeze(-1) * weights).mean()
        return loss
