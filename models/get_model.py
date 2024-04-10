from torch import nn

from models.policy import UrbanPlanningPolicy
from models.value import UrbanPlanningValue


def create_model(cfg):
    """
    Create policy and value network from a config file.

    Args:
        cfg: A config object.
    Returns:
        A tuple containing the policy network and the value network.
    """
    policy_net = UrbanPlanningPolicy(cfg)
    value_net = UrbanPlanningValue(cfg)

    return policy_net, value_net


class ActorCritic(nn.Module):
    """
    An Actor-Critic network for parsing parameters.

    Args:
        actor_net (nn.Module): actor network.
        value_net (nn.Module): value network.
    """

    def __init__(self, actor_net, critic_net):
        super().__init__()
        self.actor_net = actor_net
        self.critic_net = critic_net
