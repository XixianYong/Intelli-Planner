import torch.nn.functional as F
from torch import nn


class UrbanPlanningPolicy(nn.Module):
    """
    Policy network for urban planning.
    """
    def __init__(self, cfg):
        super(UrbanPlanningPolicy, self).__init__()
        self.cfg = cfg
        self.policy_head = self.create_policy_head(cfg['policy_head_input_size'], cfg['policy_head_hidden_size'], 'policy')

    def create_policy_head(self, input_size, hidden_size, name):
        """
        Create the policy land_use head.
        """
        policy_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i])
                )
            if i < len(hidden_size) - 1:
                policy_head.add_module(
                    '{}_tanh_{}'.format(name, i),
                    nn.Tanh()
                )
        return policy_head

    def forward(self, x):
        x = self.policy_head(x)
        return F.softmax(x, dim=1)
