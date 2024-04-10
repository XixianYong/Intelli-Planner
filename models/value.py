from torch import nn


class UrbanPlanningValue(nn.Module):
    """
    Value network for urban planning.
    """
    def __init__(self, cfg):
        super(UrbanPlanningValue, self).__init__()
        self.cfg = cfg
        self.value_head = self.create_value_head(cfg['value_head_input_size'], cfg['value_head_hidden_size'], 'value')

    def create_value_head(self, input_size, hidden_size, name):
        """
        Create the value head.
        """
        value_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                value_head.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                value_head.add_module(
                    'linear_{}'.format(i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i])
                )
            if i < len(hidden_size) - 1:
                value_head.add_module(
                    'tanh_{}'.format(i),
                    nn.Tanh()
                )
        return value_head

    def forward(self, x):
        x = self.value_head(x)
        return x
