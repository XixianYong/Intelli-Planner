cfg = {
    # policy network
    'policy_head_input_size': 52,
    'policy_head_hidden_size': [128, 64, 32, 8],

    # value network
    'value_head_input_size': 18,
    'value_head_hidden_size': [128, 64, 32, 8],

    'actor_lr': 1e-5,
    'critic_lr': 1e-5,
    'gamma': 0.98,
    'lmbda': 0.95,
    'epochs': 10,
    'eps': 0.2,
}
