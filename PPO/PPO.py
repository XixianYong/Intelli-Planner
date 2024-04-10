import csv

from .utils import compute_advantage
import torch
import numpy as np
import torch.nn.functional as F

import sys

sys.path.append('.')
from LLM.agent import LLM_select_action


class PPO:
    """
    PPO-Clip
    """

    def __init__(self, policy_net, value_net, cfg, LLM, device, args):
        self.actor = policy_net.to(device)
        self.critic = value_net.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])

        self.gamma = cfg['gamma']
        self.lmbda = cfg['lmbda']
        self.epochs = cfg['epochs']
        self.eps = cfg['eps']
        self.LLM_agent = LLM
        self.device = device
        self.area_info = load_features(args.geo_info_path)
        self.area_middle = find_middle_area(self.area_info)

    def take_action(self, state, mode='train'):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # probs = self.action_masked(state, probs)
        if mode == 'train':
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            action_selected = action.item()
        elif mode == 'eval':
            action_selected = torch.argmax(probs).item()

        return action_selected

    def take_action_from_LLM(self, env, state, mode='train'):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        idx = env.state.index(0)
        live_circle_info_list = env.get_living_circle_info(idx)
        global_info_dict = env.get_global_info()
        if self.area_info[idx] < self.area_middle:
            area_type = 'small'
        else:
            area_type = 'large'
        action_candidates = LLM_select_action(area_type, live_circle_info_list, global_info_dict, self.LLM_agent)

        probs = self.actor(state)
        probs = self.action_masked(state, probs)
        if mode == 'train':
            for action in action_candidates:
                probs[0][action - 1] = probs[0][action - 1] * 2
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            action_selected = action.item()
        elif mode == 'eval':
            action_selected = torch.argmax(probs).item()
        return action_selected

    def action_masked(self, state, probs):
        state = state.tolist()[0]
        area_index = state.index(0)
        area = self.area_info[area_index]
        if area < self.area_middle:
            for i in range(len(probs[0])):
                if i not in [0, 1, 2, 3, 4, 5, 7]:
                    probs[0][i] = 0
        else:
            for i in range(len(probs[0])):
                if i not in [0, 1, 2, 3, 4, 5, 6, 7]:
                    probs[0][i] = 0
        return probs

    def update(self, transition_dict):
        action_state = torch.tensor(np.array(transition_dict['action_state']), dtype=torch.float).to(self.device)
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        critic_states = torch.tensor(transition_dict['critic_states'], dtype=torch.float).to(self.device)
        next_critic_states = torch.tensor(transition_dict['next_critic_states'], dtype=torch.float).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_critic_states) * (1 - dones)
        td_delta = td_target - self.critic(critic_states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(action_state).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(action_state).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(critic_states), td_target.detach()))
            # print('actor_loss:', actor_loss)
            # print('critic_loss:', critic_loss)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def load_features(csv_path):
    csv_reader = csv.reader(open(csv_path, encoding='utf-8'))
    features_list = []
    for line in csv_reader:
        if line[0] == '':
            pass
        else:
            features = float(line[3])
            features_list.append(features)
    return features_list


def find_middle_area(numbers):
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    split_index = len(sorted_numbers) // 2
    middle = sorted_numbers[split_index]

    return middle

