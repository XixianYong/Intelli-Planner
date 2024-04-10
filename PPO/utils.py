from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import math


def train_on_policy_agent(env, agent, num_episodes, LLM_mode = False):
    reward_list = dict()
    reward_list['service_reward_list'] = []
    reward_list['ecology_reward_list'] = []
    reward_list['economic_reward_list'] = []
    reward_list['equity_reward_list'] = []
    reward_list['satisfaction_reward_list'] = []
    reward_list['return_list'] = []
    log_info = dict()
    log_info['train_best_episode'] = 0
    log_info['train_best_service_reward'] = 0
    log_info['train_best_ecology_reward'] = 0
    log_info['train_best_economic_reward'] = 0
    log_info['train_best_equity_reward'] = 0
    log_info['train_best_satisfaction_reward'] = 0
    log_info['train_best_return'] = 0
    log_info['train_best_plan'] = []
    log_info['train_best_type_ratio'] = dict()

    alpha = 0.995

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                LLM_probs = math.pow(alpha, i_episode + num_episodes / 10 * i)

                service_return = 0
                ecology_return = 0
                economic_return = 0
                equity_return = 0
                satisfaction_return = 0
                episode_return = 0
                transition_dict = {'action_state': [],
                                   'states': [],
                                   'actions': [],
                                   'next_states': [],
                                   'rewards': [],
                                   'dones': [],
                                   'critic_states': [],
                                   'next_critic_states': []}
                state = env.reset()
                done = False

                service_reward_last = 0
                ecology_reward_last = 0
                economic_reward_last = 0
                equity_reward_last = 0
                satisfaction_reward_last = 0
                total_reward_last = 0
                while not done:
                    if random.random() < LLM_probs and LLM_mode:
                        action_state = env.get_action_state()
                        action = agent.take_action_from_LLM(env, action_state)
                    else:
                        action_state = env.get_action_state()
                        action = agent.take_action(action_state)

                    old_state = deepcopy(state)
                    next_state, reward_info, done, critic_state, next_critic_state = env.step(action)
                    transition_dict['action_state'].append(np.array(action_state))
                    transition_dict['states'].append(np.array(old_state))
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(np.array(next_state))
                    transition_dict['rewards'].append(reward_info['total_reward'] - total_reward_last)
                    transition_dict['dones'].append(done)
                    transition_dict['critic_states'].append(critic_state)
                    transition_dict['next_critic_states'].append(next_critic_state)

                    # reward
                    service_return += reward_info['service_reward'] - service_reward_last
                    service_reward_last = reward_info['service_reward']
                    ecology_return += reward_info['ecology_reward'] - ecology_reward_last
                    ecology_reward_last = reward_info['ecology_reward']
                    economic_return += reward_info['economic_reward'] - economic_reward_last
                    economic_reward_last = reward_info['economic_reward']
                    equity_return += reward_info['equity_reward'] - equity_reward_last
                    equity_reward_last = reward_info['equity_reward']
                    satisfaction_return += reward_info['satisfaction_reward'] - satisfaction_reward_last
                    satisfaction_reward_last = reward_info['satisfaction_reward']
                    episode_return += reward_info['total_reward'] - total_reward_last
                    total_reward_last = reward_info['total_reward']

                    state = next_state

                reward_list['service_reward_list'].append(service_return)
                reward_list['ecology_reward_list'].append(ecology_return)
                reward_list['economic_reward_list'].append(economic_return)
                reward_list['equity_reward_list'].append(equity_return)
                reward_list['satisfaction_reward_list'].append(satisfaction_return)
                reward_list['return_list'].append(episode_return)
                agent.update(transition_dict)

                if log_info['train_best_return'] < episode_return:
                    log_info['train_best_episode'] = i_episode + num_episodes / 10 * i
                    log_info['train_best_service_reward'] = service_return
                    log_info['train_best_ecology_reward'] = ecology_return
                    log_info['train_best_economic_reward'] = economic_return
                    log_info['train_best_equity_reward'] = equity_return
                    log_info['train_best_satisfaction_reward'] = satisfaction_return
                    log_info['train_best_return'] = episode_return
                    log_info['train_best_plan'] = transition_dict['next_states'][-1]
                    log_info['train_best_type_ratio'] = env.get_type_ratio()

                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(reward_list['return_list'][-20:])})
                pbar.update(1)

    return reward_list, log_info


def get_final_plan(env, agent):
    eval_log = dict()
    eval_log['eval_service_reward'] = 0
    eval_log['eval_ecology_reward'] = 0
    eval_log['eval_economic_reward'] = 0
    eval_log['eval_equity_reward'] = 0
    eval_log['eval_satisfaction_reward'] = 0
    eval_log['eval_return'] = 0
    eval_log['eval_plan'] = []
    eval_log['eval_type_ratio'] = dict()

    episode_return = 0
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    state = env.reset()
    done = False
    reward_last = 0
    while not done:
        action_state = env.get_action_state()
        action = agent.take_action(action_state, mode='eval')
        old_state = deepcopy(state)
        next_state, reward_info, done, _, _ = env.step(action)
        transition_dict['states'].append(np.array(old_state))
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(np.array(next_state))
        transition_dict['rewards'].append(reward_info['total_reward'] - reward_last)
        transition_dict['dones'].append(done)
        episode_return += reward_info['total_reward'] - reward_last
        reward_last = reward_info['total_reward']
        state = next_state
        if done:
            eval_log['eval_service_reward'] = reward_info['service_reward']
            eval_log['eval_ecology_reward'] = reward_info['ecology_reward']
            eval_log['eval_economic_reward'] = reward_info['economic_reward']
            eval_log['eval_equity_reward'] = reward_info['equity_reward']
            eval_log['eval_satisfaction_reward'] = reward_info['satisfaction_reward']
            eval_log['eval_return'] = episode_return
            eval_log['eval_plan'] = transition_dict['next_states'][-1]
            eval_log['eval_type_ratio'] = env.get_type_ratio()
    return eval_log


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    service_reward_list = []
    ecology_reward_list = []
    economic_reward_list = []
    equity_reward_list = []
    return_list = []
    log_info = dict()
    log_info['train_best_episode'] = None
    log_info['train_best_service_reward'] = None
    log_info['train_best_ecology_reward'] = None
    log_info['train_best_economic_reward'] = None
    log_info['train_best_equity_reward'] = None
    log_info['train_best_return'] = None
    log_info['train_best_plan'] = None
    log_info['train_best_type_ratio'] = None

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                service_return = 0
                ecology_return = 0
                economic_return = 0
                equity_return = 0
                episode_return = 0
                state = env.reset()
                done = False

                service_reward_last = 0
                ecology_reward_last = 0
                economic_reward_last = 0
                equity_reward_last = 0
                total_reward_last = 0
                while not done:
                    action = agent.take_action(state)
                    old_state = deepcopy(state)
                    next_state, reward_info, done = env.step(action)
                    replay_buffer.add(old_state, action, reward_info['total_reward'] - total_reward_last, next_state, done)

                    # reward
                    service_return += (reward_info['service_reward'] - service_reward_last)
                    service_reward_last = reward_info['service_reward']
                    ecology_return += (reward_info['ecology_reward'] - ecology_reward_last)
                    ecology_reward_last = reward_info['ecology_reward']
                    economic_return += (reward_info['economic_reward'] - economic_reward_last)
                    economic_reward_last = reward_info['economic_reward']
                    equity_return += (reward_info['equity_reward'] - equity_reward_last)
                    equity_reward_last = reward_info['equity_reward']
                    episode_return += (reward_info['total_reward'] - total_reward_last)
                    total_reward_last = reward_info['total_reward']

                    state = next_state

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)

                service_reward_list.append(service_return)
                ecology_reward_list.append(ecology_return)
                economic_reward_list.append(economic_return)
                equity_reward_list.append(equity_return)
                return_list.append(episode_return)

                if (i_episode + 1) % 1 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-20:])})
                pbar.update(1)

    return service_reward_list, ecology_reward_list, economic_reward_list, equity_reward_list, return_list, log_info


def moving_average(a, window_size):
    smoothed_rewards = np.convolve(a, np.ones(window_size) / window_size, mode='valid')
    return smoothed_rewards


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

