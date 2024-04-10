import argparse
import json
import os
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
import geopandas as gpd
from langchain.chat_models import ChatOpenAI

from PPO.PPO import PPO
from PPO.utils import train_on_policy_agent, train_off_policy_agent, moving_average, get_final_plan, ReplayBuffer
from envs.urban_env import UrbanEnv
from utils.config import cfg
from models.get_model import create_model
from utils.data_preprocessing import load_features
from envs.urban_config import *
from LLM.agent import create_LLM_agent
from envs.urban_env import create_reward_agent

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train.')
parser.add_argument('--shp_file_path', type=str, help='Path to load shp file.')
parser.add_argument('--geo_info_path', type=str, help='Path to load masked_geo_info.')
parser.add_argument('--save_path', type=str, help='Path to save models and logs.')
parser.add_argument('--city_name', type=str, default='BEIJING', help='City name.')
parser.add_argument('--on_policy', type=bool, default=True, help='On-policy or off-policy.')
parser.add_argument('--minimal_size', type=int, default=1000, help='Minimal size of replay buffer for off-policy.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of replay buffer for off-policy.')
parser.add_argument('--LLM_mode', type=bool, default=True, help='Use LLM or not.')

args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# geo features
geo_info = load_features(args.geo_info_path)

# ppo model
policy_net, value_net = create_model(cfg)

# language model
llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
with open(r'E:\UP\LLM\prompt.json', 'r') as f:
    prompt_dict = json.load(f)
action_agent = create_LLM_agent(llm_model, prompt_dict)

# ppo agent
agent = PPO(policy_net, value_net, cfg, action_agent, device, args)

# reward agent and env
reward_agents = create_reward_agent(llm_model, prompt_dict)
env = UrbanEnv(geo_info, reward_agents)

#####################################
# training
# on-policy / off-policy training
if args.on_policy:
    reward_list, log_info = train_on_policy_agent(env, agent, args.episodes, args.LLM_mode)
else:
    replay_buffer = ReplayBuffer(5000)
    minimal_size = args.minimal_size
    batch_size = args.batch_size
    reward_list, log_info = train_off_policy_agent(env, agent, args.episodes, replay_buffer, minimal_size, batch_size)
#####################################

# evaluation
eval_log = get_final_plan(env, agent)
print(eval_log['eval_return'])
print(eval_log['eval_plan'])

# save models and logs
log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
save_path = args.save_path + '\\' + args.city_name + '\\' + log_time
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(policy_net.state_dict(), save_path + '\\policy_net.pth')
torch.save(value_net.state_dict(), save_path + '\\value_net.pth')

returns = reward_list
np.save(save_path + '\\train_returns.npy', returns)

with open(save_path + '\\log.txt', 'w') as f:
    f.write('train_on_policy: {}\n'.format(args.on_policy))
    f.write('train_with_LLM: {}\n'.format(args.LLM_mode))
    f.write('-----------------------------------------------------------\n')
    f.write('train_best_episode: {}\n'.format(log_info['train_best_episode']))
    f.write('train_best_service_reward: {}\n'.format(log_info['train_best_service_reward']))
    f.write('train_best_ecology_reward: {}\n'.format(log_info['train_best_ecology_reward']))
    f.write('train_best_economic_reward: {}\n'.format(log_info['train_best_economic_reward']))
    f.write('train_best_equity_reward: {}\n'.format(log_info['train_best_equity_reward']))
    f.write('train_best_satisfaction: {}\n'.format(log_info['train_best_satisfaction_reward']))
    f.write('train_best_return: {}\n'.format(log_info['train_best_return']))
    f.write('train_best_plan: {}\n'.format(log_info['train_best_plan']))
    f.write('train_best_type_ratio: {}\n'.format(log_info['train_best_type_ratio']))
    f.write('-----------------------------------------------------------\n')
    f.write('seed: {}\n'.format(args.seed))
    f.write('total_episodes: {}\n'.format(args.episodes))
    f.write('device: {}\n'.format(device))
    f.write('cfg: {}\n'.format(cfg))
    f.write('LIFE_CIRCLE_SIZE: {}\n'.format(LIFE_CIRCLE_SIZE))
    f.write('-----------------------------------------------------------\n')
    f.write('BUSINESS_COVERAGE_DEMANDS: {}\n'.format(BUSINESS_COVERAGE_DEMANDS))
    f.write('OFFICE_COVERAGE_DEMANDS: {}\n'.format(OFFICE_COVERAGE_DEMANDS))
    f.write('RECREATION_COVERAGE_DEMANDS: {}\n'.format(RECREATION_COVERAGE_DEMANDS))
    f.write('GREEN_COVERAGE_DEMANDS: {}\n'.format(GREEN_COVERAGE_DEMANDS))
    f.write('OPEN_SPACE_COVERAGE_DEMANDS: {}\n'.format(OPEN_SPACE_COVERAGE_DEMANDS))
    f.write('HOSPITAL_NUM: {}\n'.format(HOSPITAL_NUM))
    f.write('CLINIC_NUM: {}\n'.format(CLINIC_NUM))
    f.write('KINDERGARTEN_NUM: {}\n'.format(KINDERGARTEN_NUM))
    f.write('PRIMARY_SCHOOL_NUM: {}\n'.format(PRIMARY_SCHOOL_NUM))
    f.write('MIDDLE_SCHOOL_NUM: {}\n'.format(MIDDLE_SCHOOL_NUM))
    f.write('-----------------------------------------------------------\n')
    f.write('eval_service_reward: {}\n'.format(eval_log['eval_service_reward']))
    f.write('eval_ecology_reward: {}\n'.format(eval_log['eval_ecology_reward']))
    f.write('eval_economic_reward: {}\n'.format(eval_log['eval_economic_reward']))
    f.write('eval_equity_reward: {}\n'.format(eval_log['eval_equity_reward']))
    f.write('eval_satisfaction_reward: {}\n'.format(eval_log['eval_satisfaction_reward']))
    f.write('eval_return: {}\n'.format(eval_log['eval_return']))
    f.write('eval_plan: {}\n'.format(eval_log['eval_plan']))
    f.write('eval_type_ratio: {}\n'.format(eval_log['eval_type_ratio']))

shp_file = args.shp_file_path
gdf = gpd.read_file(shp_file)
gdf['type'] = eval_log['eval_plan']
cmap = plt.get_cmap('viridis', len(gdf['id'].unique()))
fig, ax = plt.subplots(figsize=(12, 8))
gdf['color'] = gdf['type'].map(color_mapping)
gdf.plot(ax=ax, color=gdf['color'], legend=True)

handles = []
labels = []
for key, value in FUNC_TYPES.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[key], markersize=10))
    labels.append(value)
plt.legend(handles, labels, title='Functional Types', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('eval_plan')
plt.savefig(save_path + '\\eval_plan_' + str(eval_log['eval_return']) + '.pdf')
plt.close()

window_size = 10

if args.on_policy:
    gdf['type'] = log_info['train_best_plan']
    cmap = plt.get_cmap('viridis', len(gdf['id'].unique()))
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf['color'] = gdf['type'].map(color_mapping)
    gdf.plot(ax=ax, color=gdf['color'], legend=True)
    plt.legend(handles, labels, title='Functional Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('train_best_plan')
    plt.savefig(save_path + '\\train_best_plan_' + str(log_info['train_best_return']) + '.pdf')
    plt.close()

episodes_list = list(range(len(reward_list['return_list'])))
mv_return = moving_average(reward_list['return_list'], window_size)
# plt.plot(episodes_list, reward_list['return_list'], alpha=0.5)
plt.fill_between(range(window_size-1, len(episodes_list)), reward_list['return_list'][window_size-1:], mv_return, color='orange', alpha=0.3)

plt.plot(mv_return, color='orange')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on Urban Planning')
plt.savefig(save_path + '\\return_ma.pdf')
plt.close()

episodes_list = list(range(len(reward_list['return_list'])))
mv_service_list = moving_average(reward_list['service_reward_list'], window_size)
mv_ecology_list = moving_average(reward_list['ecology_reward_list'], window_size)
mv_economic_list = moving_average(reward_list['economic_reward_list'], window_size)
mv_equity_list = moving_average(reward_list['equity_reward_list'], window_size)
mv_satisfaction_list = moving_average(reward_list['satisfaction_reward_list'], window_size)
plt.plot(mv_service_list, alpha=0.5)
plt.plot(mv_ecology_list, alpha=0.5)
plt.plot(mv_economic_list, alpha=0.5)
plt.plot(mv_equity_list, alpha=0.5)
plt.plot(mv_satisfaction_list, alpha=0.5)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('service, ecology, economic, equity, satisfaction rewards')
plt.legend(['service', 'ecology', 'economic', 'equity', 'satisfaction'])
plt.savefig(save_path + '\\service_ecology_economic_equity_satisfaction_return.pdf')
plt.close()

