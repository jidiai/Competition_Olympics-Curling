# -*- coding:utf-8  -*-
# Time  : 2022/1/29 上午10:48
# Author: Yahui Cui
import argparse
import os
import sys

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(Actor, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Args:
    action_space = 36
    state_space = 900


ppo_args = Args()
device = 'cpu'


class PPO:
    action_space = ppo_args.action_space
    state_space = ppo_args.state_space

    def __init__(self):
        super(PPO, self).__init__()
        self.args = ppo_args
        self.actor_net = Actor(self.state_space, self.action_space).to(device)

    def select_action(self, state, train=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
            # action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def load(self, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.dirname(__file__)
        print("base_path: ", base_path)
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')

        if os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')


# parser = argparse.ArgumentParser()
# parser.add_argument("--load_episode", default=300, type=int)
# args = parser.parse_args()
model = PPO()
model.load(episode=300)


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}


def my_controller(observation, action_space, is_act_continuous=False):
    obs_ctrl_agent = np.array(observation['obs']).flatten()
    action_ctrl_raw, action_prob = model.select_action(obs_ctrl_agent, False)
    # inference
    action_ctrl = actions_map[action_ctrl_raw]
    agent_action = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action

    return agent_action
