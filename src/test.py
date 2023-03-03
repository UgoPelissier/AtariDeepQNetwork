# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:53:35 2023

@author: ugo.pelissier
"""

import torch
import itertools

from utils.env import make_atari
from utils.network import Network
from gymnasium.utils.save_video import save_video

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = make_atari("SpaceInvaders-v4", render_mode="rgb_array")
    
    net = Network(env, device)
    net = net.to(device)
    
    net.load('./model/model.pack')
    
    frames = []
    step_starting_index = 0
    episode_index = 0
    
    obs, _ = env.reset()
    beginning_episode = True
    for t in itertools.count():
        action = net.act(obs, 0.0)
    
        if beginning_episode:
            action = 1
            beginning_episode = False
    
        obs, rew, done, _, info = env.step(action)
        frames.append(env.render())
    
        if done:
            
            save_video(
               frames,
               "videos",
               fps=30,
               step_starting_index=step_starting_index,
               episode_index=episode_index
            )
            step_starting_index = t + 1
            episode_index += 1
            
            obs, _ = env.reset()
            beginning_episode = True
            
            break