# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:53:35 2023

@author: ugo.pelissier
"""

import time
import torch
from collections import deque
import itertools
import numpy as np
import random

from utils.network import Network
from utils.env import make_atari
from utils.parameters import Parameters

from torch.utils.tensorboard import SummaryWriter

opt = Parameters().parse()
        
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
    env = make_atari("SpaceInvaders-v4", render_mode=None)
    
    replay_buffer = deque(maxlen=opt.BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)
    
    episode_count = 0
    
    summary_writer = SummaryWriter(opt.LOG_DIR)
    
    online_net = Network(env, device=device)
    target_net = Network(env, device=device)
    
    online_net = online_net.to(device)
    target_net = target_net.to(device)
    
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = torch.optim.Adam(online_net.parameters(), lr=opt.LR)
    
    # Initialize replay buffer
    obs, _ = env.reset()
    for _ in range(opt.MIN_REPLAY_SIZE):
        action = env.action_space.sample()
    
        new_obs, rew, done, _, _ = env.step(action)
        
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
                
        obs = new_obs
            
    # Main Training Loop
    eprew = []
    eplen = 0
    t_start = time.time()
    
    obs, _ = env.reset()
    for step in itertools.count():
        epsilon = np.interp(step, [0, opt.EPSILON_DECAY], [opt.EPSILON_START, opt.EPSILON_END])
        
        action = online_net.act(obs, epsilon)
        
        new_obs, rew, done, _, info = env.step(action)     
        eprew.append(rew)
        
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
        
        if done:
            epinfo = {"r": round(sum(eprew), 6), "l": len(eprew), "t": round(time.time() - t_start, 6)}
            
            eprew = []
            t_start = time.time()
            
            episode_count += 1
            
            obs, _ = env.reset()
    
        # Start gradient step
        transitions = random.sample(replay_buffer, opt.BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net, opt.GAMMA)
        
        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update Target Net
        if step % opt.TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())
    
        # Logging
        if step % opt.LOG_INTERVAL == 0:
    
            if (episode_count==0):
                epinfo = {"r": 0, "l": 0, "t": 0}
    
            print()
            print('Step', step)
            print('Rew', epinfo['r'])
            print('Ep Len', epinfo['l'])
            print('Episodes', episode_count)
            
            summary_writer.add_scalar('EpLen', epinfo['l'], global_step=step)
            summary_writer.add_scalar('Rew', epinfo['r'], global_step=step)
            
        # Save
        if step % opt.SAVE_INTERVAL == 0 and step !=0:
            print('Saving...')
            online_net.save(opt.SAVE_PATH)