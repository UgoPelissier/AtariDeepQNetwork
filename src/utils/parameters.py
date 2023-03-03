# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:49:30 2023

@author: ugo.pelissier
"""

import optparse

class Parameters:
    def __init__(self):
        self.parser = optparse.OptionParser()
        
        self.parser.add_option('--batch_size', dest = 'BATCH_SIZE', type = int, help = '')
        self.parser.add_option('--buffer_size', dest = 'BUFFER_SIZE', type = int, help = '')
        self.parser.add_option('--min_replay_size', dest = 'MIN_REPLAY_SIZE', type = int, help = '')
        self.parser.add_option('--epsilon_decay', dest = 'EPSILON_DECAY', type = int, help = '')
        self.parser.add_option('--target_update_frequency', dest = 'TARGET_UPDATE_FREQ', type = int, help = '')
        self.parser.add_option('--save_interval', dest = 'SAVE_INTERVAL', type = int, help = '')
        self.parser.add_option('--log_interval', dest = 'LOG_INTERVAL', type = int, help = '')
        
        self.parser.add_option('--gamma', dest = 'GAMMA', type = float, help = '')
        self.parser.add_option('--epsilon_start', dest = 'EPSILON_START', type = float, help = '')
        self.parser.add_option('--epsilons_end', dest = 'EPSILON_END', type = float, help = '')
        self.parser.add_option('--learning_rate', dest = 'LR', type = float, help = '')
        
        self.parser.add_option('--save_path', dest = 'SAVE_PATH', type = str, help = '')
        self.parser.add_option('--log_dir', dest = 'LOG_DIR', type = str, help = '')
        
    def parse(self):
        self.options, _ = self.parser.parse_args()
        
        args = vars(self.options)
        
        print('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------\n')
        
        return self.options