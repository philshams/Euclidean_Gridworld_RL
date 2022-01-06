import numpy as np

total_num_timesteps = 1000

all_actions = ['N','S','W','E','NW','NE','SW','SE']

open_field_env =  np.array([['', '', '', '', '', '', '', '', ''], # ENVIRONMENT 1: str indicates obstacles
                            ['', '', '', '', '', '', '', '', ''], # here, there is no obstacle (open field)
                            ['', '', '', '', '', '', '', '', ''],
                            ['', '', '', '', '', '', '', '', ''],
                            ['', '', '', '', '', '', '', '', ''],
                            ['', '', '', '', '', '', '', '', ''],
                            ['', '', '', '', '', '', '', '', ''],
                            ['', '', '', '', '', '', '', '', ''],
                            ['', '', '', '', '', '', '', '', '']]) 

obstacle_env =    np.array([['', '',  '',  '',  '', '', '', '', ''], # ENVIRONMENT 2: str indicates obstacles
                            ['', '',  '',  '',  '', '', '', '', ''], # here, there is an obstacle in the middle
                            ['', '',  '',  '',  '', '', '', '', ''],
                            ['', '',  '',  '',  '', '', '', '', ''],
                            ['','X', 'X', 'X', 'X','X','X','X', ''],
                            ['', '',  '',  '',  '', '', '', '', ''],
                            ['', '',  '',  '',  '', '', '', '', ''],
                            ['', '',  '',  '',  '', '', '', '', ''],
                            ['', '',  '',  '',  '', '', '', '', '']])                             

reward_structure =np.array([[0,   0,  0,  0,  0,  0,  0,  0,   0], # REWARD STRUCTURE: 0 ~ no reward; 1 ~ shelter/reward
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  0,  0,  0,  0,   0],
                            [0,   0,  0,  0,  1,  0,  0,  0,   0]])                               

start_locations =  np.array([['', '', '', '', '!','', '', '', ''], # START ZONE: str '!' ~ locations used in test-time
                             ['', '', '', '', '', '', '', '', ''], # to see how many steps it takes to get a reward 
                             ['', '', '', '', '', '', '', '', ''],
                             ['', '', '', '', '', '', '', '', ''],
                             ['', '', '', '', '', '', '', '', ''],
                             ['', '', '', '', '', '', '', '', ''],
                             ['', '', '', '', '', '', '', '', ''],
                             ['', '', '', '', '', '', '', '', ''],
                             ['', '', '', '', '', '', '', '', '']]) 
                            
                            
initial_value_0  =np.zeros_like(reward_structure, dtype=float)                               

                                                      
x_idx1 = np.array([[i for i in range(len(reward_structure))] for _ in range(len(reward_structure))])
x_idx2 = np.array([[i for i in range(len(reward_structure)-1,-1,-1)] for _ in range(len(reward_structure))])
x_dist = abs(x_idx1 - x_idx2)/2
y_dist = np.array([[i for _ in range(len(reward_structure))] for i in range(len(reward_structure)-1,-1,-1)])   
initial_value_dist = -(x_dist**2 + y_dist**2)**.5 * 10**-17