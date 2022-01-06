import numpy as np
import matplotlib.pyplot as plt
from rl_nav.config import open_field_env, obstacle_env, reward_structure, start_locations, total_num_timesteps, all_actions, initial_value_dist, initial_value_0

class RL_nav:
    def __init__(self, env):
        if env=='open field': self.env = open_field_env
        if env=='obstacle':   self.env = obstacle_env
        self.env_type        = env
        self.rewards         = reward_structure
        self.start           = start_locations
        self.num_timesteps   = total_num_timesteps
        self.actions         = all_actions
        self.value_funcs     = {key:[] for key in ['model-free', 'model-based', 'SR', 'FR', 'optimal']}
        self.learning_curves = {key:[] for key in ['model-free', 'model-based', 'SR', 'FR', 'timesteps']}
        self.learning_curves['timesteps'] = np.arange(0,self.num_timesteps,10)

    def select_hyperparameters(self):
        self.alpha_, self.gamma_, self.epsilon_, self.lambda_ = 0.01, 0.8, 0.1, 0.8
        # TODO: do a grid search for optimal parameters in each setting

    def model_free_learning(self, initialization='zero', policy='random'):
        self.select_hyperparameters()
        self.initialize_exploration(initialization, strategy='model-free')
        for i in range(self.num_timesteps):
            self.take_action(policy)
            self.take_reward()
            self.calculate_td_error()
            self.update_eligibility_trace()
            self.update_value_func('model-free')
            if i in self.learning_curves['timesteps']:
                self.test_performance()

    # -------- EPISODE MECHANICS ----------------------------------------------------------------      
    def take_action(self, policy='random'):
        if policy=='random':
            action = np.random.choice(self.actions)
        elif policy=='epsilon-greedy':
            if np.random.random() < self.epsilon_:
                self.take_action('random')
            else:
                self.take_action('greedy')
            return
        elif policy=='greedy': # in a tie, select randomly among highest-valued states
            neighboring_locs   = self.get_neighboring_locs()
            neighboring_values = self.get_neighboring_values(neighboring_locs)            
            action = self.actions[np.random.choice(np.where(neighboring_values==np.max(neighboring_values))[0])] 

        self.prev_loc      = self.loc

        if 'N' in action:
            self.loc = max(0, self.loc[0]-1), self.loc[1]
        if 'S' in action:
            self.loc = min(len(self.env)-1, self.loc[0]+1), self.loc[1]
        if 'W' in action:
            self.loc = self.loc[0], max(0, self.loc[1]-1)
        if 'E' in action:
            self.loc = self.loc[0], min(len(self.env[0])-1, self.loc[1]+1)

        if self.env[self.loc]=='X' or -1 in self.loc or len(self.env) in self.loc:
            self.loc = self.prev_loc # if the agent runs into an obstacle, it remains in the same position

        self.time_taken_by_action = max(1, ((self.loc[0]-self.prev_loc[0])**2+(self.loc[1]-self.prev_loc[1])**2)**.5)
        self.discount_factor      = self.gamma_**self.time_taken_by_action

    def take_reward(self):
        self.reward = self.rewards[self.prev_loc]

    def calculate_td_error(self):
        self.td_error = self.reward + self.discount_factor * self.value_funcs['model-free'][self.loc] -  self.value_funcs['model-free'][self.prev_loc]

    def update_eligibility_trace(self):
        self.eligibility_trace*=self.discount_factor*self.lambda_
        self.eligibility_trace[self.prev_loc] += 1

    def update_value_func(self, strategy='model-free'):
        if strategy=='model-free':
            self.value_funcs['model-free'] += self.alpha_*self.td_error*self.eligibility_trace
        elif strategy=='model-based':
            pass # TODO: model-based value func updating
        elif strategy=='SR':
            pass # TODO: SR value func updating
        elif strategy=='FR':
            pass # TODO: FR value func updating

    def initialize_exploration(self, initialization, strategy):
        self.loc               = tuple(np.random.randint(len(self.env), size=2))
        self.prev_loc          = None
        self.eligibility_trace = np.zeros_like(self.env, dtype=float)
        self.learning_curves[strategy].append([])
        if initialization=='zero':
            self.value_funcs[strategy] = initial_value_0.copy()
        elif initialization=='euclidean':
            self.value_funcs[strategy] = initial_value_dist.copy()

    def get_neighboring_locs(self): # locations corresponding to actions N,S,W,E,NW,NE,SW,SE
        return [(max(0, self.loc[0]-1), self.loc[1]),
                (min(len(self.env)-1, self.loc[0]+1), self.loc[1]),
                (self.loc[0], max(0, self.loc[1]-1)),
                (self.loc[0], min(len(self.env[0])-1, self.loc[1]+1)),
                (max(0, self.loc[0]-1), max(0, self.loc[1]-1)), 
                (max(0, self.loc[0]-1), min(len(self.env[0])-1, self.loc[1]+1)), 
                (min(len(self.env[0])-1, self.loc[0]+1), max(0, self.loc[1]-1)),
                (min(len(self.env[0])-1, self.loc[0]+1), min(len(self.env[0])-1, self.loc[1]+1))]
    
    def get_neighboring_values(self, neighboring_locs):
       return np.array([self.value_funcs['model-free'][loc] for loc in neighboring_locs])
    
    # -------- PERFORMANCE TESTING ---------------------------------------------------------------- 
    def test_performance(self):
        exploration_locs = self.prev_loc, self.loc # store where the agent was located so this performance test doesn't disrupt exploration
        avg_timesteps_to_shelter = 0
        start_zone = np.argwhere(self.start)
        for start_loc in start_zone:
            self.reward = 0
            timesteps_to_shelter = 0
            self.loc, self.prev_loc = tuple(start_loc), None
            # plt.figure()
            # plt.imshow(self.value_funcs['model-free'])
            while not self.reward:
                # plt.scatter(self.loc[0], self.loc[1], color='red')
                # plt.pause(.001)
                self.prev_prev_loc = self.prev_loc
                self.take_action('greedy')
                self.take_reward()
                timesteps_to_shelter += self.time_taken_by_action # get the avg timesteps from each state in threat zone
                if self.loc == self.prev_loc or self.loc==self.prev_prev_loc: 
                    self.take_action('random') # to avoid endless loops, don't allow agents to stay in the same position with greedy policy
                if timesteps_to_shelter > self.env.size:
                    break

                # plt.plot([self.prev_loc[0], self.loc[0]],[self.prev_loc[1], self.loc[1]], color='red')

            avg_timesteps_to_shelter += timesteps_to_shelter/len(start_zone)
            # plt.scatter(self.loc[0], self.loc[1], color='green')
            # plt.pause(.1)
            # plt.close()    
            # print(timesteps_to_shelter)
        self.learning_curves['model-free'][-1].append(np.round(timesteps_to_shelter, 1))
        self.prev_loc, self.loc = exploration_locs[0], exploration_locs[1] # reset the agent to where it was during exploration

    def display_value_func(self):
        plt.figure()
        plt.imshow(self.value_funcs['model-free'])
        plt.show()

    def display_learning_curve(self):
        x = self.learning_curves['timesteps']
        y = np.mean(self.learning_curves['model-free'], axis=0)
        y_err = np.std(self.learning_curves['model-free'], axis=0) / len(self.learning_curves['timesteps'])**.5
        plt.figure()
        plt.fill_between(x, y+y_err, y-y_err, alpha = 0.5)
        plt.plot(x, y)
        # plt.plot(x, np.array(self.learning_curves['model-free']).T)
        plt.ylim([0,self.env.size])
        plt.xlabel('timesteps of random exploration')
        plt.ylabel('num steps, threat zone to shelter')
        plt.title(f'model-free learning, {self.env_type}')
        plt.show()