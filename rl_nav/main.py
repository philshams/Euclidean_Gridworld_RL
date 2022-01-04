import numpy as np
import matplotlib.pyplot as plt

class RL_nav:
    def __init__(self):
        self.arena=np.array([[0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,1,0,0,0,0]])   # 1 ~ shelter/reward
        self.threat_zone      = [(0,2), (0,3), (0,4)] # locations from which to test the agent's ability to get to shelter
        self.actions          = ['N','S','W','E','NW','NE','SW','SE']
        self.num_timesteps    = 500
        self.value_funcs      = {key:[] for key in ['model-free', 'model-based', 'SR', 'FR', 'optimal']}
        self.learning_curves  = {key:[] for key in ['model-free', 'model-based', 'SR', 'FR', 'timesteps']}
        self.learning_curves['timesteps'] = np.arange(0,self.num_timesteps,10)

    def select_hyperparameters(self):
        self.alpha_, self.gamma_, self.epsilon_, self.lambda_ = 0.01, 0.8, 0.1, 0.8
        # TODO: do a grid search for optimal parameters in each setting

    def model_free_learning(self, initialization='zero', policy='random'):
        self.select_hyperparameters()
        self.initialize_exploration(initialization, strategy='model-free')
        for i in range(self.num_timesteps):
            self.take_action('model-free',policy)
            self.take_reward()
            self.calculate_td_error()
            self.update_eligibility_trace()
            self.update_value_func('model-free')
            if i in self.learning_curves['timesteps']:
                self.test_performance()

    # -------- EPISODE MECHANICS ----------------------------------------------------------------      
    def take_action(self, strategy='model-free', policy='random'):
        if policy=='random':
            action = np.random.choice(self.actions)
        elif policy=='epsilon-greedy':
            pass # TODO: epsilon-greedy policy
        elif policy=='greedy':
            neighboring_locs   = self.get_neighboring_locs()
            neighboring_values = self.get_neighboring_values(neighboring_locs)            
            action = self.actions[np.random.choice(np.where(neighboring_values==np.max(neighboring_values))[0])] # in a tie, select randomly among winners
        self.prev_loc = self.loc
        if 'N' in action:
            self.loc = max(0, self.loc[0]-1), self.loc[1]
        if 'S' in action:
            self.loc = min(len(self.arena)-1, self.loc[0]+1), self.loc[1]
        if 'W' in action:
            self.loc = self.loc[0], max(0, self.loc[1]-1)
        if 'E' in action:
            self.loc = self.loc[0], min(len(self.arena[0])-1, self.loc[1]+1)

    def take_reward(self):
        self.reward = self.arena[self.prev_loc]

    def calculate_td_error(self):
        self.td_error = self.reward + self.gamma_ * self.value_funcs['model-free'][self.loc] -  self.value_funcs['model-free'][self.prev_loc]

    def update_eligibility_trace(self):
        self.eligibility_trace*=self.gamma_*self.lambda_
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
        self.loc               = tuple(np.random.randint(len(self.arena), size=2))
        self.eligibility_trace = np.zeros_like(self.arena, dtype=float)
        self.learning_curves[strategy].append([])
        if initialization=='zero':
            self.value_funcs[strategy] = np.zeros_like(self.arena, dtype=float)
        elif initialization=='euclidean':
            pass # TODO: euclidean initialization

    def get_neighboring_locs(self): # locations corresponding to actions N,S,W,E,NW,NE,SW,SE
        return [(max(0, self.loc[0]-1), self.loc[1]),
                (min(len(self.arena)-1, self.loc[0]+1), self.loc[1]),
                (self.loc[0], max(0, self.loc[1]-1)),
                (self.loc[0], min(len(self.arena[0])-1, self.loc[1]+1)),
                (max(0, self.loc[0]-1), max(0, self.loc[1]-1)), 
                (max(0, self.loc[0]-1), min(len(self.arena[0])-1, self.loc[1]+1)), 
                (min(len(self.arena[0])-1, self.loc[0]+1), max(0, self.loc[1]-1)),
                (min(len(self.arena[0])-1, self.loc[0]+1), min(len(self.arena[0])-1, self.loc[1]+1))]
    
    def get_neighboring_values(self, neighboring_locs):
        neighboring_values = np.array([self.value_funcs['model-free'][loc] for loc in neighboring_locs])
        if self.loc[0] == 0: # don't allow agents to stay in the same position with greedy policy
            neighboring_values[np.where(['N' in action for action in self.actions])[0]] = 0
        if self.loc[0] == len(self.arena)-1: # don't allow agents to stay in the same position with greedy policy
            neighboring_values[np.where(['S' in action for action in self.actions])[0]] = 0
        if self.loc[1] == 0: # don't allow agents to stay in the same position with greedy policy
            neighboring_values[np.where(['W' in action for action in self.actions])[0]] = 0
        if self.loc[1] == len(self.arena)-1: # don't allow agents to stay in the same position with greedy policy
            neighboring_values[np.where(['E' in action for action in self.actions])[0]] = 0
        return neighboring_values
    
    # -------- PERFORMANCE TESTING ---------------------------------------------------------------- 
    def test_performance(self):
        exploration_locs = self.prev_loc, self.loc # store where the agent was located so this performance test doesn't disrupt exploration
        timesteps_to_shelter = 0
        for start_loc in self.threat_zone:
            self.reward = 0
            self.loc = start_loc
            while not self.reward:
                self.take_action('model-free','greedy')
                self.take_reward()
                timesteps_to_shelter += 1/len(self.threat_zone) # get the avg timesteps from each state in threat zone
        self.learning_curves['model-free'][-1].append(np.round(timesteps_to_shelter, 1))
        self.prev_loc, self.loc = exploration_locs[0], exploration_locs[1] # reset the agent to where it was during exploration

    def display_value_func(self):
        plt.figure()
        plt.imshow(self.value_funcs['model-free'])
        plt.show()

    def display_learning_curve(self):
        x = self.learning_curves['timesteps']
        y = np.mean(self.learning_curves['model-free'], axis=0)
        y_err = np.std(self.learning_curves['model-free'], axis=0)
        plt.figure()
        plt.fill_between(x, y+y_err, y-y_err, alpha = 0.5)
        plt.plot(x, y)
        plt.xlabel('timesteps of random exploration')
        plt.ylabel('num steps, threat zone to shelter')
        plt.title('model free learning, open field environment')
        plt.show()