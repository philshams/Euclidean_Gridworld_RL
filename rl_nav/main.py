import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_nav.config import environment, rewards, test_loc, policy
from rl_nav.config import alpha_, gamma_, epsilon_, lambda_
from rl_nav.config import total_num_timesteps, test_every_N_trials, num_experiments

class Euclidean_Gridworld_RL:
    def __init__(self):
        self.num_timesteps   = total_num_timesteps
        self.num_experiments = num_experiments
        self.env             = environment
        self.rewards         = rewards
        self.start           = test_loc   
        self.policy          = policy   
        self.trials_to_test  = np.arange(0, total_num_timesteps, test_every_N_trials)
        self.actions         = ['N','S','W','E','NW','NE','SW','SE']
        self.learning_curves = []
        self.alpha_   = alpha_
        self.gamma_   = gamma_
        self.epsilon_ = epsilon_
        self.lambda_  = lambda_
        print( " - Generated the Euclidean Gridworld Semi-Markov Decision Process")
        print(f" - Ready for model-free learning using {self.policy} exploration")      

    # -------- HIGH-LEVEL FUNCS ----------------------------------------------------------------  
    def generate_learning_curve(self):
        for _ in tqdm(range(num_experiments)):
            self.model_free()
        self.display_learning_curve()

    def display_episode(self):
        pass

    def model_free(self):
        self.initialize_exploration()
        for i in range(self.num_timesteps):
            self.take_action(policy)
            self.compute_dwell_time()
            self.take_reward()
            self.calculate_td_error()
            self.update_eligibility_trace()
            self.update_value_func()
            if i in self.trials_to_test:
                self.test_performance()

    def successor_representation(self):
        # TODO: implement a successor representation agent as an alternative to model-free learning
        pass

    # -------- EPISODE MECHANICS ----------------------------------------------------------------      
    def take_action(self, policy='random'):
        self.select_action(policy)
        self.prev_loc = self.loc
        if 'N' in self.action: self.loc = self.go_north(self.loc)
        if 'S' in self.action: self.loc = self.go_south(self.loc)
        if 'W' in self.action: self.loc = self.go_west(self.loc)
        if 'E' in self.action: self.loc = self.go_east(self.loc)
        # if the agent runs into an obstacle, don't change location
        if self.env[self.loc]=='X': 
            self.loc = self.prev_loc

    def select_action(self, policy='random'):
        if policy=='random':
            self.action = np.random.choice(self.actions)
        elif policy=='epsilon-greedy':
            if np.random.random() < self.epsilon_:
                self.select_action('random')
            else:
                self.select_action('greedy')
            return
        elif policy=='greedy': 
            neighboring_locs   = self.get_neighboring_locs()
            neighboring_values = self.get_neighboring_values(neighboring_locs)     
            # in a tie, select randomly among highest-valued states       
            self.action = self.actions[np.random.choice(np.where(neighboring_values==np.max(neighboring_values))[0])] 

    def go_north(self, loc):
        return max(0, loc[0]-1), loc[1]

    def go_south(self, loc):
        return min(len(self.env)-1, loc[0]+1), loc[1]

    def go_west(self, loc):
        return loc[0], max(0, loc[1]-1)

    def go_east(self, loc):
        return loc[0], min(len(self.env[0])-1, loc[1]+1)
        
    def compute_dwell_time(self):
        ''' 
        Dwell time makes this a *semi* Markov Decision process
        Dwell time here is proportional to the Euclidean distance bewteen state(t) and state (t+1)
        Unless state(t)==state(t+1), in which case Dwell time is set to 1.0 (same as a N,E,S,or W action)
        This makes agents maximize time-discounted expected reward by minimizing total Euclidean path length
        '''
        self.dwell_time      = max(1, ((self.loc[0]-self.prev_loc[0])**2+(self.loc[1]-self.prev_loc[1])**2)**.5)
        self.discount_factor = self.gamma_**self.dwell_time

    def take_reward(self):
        self.reward = self.rewards[self.prev_loc]
        return self.reward

    def calculate_td_error(self):
        self.td_error = self.reward + self.discount_factor * self.value_func[self.loc] -  self.value_func[self.prev_loc]

    def update_eligibility_trace(self):
        self.eligibility_trace*=self.discount_factor*self.lambda_
        self.eligibility_trace[self.prev_loc] += 1

    def update_value_func(self):
        self.value_func += self.alpha_*self.td_error*self.eligibility_trace

    def initialize_exploration(self):
        self.loc               = tuple(np.random.randint(len(self.env), size=2))
        self.prev_loc          = None
        self.eligibility_trace = np.zeros_like(self.env, dtype=float)
        self.value_func        = np.zeros_like(self.env, dtype=float)
        self.learning_curves.append([])

    def get_neighboring_locs(self): # locations corresponding to actions N,S,W,E,NW,NE,SW,SE
        return [self.go_north(self.loc), 
                self.go_south(self.loc), 
                self.go_east(self.loc), 
                self.go_west(self.loc),
                self.go_north(self.go_west(self.loc)),
                self.go_north(self.go_east(self.loc)),  
                self.go_south(self.go_west(self.loc)),
                self.go_south(self.go_east(self.loc))]
    
    def get_neighboring_values(self, neighboring_locs):
       return np.array([self.value_func[loc] for loc in neighboring_locs])
    
    # -------- LEARNING CURVE AND EPISODE DEMONSTRATION -------------------------------------------- 
    def test_performance(self):
        exploration_locs = self.prev_loc, self.loc # store where the agent was located so this performance test doesn't disrupt exploration
        avg_timesteps_to_shelter = 0
        start_zone = np.argwhere(self.start)
        for start_loc in start_zone:
            self.reward = 0
            timesteps_to_shelter = 0
            self.loc, self.prev_loc = tuple(start_loc), None
            plt.figure()
            plt.imshow(self.value_func, zorder=0)
            plt.scatter(self.loc[1], self.loc[0], color='white', zorder=2)

            for _ in range(self.env.size):
                self.take_action('greedy')
                self.take_reward()
                if self.reward: break
                timesteps_to_shelter += self.dwell_time

                plt.plot([self.prev_loc[1], self.loc[1]],[self.prev_loc[0], self.loc[0]], color='red', zorder=1)
                plt.scatter(self.loc[1], self.loc[0], color='red')
                plt.pause(.001)

            avg_timesteps_to_shelter += timesteps_to_shelter/len(start_zone)
            if self.reward:
                plt.scatter(self.prev_loc[1], self.prev_loc[0], color='green',zorder=99)
                plt.pause(.5)
            plt.close()    
            print(timesteps_to_shelter)
        self.learning_curves[-1].append(np.round(timesteps_to_shelter, 1))
        self.prev_loc, self.loc = exploration_locs[0], exploration_locs[1] # reset the agent to where it was during exploration

    def display_learning_curve(self):
        x = self.trials_to_test
        y = np.mean(self.learning_curves, axis=0)
        y_err = np.std(self.learning_curves, axis=0) / len(self.trials_to_test)**.5
        plt.figure()
        plt.fill_between(x, y+y_err, y-y_err, alpha = 0.5)
        plt.plot(x, y)
        plt.ylim([0,self.env.size + 10])
        plt.xlabel('Timesteps of random exploration')
        plt.ylabel('Total path length from test zone to reward')
        plt.title(f'Learning curve, model-free agent with an obstacle')
        plt.show()