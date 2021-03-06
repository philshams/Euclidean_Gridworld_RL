import numpy as np
from tqdm import tqdm
from rl_nav.config import environment, rewards, test_loc
from rl_nav.config import alpha_, gamma_, epsilon_, lambda_
from rl_nav.config import total_num_timesteps, test_every_N_trials, num_experiments, timestep_to_display
from rl_nav.display_funcs import Performance

class Euclidean_Gridworld_RL:
    def __init__(self):
        self.num_timesteps   = total_num_timesteps
        self.num_experiments = num_experiments
        self.env             = environment
        self.rewards         = rewards
        self.start           = test_loc   
        self.step_to_display = timestep_to_display
        self.trials_to_test  = np.arange(0, total_num_timesteps, test_every_N_trials)
        self.actions         = ['N','S','W','E','NW','NE','SW','SE']
        self.learning_curves = []
        self.alpha_   = alpha_
        self.gamma_   = gamma_
        self.epsilon_ = epsilon_
        self.lambda_  = lambda_
        print( "Euclidean Gridworld Semi-Markov Decision Process...with an obstacle!")
        print(f" - Ready for model-free learning using random exploration")      

    # -------- HIGH-LEVEL FUNCS ----------------------------------------------------------------  
    def generate_learning_curve(self):
        print(f" - Running {self.num_experiments} experiments to produce an average learning curve")
        for _ in tqdm(range(num_experiments)):
            self.model_free_experiment(generate_learning_curve=True)
        Performance(self).display_learning_curve()

    def model_free_experiment(self, display_an_episode=False, generate_learning_curve=False):
        self.initialize_exploration()
        for i in range(self.num_timesteps):
            self.take_action('random')
            self.compute_dwell_time()
            self.take_reward()
            self.calculate_td_error()
            self.update_eligibility_trace()
            self.update_value_func()
            if generate_learning_curve and i in self.trials_to_test:
                self.learning_curves[-1].append(Performance(self).test())
            if display_an_episode and i == self.step_to_display:
                Performance(self).test('display')
                break

    def successor_representation_experiment(self):
        # TODO: implement a successor representation agent as an alternative to model-free learning
        pass

    # -------- EPISODE MECHANICS ----------------------------------------------------------------      
    def compute_dwell_time(self) -> float:
        ''' 
        Dwell time makes this a *semi* Markov Decision process
        Dwell time here is proportional to the Euclidean distance bewteen state(t) and state (t+1)
        (Unless state(t)==state(t+1), in which case Dwell time is set to 1.0 (same as a N,E,S,or W action))
        Thus the agent maximizes time-discounted expected reward by minimizing integrated Euclidean path length
        '''
        self.dwell_time      = max(1, ((self.loc[0]-self.prev_loc[0])**2+(self.loc[1]-self.prev_loc[1])**2)**.5)
        self.discount_factor = self.gamma_**self.dwell_time
        return self.dwell_time

    def take_action(self, policy='random') -> tuple:
        self.prev_loc = self.loc
        self.select_action(policy)
        if 'N' in self.action: self.go_north()
        if 'S' in self.action: self.go_south()
        if 'W' in self.action: self.go_west()
        if 'E' in self.action: self.go_east()
        self.check_for_boundaries()

    def select_action(self, policy='random'):
        if policy=='random':
            self.action = np.random.choice(self.actions)
        elif policy=='greedy': 
            neighboring_locs   = self.get_neighboring_locs()
            neighboring_values = self.get_neighboring_values(neighboring_locs)     
            # in a tie, select randomly among highest-valued states:       
            self.action = self.actions[np.random.choice(np.where(neighboring_values==np.max(neighboring_values))[0])] 

    def go_north(self, *args) -> tuple:
        self.loc = self.loc[0]-1, self.loc[1]
        return self.loc
  
    def go_south(self, *args) -> tuple:
        self.loc = self.loc[0]+1, self.loc[1]
        return self.loc

    def go_west(self, *args) -> tuple:
        self.loc =  self.loc[0],self.loc[1]-1
        return self.loc

    def go_east(self, *args) -> tuple:
        self.loc =  self.loc[0], self.loc[1]+1
        return self.loc

    def query_loc(self, *args):
        self.check_for_boundaries()
        query_loc, self.loc = self.loc, self.prev_loc
        return query_loc

    def check_for_boundaries(self):
        # if action took self.loc out of bounds of the environment, bring it back in bounds
        self.loc = tuple(min(max(0,x),len(self.env)-1) for x in self.loc)

        # if action hit an obsatcle (indicated by 'X' in self.env), undo that action
        if self.env[self.loc]=='X': 
            self.loc = self.prev_loc

    def take_reward(self) -> float:
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
        return [self.query_loc(self.go_north()), 
                self.query_loc(self.go_south()), 
                self.query_loc(self.go_west()), 
                self.query_loc(self.go_east()),
                self.query_loc(self.go_north(self.go_west())),
                self.query_loc(self.go_north(self.go_east())),  
                self.query_loc(self.go_south(self.go_west())),
                self.query_loc(self.go_south(self.go_east()))]
    
    def get_neighboring_values(self, neighboring_locs) -> np.ndarray:
        # Do not consider actions that do not change the agent's location, during the greedy policy
        return np.array([self.value_func[loc] if loc!=self.loc else -np.inf for loc in neighboring_locs])