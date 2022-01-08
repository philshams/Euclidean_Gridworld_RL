import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Performance:
    def __init__(self, Euclidean_Gridworld_RL):
        self.model = Euclidean_Gridworld_RL

    # --------HIGH-LEVEL FUNCS--------------------------------------------------------------------    
    def test(self, purpose = 'update learning curve'):
        '''
        If the purpose is 'update learning curve', just return the path length from start_loc to a reward
        If the purpose is 'display', display each state and action in a plot until the episode terminates
        '''
        self.reset_agent_location_for_testing()
        if purpose=='display': 
            self.initialize_episode_figure()
        for _ in range(self.model.env.size):
            self.model.take_action('greedy')
            self.model.time_to_reward += self.model.compute_dwell_time()
            if self.model.take_reward(): break
            if purpose=='display': 
                self.plot_action()
        if purpose=='display': 
            self.plot_final_location()
        self.reset_agent_location_for_training()
        return self.model.time_to_reward

    def display_learning_curve(self):
        x_data, y_data, y_err = self.get_data_for_learning_curve_plot()
        self.initialize_learning_curve_figure()
        plt.plot(x_data, y_data)
        plt.fill_between(x_data, y_data+y_err, y_data-y_err, alpha = 0.5)
        plt.show()

    # -------- PERFORMANCE TESTING MECHANICS-------------------------------------------------------          
    def reset_agent_location_for_testing(self):
        '''
        Cache the original agent location and previous location, so that performance testing does not disrupt training
        Set the agent's location to the test location '!'
        Initialize the time-to-reward
        '''
        self.model.loc_cached, self.model.prev_loc_cached = self.model.loc, self.model.prev_loc 
        self.model.loc, self.model.prev_loc = tuple(np.argwhere(self.model.start)[0]), None
        self.model.time_to_reward = 0

    def reset_agent_location_for_training(self):
        '''
        Reset the original agent location and previous location, so that performance testing does not disrupt training
        '''
        self.model.loc, self.model.prev_loc = self.model.loc_cached, self.model.prev_loc_cached

    def plot_action(self):
        plt.scatter(self.model.loc[1], self.model.loc[0], color='red')
        plt.plot([self.model.prev_loc[1], self.model.loc[1]],[self.model.prev_loc[0], self.model.loc[0]], color='red', alpha=0.6, zorder=1)
        plt.pause(.05)

    def plot_final_location(self):
        if self.model.reward: 
            color = 'green'
        else:
            color = 'gray'
        plt.scatter(self.model.prev_loc[1], self.model.prev_loc[0], s=75, color=color,zorder=99)
        plt.show()
    
    def get_data_for_learning_curve_plot(self):        
        x_data = self.model.trials_to_test
        y_data = np.mean(self.model.learning_curves, axis=0)
        y_err = np.std(self.model.learning_curves, axis=0) / len(self.model.trials_to_test)**.5
        return x_data, y_data, y_err       

    def initialize_learning_curve_figure(self):
        plt.figure()
        plt.xlabel('Timesteps of random exploration')
        plt.ylabel('Total path length from test zone to reward')
        plt.title(f'Learning curve, model-free agent with an obstacle')

    def initialize_episode_figure(self):
        plt.figure()
        plt.axis('off')
        plt.imshow(self.model.value_func, zorder=0) # plot the current value func as background
        plt.scatter(self.model.loc[1], self.model.loc[0], color='white', s=75, zorder=2) # plot start loc in white
        # show the obstacle in white
        for obstacle_loc in np.argwhere(self.model.env=='X'):
            white_square = patches.Rectangle(obstacle_loc[::-1]-.5, 1, 1, facecolor='white')
            plt.gca().add_patch(white_square)           