from rl_nav.model import Euclidean_Gridworld_RL

class Performance_Test(Euclidean_Gridworld_RL):
    # -------- HIGH-LEVEL FUNC ----------------------------------------------------------------  
    def test_agent_performance(self, purpose = 'update learning curve'):
        self.reset_agent_location_for_testing()
        self.initialize_episode_figure(purpose)
        for _ in range(self.env.size):
            self.take_action('greedy')
            if self.loc==self.prev_loc:
                print('waerawefj')
            self.time_to_reward += self.compute_dwell_time()
            if self.take_reward(): break
            self.plot_action(purpose)
        self.plot_final_location(purpose)
        self.update_learning_curve(purpose)
        self.reset_agent_location_for_training()

    # -------- PERFORMANCE TESTING MECHANICS-------------------------------------------------------          
    def update_learning_curve(self, purpose):
        if purpose=='update learning curve':
            self.learning_curves[-1].append(self.time_to_reward)

    def initialize_episode_figure(self, purpose):
        if purpose=='display an episode': 
            plt.figure()
            plt.axis('off')
            plt.imshow(self.value_func, zorder=0) # plot the current value func as background
            plt.scatter(self.loc[1], self.loc[0], color='white', zorder=2) # plot start loc in white
            # show the obstacle in white
            for obstacle_loc in np.argwhere(self.env=='X'):
                white_square = patches.Rectangle(obstacle_loc[::-1]-.5, 1, 1, facecolor='white')
                plt.gca().add_patch(white_square)             

    def plot_action(self, purpose):
        if purpose=='display an episode': 
            plt.scatter(self.loc[1], self.loc[0], color='red')
            plt.plot([self.prev_loc[1], self.loc[1]],[self.prev_loc[0], self.loc[0]], color='red', alpha=0.6, zorder=1)
            plt.pause(.05)

    def plot_final_location(self, purpose):
        if purpose=='display an episode': 
            if self.reward: 
                color = 'green'
            else:
                color = 'gray'
            plt.scatter(self.prev_loc[1], self.prev_loc[0], s=75, color=color,zorder=99)
            plt.show()
    
    def reset_agent_location_for_testing(self):
        self.loc_cached, self.prev_loc_cached = self.loc, self.prev_loc 
        self.loc, self.prev_loc = tuple(np.argwhere(self.start)[0]), None
        self.time_to_reward = 0

    def reset_agent_location_for_training(self):
        self.loc, self.prev_loc = self.loc_cached, self.prev_loc_cached

    def display_learning_curve(self):
        x_data, y_data, y_err = self.get_data_for_learning_curve_plot()
        self.initialize_learning_curve_figure()
        plt.plot(x_data, y_data)
        plt.fill_between(x_data, y_data+y_err, y_data-y_err, alpha = 0.5)
        plt.show()

    def get_data_for_learning_curve_plot(self):        
        x_data = self.trials_to_test
        y_data = np.mean(self.learning_curves, axis=0)
        y_err = np.std(self.learning_curves, axis=0) / len(self.trials_to_test)**.5
        return x_data, y_data, y_err

    def initialize_learning_curve_figure(self):
        plt.figure()
        plt.xlabel('Timesteps of random exploration')
        plt.ylabel('Total path length from test zone to reward')
        plt.title(f'Learning curve, model-free agent with an obstacle')