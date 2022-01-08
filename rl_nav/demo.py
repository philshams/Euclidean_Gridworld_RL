from rl_nav.model import Euclidean_Gridworld_RL
from rl_nav.config import timestep_to_display

if __name__=='__main__':       
    simulation = Euclidean_Gridworld_RL()
    print(f" - Running one experiment and displaying the test episode after {timestep_to_display} training steps")
    simulation.model_free_experiment(display_an_episode=True)