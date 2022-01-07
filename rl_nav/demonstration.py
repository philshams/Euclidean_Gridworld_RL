from rl_nav.main import Euclidean_Gridworld_RL
from rl_nav.config import timestep_to_display

if __name__=='__main__':       
    simulation = Euclidean_Gridworld_RL()
    print(f" - Running one experiment to display a test episode after {timestep_to_display} timesteps of training")
    simulation.model_free_experiment(display_an_episode=True)