# RL for navigation

## Overview
This is a repo for training reinforcement-learning agents is a gridworld, such that agents optimize for the shortest integrated Euclidean distance to rewards. Currently supported models:
- Tabular Q learning
- Tabular SARSA
- State-action Successor Representation
- Tile coding (with Q learning)
- Hierarchical state space (with Q learning or SARSA)
- Model-based tree search

## Usage
- Install the repo from the directory *...\Euclidean_Gridworld_RL* using ```pip install -e .```
- Modify the config files as needed in experiments/suites/...
- Run an experiment, e.g. with the command ```python run.py --mode parallel --seeds 10 --config_path suites/a_star/config.yaml --config_changes config_changes.py```
- Visualize results, e.g. with the command ```python post_process.py --plot_t --results_folder results\...```
