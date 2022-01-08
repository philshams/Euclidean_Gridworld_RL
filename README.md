# Euclidean_Gridworld_RL

## Overview
This is a lightweight repo for training reinforcement-learning agents is a gridworld, such that agents optimize for the shortest integrated *Euclidean* distance
to rewards rather than the shortest grid ('Manhattan') distance. This is done by adding a dwell time *D* for each action that is proportional to the Euclidean distance
between the current and upcoming state. Strictly speaking, adding dwell time makes this a *Semi-Markov Decision Process*. This is one way to allow agents to favor 
effient paths using diagonal trajectories (see example below). Model-free learning is currently supported.

## Usage
- Install the repo from the directory *...\Euclidean_Gridworld_RL* using ```pip install -e .```
- Modify the config file as needed, to adjust the environment, number of trials, or hyperparameters
- To train an agent and generate a learning curve, in *...\Euclidean_Gridworld_RL\rl_nav* run a command like ```python learning_curve.py```
- To examine an example episode of the agent, starting in the test area and ending when the agent reaches a reward: ```python demo.py```


## Output
### demo.py
The demo script will output a series of images of the agent's state (red circles), starting in the test location (white circle) and ending with reward (green circle). 
This episode will automatically terminate after number-of-states steps. The background is color-coded to represens the agent's current (frozen) value function.
<p float="left">
<img src="https://github.com/philshams/philshams/blob/main/test_0.gif" width="200"/>
<img src="https://github.com/philshams/philshams/blob/main/test_300.gif" width="200"/>
<img src="https://github.com/philshams/philshams/blob/main/test_500.gif" width="200"/>
<img src="https://github.com/philshams/philshams/blob/main/test_1000.gif" width="200"/>
</p>

### learning_curve.py
<img align="left" width="400" src="https://github.com/philshams/philshams/blob/main/learning%20curve.png">
<br/><br/><br/>
The learning_curve script will output a plot showing the progression of learning over a desired number of timesteps. Performance is defined by the 
integrated Euclidean distance between the user-defined starting location and a reward. The curve shows the mean of multiple experiments, and one standard error.

