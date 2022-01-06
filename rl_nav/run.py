from rl_nav.main import RL_nav

# sim = RL_nav('obstacle')
sim = RL_nav('open field')
for i in range(20):
    print(i)
    sim.model_free_learning(initialization='zero',policy='epsilon-greedy')
# sim.display_value_func()
sim.display_learning_curve()

print('done')
