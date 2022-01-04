from rl_nav.main import RL_nav
import matplotlib.pyplot as plt


sim = RL_nav()
for _ in range(100):
    sim.model_free_learning()
# sim.display_value_func()
sim.display_learning_curve()


print('done')