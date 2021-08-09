from predator_prey_sys import *
from infrastructure import *
import matplotlib.pyplot as plt
methods = {'spectral': True, 'simple': False, 'discrete': False}

layers = 2
segments = 30
length = 5
tot_points = layers*segments
Mx = spectral_method(length, segments=segments, layers=layers) #spectral_method(length, segments=segments, layers=layers) #simple_method(length, tot_points) #discrete_patches(length, total_points=tot_points)

steps = 10
car_cap = np.linspace(1, 4, steps)
populations = np.zeros((steps,2))
min_pop = 0
out = threed_predator_prey_dyn(res_conc_f = lambda x: 1/(1+x)*1/np.log(length+1), beta_f=lambda x: 1/(1+x)+0.001,
                               Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=0.1, minimal_pops=min_pop)

for i, inv_car_cap in enumerate(car_cap):
    out = threed_predator_prey_dyn(res_conc_f = lambda x: 1/(1+x)*1/np.log(length+1), beta_f=lambda x: 1/(1+x)+0.001,
                                   Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=car_cap[i],
                                   minimal_pops=min_pop, warmstart_info=out)
    populations[i] = out['x0'][-4:-2]-min_pop

plt.plot(car_cap, populations[:,0])
plt.plot(car_cap, populations[:,1])
plt.show()

#simple_method(length, tot_points)

steps = 8
lengths = np.linspace(5, 8, steps)
#outputs = []
populations = np.zeros((steps,2))
min_pop = 0
Mx = spectral_method(lengths[0], segments=segments, layers=layers)#simple_method(length, tot_points)
out = threed_predator_prey_dyn(res_conc_f = lambda x: 1/(1+x**2)*1/np.log(length+1), beta_f=lambda x: 1/(1+x)+0.001,
                               Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=2, minimal_pops=min_pop)
populations[0] = out['x0'][-4:-2] - min_pop

for i in range(1, steps):
    Mx = spectral_method(lengths[i], segments=segments, layers=layers) #(lengths[i], total_points=tot_points)  # simple_method(length, tot_points)

    out = threed_predator_prey_dyn(res_conc_f=lambda x: 1 / (1 + x ** 2) * 1 / np.log(lengths[i] + 1),
                                   beta_f=lambda x: 1 / (1 + x)+0.001, Mx=Mx, fixed_point=True, warmstart_out=True, car_cap=2, minimal_pops=min_pop, warmstart_info=out)
    populations[i] = out['x0'][-4:-2]-min_pop

plt.plot(lengths, populations[:,0])
plt.plot(lengths, populations[:,1])
plt.show()