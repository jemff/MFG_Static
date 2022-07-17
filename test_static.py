from predator_prey_sys import twod_predator_prey_dyn, dynamics_static
import matplotlib.pyplot as plt
from infrastructure import *
from scipy.integrate import solve_ivp

layers = 1
segments = 60
length = 1
tot_points = layers * segments
Mx = simple_method(length, tot_points)
inte = np.ones(tot_points).reshape(1, tot_points)

t_begin = 0
t_end = 120
t_eval = np.linspace(t_begin, t_end, 500)
car_cap = 2


par = {'res_renew': 1, 'eff': 0.1, 'c_enc_freq': 330/12 * 0.01 ** (0.75), 'p_handle': 12 / 15 * (10) ** (-0.75),
       'p_enc_freq': 330/12 * 10 ** (0.75), 'p_met_loss': 15/12 * 0.05 * (10) ** (0.75), 'competition': 0, 'q': 5}
print(par)
#dyn_data_1 = solve_ivp(lambda t, y: dynamics_static(t, y, par=par, Mx=Mx, inte=inte, car_cap=car_cap),
#                       t_span=[t_begin, t_end], y0=np.array([0.025, 0.05]), dense_output=True, t_eval=t_eval)
dyn_data_1 = solve_ivp(lambda t, y: dynamics_static(t, y, par=par, Mx=Mx, inte=inte, car_cap=car_cap),
                       t_span=[t_begin, t_end], y0=np.array([0.025, 0.05]), dense_output=True, t_eval=t_eval)

print(dyn_data_1)

plt.plot(dyn_data_1.y[0,:], dyn_data_1.y[1,:])
plt.show()
print(par)