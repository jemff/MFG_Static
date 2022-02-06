from predator_prey_sys import *
from infrastructure import *
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

layers = 2
segments = 15
length = 5
tot_points = layers*segments
Mx = simple_method(length, tot_points)
inte = np.ones(tot_points).reshape(1,tot_points)
par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
       'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0.1, 'q': 3}


def dynamics(t, y, par = None, car_cap = 2, Mx = None, inte = None):
    beta = np.exp(-Mx.x ** 2)
    res_conc = np.exp(-Mx.x) + 0.001
    print(t, y)
    out = twod_predator_prey_dyn(pops=y, Mx=Mx, fixed_point=True, warmstart_out=True, car_cap=car_cap,
                             minimal_pops=0, par=par)
    sigma = out['x0'][0: tot_points]
    sigma_p = out['x0'][tot_points: 2 * tot_points]
    cons_dyn = inte @ (Mx.M @ (sigma * y[0] / par['c_enc_freq'] * (
            1 - y[0] * sigma / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                       Mx.M @ (y[0] * y[1] * sigma * beta * sigma_p)) / (
                       par['p_enc_freq'] + par['p_handle'] * inte @ (
                       Mx.M @ (y[0] * sigma * beta * sigma_p))) - par['c_met_loss'] * y[0]
    pred_dyn = par['eff'] * inte @ (Mx.M @ (y[0] * y[1] * sigma * beta * sigma_p)) / (
            par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (y[0] * sigma * beta * sigma_p))) - par[
                   'p_met_loss'] * y[1]
    return np.array([cons_dyn[0], pred_dyn[0]])


t_begin = 0
t_end = 30

dyn_data = solve_ivp(lambda t, y: dynamics(t, y, par = par,Mx = Mx, inte = inte), t_span = [t_begin, t_end], y0 = np.array([0.5, 0.5]), method='RK23')

#plt.plot(dyn_data.t, )
plt.plot(dyn_data.y[0,:], dyn_data.y[1,:])

plt.show()