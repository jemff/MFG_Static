from predator_prey_sys import twod_predator_prey_dyn, dynamics_static
import matplotlib.pyplot as plt
from infrastructure import *
from scipy.integrate import solve_ivp

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 10)
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

layers = 2
segments = 60
length = 1
tot_points = layers*segments
Mx = simple_method(length, tot_points)

pop_max = 1.5
pop_min = 0
car_cap = 40
fidelity = 15
ppop_max = 0.6
pop_varc = np.linspace(0, pop_max, fidelity)
pop_varp = np.linspace(0, ppop_max, fidelity)

min_pop = 0
out = twod_predator_prey_dyn(Mx = Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap)
inte = np.ones(tot_points).reshape(1,tot_points)
par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
       'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0.1, 'q': 3}

beta = np.exp(-(par['q'] * Mx.x) ** 2)  # +0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
beta = 0.5 * 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

res_conc = np.exp(-par['q'] * Mx.x)  # np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001
#, beta_f=lambda x: np.exp(-(par['q'] * x) ** 2) + 0.001,
#                                     res_conc_f=lambda x: np.exp(-par['q'] * x)
gridx, gridy= np.meshgrid(pop_varc, pop_varp)
vectors = np.zeros((fidelity, fidelity, 2))
for i in range(fidelity):
    for j in range(fidelity):
        pop_ij = np.array([pop_varc[i], pop_varp[j]])
        sigma = np.ones(segments*layers)
        sigma_p = np.ones(segments*layers)
        cons_dyn = inte @ (Mx.M @ (sigma * pop_ij[0] / par['c_enc_freq'] * (
                    1 - pop_ij[0] * sigma / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                               Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                               par['p_enc_freq'] + par['p_handle'] * inte @ (
                                   Mx.M @ (pop_ij[0] * sigma * beta * sigma_p))) - par['c_met_loss'] * pop_ij[0]
        pred_dyn = par['eff'] * inte @ (Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                    par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (pop_ij[0] * sigma * beta * sigma_p))) - par[
                       'p_met_loss'] * pop_ij[1] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta)) * \
                   pop_ij[1]

        vectors[i,j,0] = cons_dyn
        vectors[i,j,1] = pred_dyn
        N = np.linalg.norm(vectors[i, j, :])
        vectors[i, j,:] = vectors[i, j,:]/N

fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))
q = ax.quiver(gridx, gridy, vectors[:,:,0].T, vectors[:,:,1].T, scale=50, headwidth=1, color=tableau20[14])
ax.set_xlabel("Biomass, (C)")
ax.set_ylabel("Biomass, (P)")


t_begin = 0
t_end = 20
t_eval = np.linspace(t_begin, t_end, 50)
#dyn_data = solve_ivp(lambda t, y: dynamics_static(t, y, par = par,Mx = Mx, inte = inte, car_cap=car_cap), t_span=[t_begin, t_end], y0 = np.array([2*pop_max/3, 2*ppop_max/3]))
dyn_data_1 = solve_ivp(lambda t, y: dynamics_static(t, y, par = par,Mx = Mx, inte = inte, car_cap=car_cap), t_span=[t_begin, t_end], y0 = np.array([1.5/2, 1/3]), dense_output=True, t_eval = t_eval)
#dyn_data_2 = solve_ivp(lambda t, y: dynamics_static(t, y, par = par,Mx = Mx, inte = inte, car_cap=car_cap), t_span=[t_begin, t_end],  y0 = np.array([pop_max*1/3, ppop_max/4]))

#ax.plot(dyn_data.y[0,:], dyn_data.y[1,:], color=tableau20[0])
ax.plot(dyn_data_1.y[0,:], dyn_data_1.y[1,:], color=tableau20[8])
#ax.plot(dyn_data_2.y[0,:], dyn_data_2.y[1,:], color=tableau20[6])

plt.savefig('results/plots/dynamics_static.pdf')

