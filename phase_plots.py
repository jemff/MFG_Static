from predator_prey_sys import twod_predator_prey_dyn, dynamics
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

layers = 1
segments = 30
length = 1
tot_points = layers*segments
Mx = simple_method(length, tot_points)

pop_max = 10
pop_min = 0
car_cap = 40
fidelity = 15
ppop_max = 10
pop_varc = np.linspace(0, pop_max, fidelity)
pop_varp = np.linspace(0, ppop_max, fidelity)

min_pop = 0
out = twod_predator_prey_dyn(Mx = Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap)
inte = np.ones(tot_points).reshape(1,tot_points)
par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
       'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0, 'q': 3}

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
        out = twod_predator_prey_dyn(pops=pop_ij, Mx=Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap,
                                     minimal_pops=0, par=par, warmstart_info=out)
        sigma = out['x0'][0: tot_points]
        sigma_p = out['x0'][tot_points: 2*tot_points]
        cons_dyn = inte @ (Mx.M @ (sigma * pop_ij[0] * (
                    1 - pop_ij[0] * sigma / (res_conc * car_cap)))) - inte @ (
                               Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                               par['p_enc_freq'] + par['p_handle'] * inte @ (
                                   Mx.M @ (pop_ij[0] * sigma * beta * sigma_p)))
        pred_dyn = par['eff'] * inte @ (Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                    par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (pop_ij[0] * sigma * beta * sigma_p))) - par[
                       'p_met_loss'] * pop_ij[1] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta)) * \
                   pop_ij[1]**2

        vectors[i,j,0] = cons_dyn
        vectors[i,j,1] = pred_dyn
        N = np.linalg.norm(vectors[i, j, :])
        vectors[i, j,:] = vectors[i, j,:]/N

fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))
q = ax.quiver(gridx, gridy, vectors[:,:,0].T, vectors[:,:,1].T, scale=50, headwidth=1, color=tableau20[14])
ax.set_xlabel("Biomass, (C)")
ax.set_ylabel("Biomass, (P)")

step_size = 0.1
num_points = 300
dyn_data = np.zeros((num_points, 2))
y0 = np.array([0.8*pop_max, 0.8*ppop_max])
dyn_data[0] = y0
out = twod_predator_prey_dyn(Mx = Mx, pops=dyn_data[i], fixed_point=False, warmstart_out=True, car_cap=car_cap)

for i in range(1,num_points):
    out = twod_predator_prey_dyn(pops=dyn_data[i-1], Mx=Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap,
                                 minimal_pops=0, par=par, warmstart_info=out)
    sigma = out['x0'][0: tot_points]
    sigma_p = out['x0'][tot_points: 2 * tot_points]
    dyns = dynamics(dyn_data[i-1], par = par,Mx = Mx, inte = inte, car_cap=car_cap, sigma = sigma, sigma_p = sigma_p)
    dyn_data[i] = dyn_data[i-1] + step_size*dyns


ax.plot(dyn_data[:,0], dyn_data[:,1], color=tableau20[0])
plt.savefig('results/plots/dynamics.pdf')

plt.show()

