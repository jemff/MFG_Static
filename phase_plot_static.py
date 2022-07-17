from predator_prey_sys import twod_predator_prey_dyn, dynamics_static
import matplotlib.pyplot as plt
from infrastructure import *
from scipy.integrate import solve_ivp

def stat_pp():
    layers = 1
    segments = 60
    length = 1
    tot_points = layers*segments
    Mx = simple_method(length, tot_points)

    pop_max = 0.6
    car_cap = 2
    fidelity = 15
    ppop_max = 0.1
    pop_varc = np.linspace(0, pop_max, fidelity)
    pop_varp = np.linspace(0, ppop_max, fidelity)

    inte = np.ones(tot_points).reshape(1,tot_points)
    #par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
    #       'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0, 'q': 3}
    par = {'res_renew': 1, 'eff': 0.1, 'c_enc_freq': 330/12 * 0.01 ** (0.75), 'p_handle': 12 / 15 * (10) ** (-0.75),
           'p_enc_freq': 330/12 * 10 ** (0.75), 'p_met_loss': 15/12 * 0.05 * (10) ** (0.75), 'competition': 0, 'q': 5}


#    par = {'res_renew': 1, 'eff': 0.1, 'c_enc_freq': 300 * 0.01 ** (0.75),
##           'p_handle': 1 / 10 * (100) ** (-0.75), 'p_enc_freq': 100 * 100 ** (0.75),
#          'p_met_loss': 10 * 0.05 * (100) ** (0.75), 'competition': 0}

    #from scipy.special import lambertw

    #D = lambda x: np.real(2 * lambertw(10 / 2 * np.sqrt(np.exp(-10 * x) / (10 ** (4) * (10 ** (1) + np.exp(-10 * x))))) / 10)
    #beta_f = lambda x: 2/(1+np.exp(5*(x))) #np.exp(-x**2/0.01)
    #beta_f = lambda x: D(x)/D(0)
    #beta_f = None
    #res_conc_f = lambda x: 2/(1+np.exp(5*(x-0.2)))

    #res_conc = res_conc_f(Mx.x)
    #res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001

    #beta = beta_f(Mx.x)
    #beta = 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

    t_begin = 0
    t_end = 120
    t_eval = np.linspace(t_begin, t_end, 500)
    car_cap = 2


    beta = np.exp(-(par['q'] * Mx.x) ** 2)  # +0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
    beta = 0.5 * 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

    res_conc = np.exp(-par['q'] * Mx.x)  # np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
    res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001

    gridx, gridy= np.meshgrid(pop_varc, pop_varp)
    vectors = np.zeros((fidelity, fidelity, 2))
    for i in range(fidelity):
        for j in range(fidelity):
            pop_ij = np.array([pop_varc[i], pop_varp[j]])
            sigma = np.ones(segments*layers)
            sigma_p = np.ones(segments*layers)
            cons_dyn = par['c_enc_freq']*inte @ (Mx.M @ (sigma * pop_ij[0] * (
                        1 - pop_ij[0] * sigma / (res_conc * car_cap)))) - inte @ ( par['p_enc_freq']*
                                   Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                                   1 + par['p_enc_freq']*par['p_handle'] * inte @ (
                                       Mx.M @ (pop_ij[0] * sigma * beta * sigma_p)))
            pred_dyn = par['eff'] * par['p_enc_freq']*inte @ (Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                        1 + par['p_enc_freq']*par['p_handle'] * inte @ (Mx.M @ (pop_ij[0] * sigma * beta * sigma_p))) - par[
                           'p_met_loss'] * pop_ij[1] - par['p_enc_freq']*par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta)) * \
                       pop_ij[1]**2

            vectors[i,j,0] = cons_dyn
            vectors[i,j,1] = pred_dyn
            N = np.linalg.norm(vectors[i, j, :])
            vectors[i, j,:] = vectors[i, j,:]/N



    dyn_data_1 = solve_ivp(lambda t, y: dynamics_static(t, y, par = par,Mx = Mx, inte = inte, car_cap=car_cap), t_span=[t_begin, t_end], y0=np.array([0.025, 0.05]), dense_output=True, t_eval = t_eval)

    return gridx, gridy, vectors[:,:,0].T, vectors[:,:,1].T, dyn_data_1
#ax.plot(dyn_data_1.y[0,:], dyn_data_1.y[1,:], color=tableau20[8])

#plt.savefig('results/plots/dynamics_static.pdf')


