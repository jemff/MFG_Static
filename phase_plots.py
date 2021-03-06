from predator_prey_sys import twod_predator_prey_dyn, dynamics
import matplotlib.pyplot as plt
from infrastructure import *
from scipy.integrate import solve_ivp
def dyn_pp():
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
    res_c = None
    beta_f = None

    out = twod_predator_prey_dyn(Mx = Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap)
    inte = np.ones(tot_points).reshape(1,tot_points)
    #par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
    #       'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0, 'q': 3}

    #, beta_f=lambda x: np.exp(-(par['q'] * x) ** 2) + 0.001,
    #                                     res_conc_f=lambda x: np.exp(-par['q'] * x)
    #par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
    #       'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0, 'q': 3}
    #print(par)
    par = {'res_renew': 1, 'eff': 0.1, 'c_enc_freq': 330/12 * 0.01 ** (0.75), 'p_handle': 12 / 15 * (10) ** (-0.75),
           'p_enc_freq': 330/12 * 10 ** (0.75), 'p_met_loss': 15/12 * 0.05 * (10) ** (0.75), 'competition': 0, 'q': 5}

    beta = np.exp(-(par['q'] * Mx.x) ** 2)  # +0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
    beta = 0.5 * 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

    res_conc = np.exp(-par['q'] * Mx.x)  # np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
    res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001

    print(par)
#    from scipy.special import lambertw
#    D = lambda x: np.real(2 * lambertw(10 / 2 * np.sqrt(np.exp(-20 * x) / (10 ** (4) * (10 ** (1) + np.exp(-20 * x))))) / 10)

#    beta_f = lambda x: D(x)/D(0)
#    beta = beta_f(Mx.x)
#    beta = 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

#    res_c = lambda x: 2/(1+np.exp(5*(x-0.2)))
 #   res_conc = res_c(Mx.x)
 #   res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001

    #beta = 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

    gridx, gridy= np.meshgrid(pop_varc, pop_varp)
    vectors = np.zeros((fidelity, fidelity, 2))
    for i in range(fidelity):
        for j in range(fidelity):
            pop_ij = np.array([pop_varc[i], pop_varp[j]])
            out = twod_predator_prey_dyn(pops=pop_ij, Mx=Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap,
                                         minimal_pops=0, par=par, warmstart_info=out, res_conc_f=res_c, beta_f=beta_f)
            sigma = out['x0'][0: tot_points]
            sigma_p = out['x0'][tot_points: 2*tot_points]
            cons_dyn = par['c_enc_freq']*inte @ (Mx.M @ (sigma * pop_ij[0] * ( 1 - pop_ij[0] * sigma / (res_conc * car_cap)))) \
                       - inte @ ( par['p_enc_freq']* Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (1 + par['p_enc_freq']*par['p_handle'] * inte @ (Mx.M @ (pop_ij[0] * sigma * beta * sigma_p)))
            pred_dyn = par['eff'] * par['p_enc_freq']*inte @ (Mx.M @ (pop_ij[0] * pop_ij[1] * sigma * beta * sigma_p)) / (
                        1 + par['p_enc_freq']*par['p_handle'] * inte @ (Mx.M @ (pop_ij[0] * sigma * beta * sigma_p))) - par[
                           'p_met_loss'] * pop_ij[1] - par['p_enc_freq']*par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta)) * \
                       pop_ij[1]**2

            vectors[i,j,0] = cons_dyn
            vectors[i,j,1] = pred_dyn
            N = np.linalg.norm(vectors[i, j, :])
            vectors[i, j,:] = vectors[i, j,:]/N

    print("Simulating dynamics")

    step_size = 0.1
    num_points = 500
    dyn_data = np.zeros((num_points, 2))
    y0 = 1.5*np.array([0.02668011, 0.00326092]) #This is the fixed point
        #np.array([0.8*pop_max, 0.8*ppop_max])
    dyn_data[0] = y0
    out = twod_predator_prey_dyn(Mx = Mx, pops=dyn_data[0], fixed_point=False, warmstart_out=True, car_cap=car_cap, res_conc_f=res_c, beta_f=beta_f, par=par)

    sigma_c_hist = np.zeros((num_points, segments))
    sigma_p_hist = np.zeros((num_points, segments))
    sigma_c_hist[0] = np.array(out['x0'][0: tot_points]).flatten()
    sigma_p_hist[0] = np.array(out['x0'][tot_points: 2*tot_points]).flatten()

    for i in range(1,num_points):
        out = twod_predator_prey_dyn(pops=dyn_data[i-1], Mx=Mx, fixed_point=False, warmstart_out=True, car_cap=car_cap,
                                     minimal_pops=0, par=par, warmstart_info=out, res_conc_f=res_c, beta_f=beta_f)
        sigma = out['x0'][0: tot_points]
        sigma_p = out['x0'][tot_points: 2 * tot_points]

        sigma_c_hist[i] = np.copy(sigma[::-1])
        sigma_p_hist[i] = np.copy(sigma_p[::-1])

        dyns = dynamics(dyn_data[i-1], par = par,Mx = Mx, inte = inte, car_cap=car_cap, sigma = sigma, sigma_p = sigma_p)
        dyn_data[i] = dyn_data[i-1] + step_size*dyns
        print(dyn_data[i], "Simulated dynamical populations")

    return gridx, gridy, vectors[:,:,0].T, vectors[:,:,1].T, dyn_data, sigma_c_hist, sigma_p_hist


