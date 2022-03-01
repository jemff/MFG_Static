import numpy as np
import casadi as ca

def twod_predator_prey_dyn(beta_f = None,
                           res_conc_f = None, minimal_pops = 10**(-5), fixed_point = False,
                           pops = np.array([1,1]), Mx = None, warmstart_info = None,
                           warmstart_out = False, par = None, car_cap = 1, calc_funcs = False, x_in = None):

    tot_points = Mx.x.size
    inte = np.ones(tot_points).reshape(1,tot_points)

    if par is None:
        par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01, 'p_enc_freq': np.sqrt(0.1), 'p_met_loss': 0.15, 'competition': 0.1, 'q': 3}

    if res_conc_f is None:
        res_conc = np.exp(-par['q']*Mx.x) #np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
        res_conc = 1/(inte @ (Mx.M @ res_conc))*res_conc + 0.0001
    else:
        res_conc = res_conc_f(Mx.x)

    if beta_f is None:
        beta = np.exp(-(par['q']*Mx.x)**2) #+0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
        beta = 0.5*1 / (inte @ (Mx.M @ beta)) * beta +0.0001
    else:
        beta = beta_f(Mx.x)


    lam = ca.MX.sym('lam', 2)

    sigma = ca.MX.sym('sigma', Mx.x.shape[0])
    sigma_p = ca.MX.sym('sigma_p', Mx.x.shape[0])
#    mu1 = ca.MX.sym('mu1', Mx.x.shape[0])
#    mu2 = ca.MX.sym('mu2', Mx.x.shape[0])


    if fixed_point is True:
        state = ca.MX.sym('state', 2)
        state_ss = state+minimal_pops
        vars = 5
    else:
        state_ss = pops
        state = state_ss
        vars = 3



    cons_dyn = inte @ (Mx.M @ (sigma*(1-state_ss[0]*sigma/(res_conc*car_cap)))) - inte @ (Mx.M @ (state_ss[1]*sigma*beta*sigma_p))/(1+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p)))
    pred_dyn = par['eff']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))/(1+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - par['p_met_loss'] - (state_ss[1]**2)* par['competition']*inte @ (Mx.M @ ((sigma_p**2)*beta))

    df1 =(1-state_ss[0]*sigma/(res_conc*car_cap)) - state_ss[1]*sigma_p*beta/(1+inte @ (Mx.M @ (par['p_handle']*state_ss[0]*sigma*beta*sigma_p))) - lam[0]*np.ones(tot_points)
    df2 = par['eff']*state_ss[0]*sigma*beta/(1+inte @ (Mx.M @ (par['p_handle']* state_ss[0]*sigma*beta*sigma_p)))**2 - lam[1]*np.ones(tot_points) - state[1]*par['competition']*sigma_p*beta


    g0 = ca.vertcat(cons_dyn, pred_dyn)
    g1 = inte @ Mx.M @ (df1*sigma) + inte @ Mx.M @ (df2*sigma_p)  #
    g2 = inte @ Mx.M @ sigma_p - 1
    g3 = inte @ Mx.M @ sigma - 1
    g4 = ca.vertcat(-df1, -df2)

    if fixed_point is True:
        g = ca.vertcat(g0, g1, g2, g3, g4) #g1,

    if fixed_point is False:
        g = ca.vertcat(g1, g2, g3, g4) #


    f = 0

    sigmas = ca.vertcat(sigma, sigma_p) #sigma_bar
    if fixed_point is True:
        x = ca.vertcat(*[sigmas, state, lam])
    else:
        x = ca.vertcat(*[sigmas, lam])

    mcp_function_ca = ca.Function('fun', [x], [ca.vertcat(df1, df2)])
    if calc_funcs is True:
        return mcp_function_ca(x_in)

    lbg = np.zeros(vars + 2*tot_points)
    ubg = ca.vertcat(*[np.zeros(vars), [ca.inf]*2*tot_points])
    if warmstart_info is None:
        s_opts = {'ipopt': {'print_level' : 3, 'linear_solver':'ma57',  'acceptable_iter': 5} }
        init = np.ones(x.size()[0]) / np.max(Mx.x)
        init[-4:] = 1

    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57',
                            'acceptable_iter': 5,'hessian_approximation':'limited-memory',
                            'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-9,
                            'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_frac': 1e-9,
                            'warm_start_slack_bound_push': 1e-9, 'warm_start_mult_bound_push': 1e-9}} #

    prob = {'x': x, 'f': f, 'g': g}
    lbx = ca.vertcat(*[np.zeros(x.size()[0] - 2), -ca.inf, -ca.inf])
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

    if warmstart_info is None:
        sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)

    else:
        sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = warmstart_info['x0'], lam_g0 = warmstart_info['lam_g0'], lam_x0 = warmstart_info['lam_x0'] )
    if warmstart_out is False:
        return np.array(sol['x']).flatten()

    else:
        ret_dict = {'x0': np.array(sol['x']).flatten(), 'lam_g0': np.array(sol['lam_g']).flatten(), 'lam_x0': np.array(sol['lam_x']).flatten()}
        return ret_dict


def threed_predator_prey_dyn(resources = None, beta_f = None, res_conc_f = None, minimal_pops = 10**(-5), fixed_point = False, pops = np.array([1,1]), Mx = None, warmstart_info = None, warmstart_out = False, par = None, car_cap = 1):

    tot_points = Mx.x.size

    if res_conc_f is None:
        res_conc = np.exp(-Mx.x)
    else:
        res_conc = res_conc_f(Mx.x)

    if beta_f is None:
        beta = np.exp(-Mx.x**2)+0.001
    else:
        beta = beta_f(Mx.x)

    lam = ca.MX.sym('lam', 2)

    sigma = ca.MX.sym('sigma', tot_points)
    sigma_p = ca.MX.sym('sigma_p', tot_points)

    if fixed_point is True:
        state = ca.MX.sym('state', 2)
        res_level = ca.MX.sym('res_level', tot_points)
        state_ss = state+minimal_pops
        vars = 5
    else:
        res_level = resources
        state_ss = pops
        vars = 3

    inte = np.ones(tot_points).reshape(1,tot_points)
    if par is None:
        par = {'res_renew': 1, 'eff': 0.3, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01, 'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0.1}

    #res_dyn = (par['res_renew']*(res_conc*car_cap - res_level) - state_ss[0]*inte @ (Mx.M @ (res_level*sigma))/(par['c_handle']*inte @ (Mx.M @ (res_level*sigma)) + par['c_enc_freq'])).T @ (par['res_renew']*(res_conc*car_cap - res_level) - state_ss[0]*inte @ (Mx.M @ (res_level*sigma))/(par['c_handle']*inte @ (Mx.M @ (res_level*sigma)) + par['c_enc_freq']))
    res_dyn_one = par['res_renew']*(res_conc*car_cap - res_level) - state_ss[0]*res_level*sigma/(par['c_handle']*inte @ (Mx.M @ (res_level*sigma)) + par['c_enc_freq'])
    #res_dyn_one = par['res_renew']*(res_conc*car_cap - res_level) - state_ss[0]*res_level*sigma/(par['c_handle']*res_level*sigma + par['c_enc_freq'])

    D_2 = (Mx.D @ Mx.D)
    D_2[0] = np.copy(Mx.D[0])
    D_2[-1] = np.copy(Mx.D[-1])
    almost_ID = np.identity(tot_points)
    almost_ID[0,0] = 0
    almost_ID[-1,-1] = 0

    #res_dyn_two = 0.05*D_2 @ res_level + almost_ID @ res_dyn_one
    #res_dyn = inte @ (Mx.M @ (res_dyn_one*res_dyn_one))/(Mx.x[-1]**2) #res_dyn_one.T @ res_dyn_one #(inte @ (Mx.M @ (res_dyn_one*res_dyn_one))) #res_dyn_one.T @ res_dyn_one/tot_points**2
    #res_dyn = res_dyn_one.T @ res_dyn_one
    pde_form = D_2 @ res_level + almost_ID @ res_dyn_one
    res_dyn = inte @ (Mx.M @ (pde_form * pde_form)) / (Mx.x[-1] ** 2)
    #res_dyn = pde_form.T @ pde_form + res_dyn_one[0]**2 + res_dyn_one[-1]**2

    #(ca.norm_2(par['res_renew']*(res_conc*car_cap - res_level) - state_ss[0]*inte @ (Mx.M @ (res_level*sigma))/(par['c_handle']*inte @ (Mx.M @ (res_level*sigma)) + par['c_enc_freq'])))
    cons_dyn = par['eff']*state_ss[0]*inte @ (Mx.M @ (sigma*res_level))/(par['c_handle']*inte @ (Mx.M @ (sigma*res_level)) + par['c_enc_freq']) - inte @ (Mx.M @ (state_ss[0]*state_ss[1]*sigma*beta*sigma_p))/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - par['c_met_loss']*state_ss[0]
    pred_dyn = par['eff']*inte @ (Mx.M @ (state_ss[0]*state_ss[1]*sigma*beta*sigma_p))/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - par['competition']*inte @ (Mx.M @ (sigma_p**2*beta)) - par['p_met_loss']*state_ss[1]

    #cons_dyn = par['eff']*state_ss[0]*inte @ (Mx.M @ (sigma*res_level/(par['c_handle']*sigma*res_level)) + par['c_enc_freq']) - inte @ (Mx.M @ ((state_ss[0]*state_ss[1]*sigma*beta*sigma_p)/(par['p_enc_freq']+par['p_handle']*state_ss[0]*sigma*beta*sigma_p))) - par['c_met_loss']*state_ss[0]
    #pred_dyn = par['eff']*inte @ (Mx.M @ ((state_ss[0]*state_ss[1]*sigma*beta*sigma_p)/(par['p_enc_freq']+par['p_handle']*state_ss[0]*sigma*beta*sigma_p))) - par['competition']*inte @ (Mx.M @ (sigma_p**2*beta)) - par['p_met_loss']*state_ss[1]

    df1 = res_level * par['c_enc_freq']/(par['c_handle']*inte @ (Mx.M @ (sigma*res_level) + par['c_enc_freq']))**2-1/par['eff']*state_ss[1]*sigma_p*beta/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - lam[0]*np.ones(tot_points)
    df2 = state_ss[0]*par['p_enc_freq']*sigma*beta/(par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))+par['p_enc_freq'])**2 - 1/par['eff']*par['competition']*sigma_p*beta - lam[1]*np.ones(tot_points)
    #df1 = res_level * par['c_enc_freq']/(par['c_handle']*sigma*res_level + par['c_enc_freq'])**2-1/par['eff']*state_ss[1]*sigma_p*beta/(par['p_enc_freq']+par['p_handle']*state_ss[0]*sigma*beta*sigma_p) - lam[0]*np.ones(tot_points)
    #df2 = state_ss[0]*par['p_enc_freq']*sigma*beta/(par['p _handle']*state_ss[0]*sigma*beta*sigma_p+par['p_enc_freq'])**2 - 1/par['eff']*par['competition']*sigma_p*beta - lam[1]*np.ones(tot_points)

    #g0 = ca.vertcat(cons_dyn, pred_dyn)
    g0 = ca.vertcat(cons_dyn, pred_dyn)
    g1 = inte @ Mx.M @ (df1*sigma) + inte @ Mx.M @ (df2*sigma_p)  #
    g2 = inte @ Mx.M @ sigma_p - 1
    g3 = inte @ Mx.M @ sigma - 1
    g4 = ca.vertcat(-df1, -df2)

    g = ca.vertcat(g0, g1, g2, g3, g4)

    #print(g0.size())
    f = res_dyn #ca.sin(res_dyn)**2 #ca.cosh(res_dyn)-1 #ca.cosh(res_dyn)-1 # ca.exp(res_dyn)-1 #ca.sqrt(res_dyn) #ca.cosh(res_dyn) - 1 #- 1 cosh most stable, then exp, then identity

    sigmas = ca.vertcat(sigma, sigma_p) #sigma_bar
    if fixed_point is True:
        x = ca.vertcat(*[sigmas, res_level, state, lam])
    else:
        x = ca.vertcat(*[sigmas, lam])

    lbg = np.zeros(vars + 2*tot_points)
    ubg = ca.vertcat(*[np.zeros(vars), [ca.inf]*2*tot_points])
    if warmstart_info is None:
        s_opts = {'ipopt': {'print_level' : 3, 'linear_solver':'ma57',  'acceptable_iter': 15}} #'hessian_approximation':'limited-memory',
        init = np.ones(x.size()[0]) / np.max(Mx.x)
        init[-4:] = 1

    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57',
                            'acceptable_iter': 15, 'hessian_approximation':'limited-memory',
                            'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-9,
                            'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_frac': 1e-9,
                            'warm_start_slack_bound_push': 1e-9, 'warm_start_mult_bound_push': 1e-9}}

    prob = {'x': x, 'f': f, 'g': g}
    lbx = ca.vertcat(*[np.zeros(x.size()[0] - 2), -ca.inf, -ca.inf])
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

    if warmstart_info is None:
        sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)

    else:
        sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = warmstart_info['x0'], lam_g0 = warmstart_info['lam_g0'], lam_x0 = warmstart_info['lam_x0'] )


    if warmstart_out is False:
        return np.array(sol['x']).flatten()

    else:
        ret_dict = {'x0': np.array(sol['x']).flatten(), 'lam_g0': np.array(sol['lam_g']).flatten(), 'lam_x0': np.array(sol['lam_x']).flatten()}
        return ret_dict

def dynamics(y, par = None, car_cap = None, Mx = None, inte = None, sigma_p = None, sigma=None): #Reimplement this as DAE in CasADi...
    beta = np.exp(-(par['q'] * Mx.x) ** 2)  # +0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
    beta = 0.5 * 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

    res_conc = np.exp(-par['q'] * Mx.x)  # np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
    res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001

    cons_dyn = inte @ (Mx.M @ (sigma * y[0] / par['c_enc_freq'] * (
            1 - y[0] * sigma / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                       Mx.M @ (y[0] * y[1] * sigma * beta * sigma_p)) / (
                       par['p_enc_freq'] + par['p_handle'] * inte @ (
                       Mx.M @ (y[0] * sigma * beta * sigma_p))) - par['c_met_loss'] * y[0]
    pred_dyn = par['eff'] * inte @ (Mx.M @ (y[0] * y[1] * sigma * beta * sigma_p)) / (
            par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (y[0] * sigma * beta * sigma_p))) - par[
                   'p_met_loss'] * y[1] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta)) * \
               y[1]

    return np.array([cons_dyn[0], pred_dyn[0]])

def dynamics_static(t, y, par = None, car_cap = 2, Mx = None, inte = None): #Reimplement this as DAE in CasADi...
    beta = np.exp(-(par['q'] * Mx.x) ** 2)  # +0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
    beta = 0.5 * 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001

    res_conc = np.exp(-par['q'] * Mx.x)  # np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
    res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001

    tot_points = Mx.x.size

    sigma = np.ones(tot_points)
    sigma_p = np.ones(tot_points)
    cons_dyn = inte @ (Mx.M @ (sigma * y[0] / par['c_enc_freq'] * (
            1 - y[0] * sigma / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                       Mx.M @ (y[0] * y[1] * sigma * beta * sigma_p)) / (
                       par['p_enc_freq'] + par['p_handle'] * inte @ (
                       Mx.M @ (y[0] * sigma * beta * sigma_p))) - par['c_met_loss'] * y[0]
    pred_dyn = par['eff'] * inte @ (Mx.M @ (y[0] * y[1] * sigma * beta * sigma_p)) / (
            par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (y[0] * sigma * beta * sigma_p))) - par[
                   'p_met_loss'] * y[1] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta)) * \
               y[1]**2
    return np.array([cons_dyn[0], pred_dyn[0]])