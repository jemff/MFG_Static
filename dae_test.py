import casadi as ca
import numpy as np
from infrastructure import *
class discrete_patches:
    def __init__(self, depth, total_points):
        self.x = np.linspace(0, depth, total_points)

        self.M = depth/total_points * np.identity(total_points)

Mx = simple_method(5, 50) #spectral_method(4, 20) #spectral_method(5,50) # discrete_patches(1,20)#

tot_points = Mx.x.size
inte = np.ones(tot_points).reshape(1,tot_points)
s = ca.MX.sym('s', 2)
#state = np.ones(2)
car_cap = 5
#s = state
par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01, 'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0, 'q': 3}

res_conc = np.exp(-par['q']*Mx.x) #np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
res_conc = 1/(inte @ (Mx.M @ res_conc))*res_conc

beta = np.exp(-(par['q']*Mx.x)**2) #+0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
beta = 0.5*1 / (inte @ (Mx.M @ beta)) * beta + 10**(-4)

lam = ca.MX.sym('lam', 2)
sigma = ca.MX.sym('sigma', Mx.x.shape[0])
sigma_p = ca.MX.sym('sigma_p', Mx.x.shape[0])

k = 0.01

gridx, gridy = np.meshgrid(Mx.x, Mx.x)
t = -1
ker = lambda x, y: np.exp((x - y) ** 2 / (4 * k * t)) + np.exp((-y - x) ** 2 / (4 * k * t)) + np.exp(
    (2 * Mx.x[-1] - x - y) ** 2 / (4 * k * t))
out = ker(gridx, gridy)
for j in range(tot_points):
    out[j,:] = out[j,:]/(inte @ Mx.M @ out[j,:])

out_inv = np.linalg.inv(np.transpose(out))


plt.plot(Mx.x, out[10,:])
plt.show()
#for j in range(tot_points):
#    sigma += sigma_w[j]*out[j,:]
#    sigma_p += sigma_p_w[j]*out[j,:]

cons_dyn = inte @ (Mx.M @ (sigma / par['c_enc_freq'] * (
            1 - s[0] * sigma ** 2 / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                       Mx.M @ (s[1] * sigma * beta * sigma_p)) / (
                       par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (s[0] * sigma * beta * sigma_p)))
pred_dyn = par['eff'] * inte @ (Mx.M @ (s[0] * sigma * beta * sigma_p)) / (
            par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (s[0] * sigma * beta * sigma_p))) - par[
               'p_met_loss'] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta))

df1 = 1 / par['c_enc_freq'] * (1 - s[0] * sigma / (res_conc * par['c_enc_freq'] * car_cap)) - s[
    1] * sigma_p * beta / (
                  par['p_enc_freq'] + inte @ (Mx.M @ (par['p_handle'] * s[0] * sigma * beta * sigma_p))) + lam[
          0]
df2 = par['eff'] * s[0] * par['p_enc_freq'] * sigma * beta / (
            inte @ (Mx.M @ (par['p_handle'] * s[0] * sigma * beta * sigma_p)) + par['p_enc_freq']) ** 2 + lam[
          1] - par['competition'] * sigma_p * beta
sdot = ca.vertcat(s[0]*cons_dyn, s[1]*pred_dyn)

g1 = inte @ (out_inv @ sigma) - 1
g2 = inte @ (out_inv @ sigma_p) - 1
g3 = ca.vertcat(-df1, -df2)

ipg1 = inte @ (out_inv @ sigma) - 1
ipg2 = inte @ (out_inv @ sigma_p) - 1
ipg3 = inte @ Mx.M @ (df1**2 + df2**2)


ipg = ca.vertcat(*[ipg1, ipg2,ipg3, cons_dyn, pred_dyn])
alg_eq = ca.vertcat(*[g1, g2, g3])
z = ca.vertcat(*[sigma, sigma_p, lam])


dae = {'x':s, 'ode': sdot, 'z': z, 'alg': alg_eq}
opts = {'tf':0.01} # interval length
I = ca.integrator('I', 'idas', dae, opts)

s_opts = {'ipopt': {'print_level': 5, 'linear_solver': 'ma57'}}

prob = {'x': ca.vertcat(z, s), 'f': ipg3, 'g': ipg}
lbx = ca.vertcat(*[np.zeros(tot_points*2), 2*[-ca.inf], np.zeros(2)])
solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

lbg = np.zeros(ipg.size()[0])
ubg = np.zeros(ipg.size()[0])
init = np.ones(ca.vertcat(z).size()[0] + 2)/np.max(Mx.x)

sol = solver(lbx =lbx, lbg=lbg, ubg=ubg, x0=init)
t_x = np.array(sol['x']).flatten()

#print(np.dot(inte, ((t_x[2:tot_points+2])**2)))
plt.plot(Mx.x, (t_x[0:tot_points]))
plt.show()
x0 = np.array(sol['x']).flatten()[-2:0]
z0 = np.array(sol['x']).flatten()[0:-2]


#np.ones(z.size()[0]) / np.max(Mx.x) #t_x[2:] #np.ones(z.size()[0]) / np.max(Mx.x) #np.array(sol['x']).flatten()
#I.setOption("linear_solver", "csparse")
r = I(x0=x0, z0=z0)
#print(r['zf'])
#print(r['xf'])



