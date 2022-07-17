import phase_plots as pp_dyn
import phase_plot_static as pp_stat
import matplotlib.pyplot as plt
import numpy as np
from infrastructure import heatmap_plotter

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

vectors = np.zeros((15, 15, 2))
times = np.linspace(0, 30, 300)

gridx, gridy, vectors[:,:,0], vectors[:,:,1], dyn_data_1 = pp_stat.stat_pp()
fig, ax = plt.subplots(1, 2, figsize=(12/2.54, 6/2.54))
q = ax[0].quiver(gridx, gridy, vectors[:,:,0], vectors[:,:,1], scale=50, headwidth=1, color=tableau20[14])
ax[0].plot(dyn_data_1.y[0,:], dyn_data_1.y[1,:], color=tableau20[4])
gridx, gridy, vectors[:,:,0], vectors[:,:,1], dyn_data, hist_c, hist_p = pp_dyn.dyn_pp()
q2 = ax[1].quiver(gridx, gridy, vectors[:,:,0], vectors[:,:,1], scale=50, headwidth=1, color=tableau20[14])
ax[1].plot(dyn_data[:,0], dyn_data[:,1], color=tableau20[4])
ax[1].annotate('t=1', (dyn_data[9,0], dyn_data[10-1,1]))
ax[1].annotate('t=5', (dyn_data[49,0], dyn_data[50-1,1]))
ax[1].annotate('t=10', (dyn_data[99,0], dyn_data[100-1,1]))
ax[1].annotate('t=20', (dyn_data[200-1,0], dyn_data[200-1,1]))
ax[1].annotate('t=50', (dyn_data[500-1,0], dyn_data[500-1,1]))
#ax[1].annotate('t=200', (dyn_data[2000-1,0], dyn_data[2000-1,1]))


#ax[1].annotate('t=0.50', (dyn_data[499,0], dyn_data[499,1]))

ax[0].set_xlabel("Consumer biomass ($N_c$)")
ax[1].set_xlabel("Consumer biomass ($N_c$)")
ax[0].set_ylabel("Predator biomass ($N_p$)")
ax[0].text(1.05, 0.8, 'A', transform=ax[0].transAxes)
ax[1].text(1.05, 0.8, 'B', transform=ax[1].transAxes)

plt.savefig('results/plots/dynamics.pdf')
plt.show()
#
#fig1, ax1 = plt.subplots(2, 1, figsize=(6/2.54, 12/2.54), sharex=True)
#ax1[0].plot(times, dyn_data[:,0], color=tableau20[0])
#ax1[1].plot(times, dyn_data[:,1], color=tableau20[6])
#ax1[1].set_ylabel("Consumer biomass ($N_c$)")
#ax1[0].set_ylabel("Predator biomass ($N_p$)")
#ax1[1].set_xlabel("Time (t)")
#plt.savefig('results/plots/explicit_dyn.pdf')

heatmap_plotter([hist_c.T, hist_p.T], "dynamic_strats",
                [0, 200, 100, 0], xlab="Time in months (t)", ylab="Depth in m")
plt.show()