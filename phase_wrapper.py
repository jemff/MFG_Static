import phase_plots as pp_dyn
import phase_plot_static as pp_stat
import matplotlib.pyplot as plt


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


gridx, gridy, vectors[:,:,0].T, vectors[:,:,1].T, dyn_data_1 = pp_stat.stat_pp()
fig, ax = plt.subplots(2, figsize=(6/2.54, 6/2.54))
q = ax[0].quiver(gridx, gridy, vectors[:,:,0].T, vectors[:,:,1].T, scale=50, headwidth=1, color=tableau20[14])
ax.set_xlabel("Biomass, (C)")
ax.set_ylabel("Biomass, (P)")
