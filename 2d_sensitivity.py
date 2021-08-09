from predator_prey_sys import *
from infrastructure import *
import matplotlib.pyplot as plt
methods = {'spectral': False, 'simple': False, 'discrete': False}

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



def giant_simulator(layers = 2, segments = 30, length = 1, car_cap = 3, steps = 20, par = None, method=simple_method, file_append = '_c'):
    tot_points = layers*segments

    Mx = method(length, tot_points)
    if par is None:
        par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01, 'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0.1, 'q': 3}

    car_caps = np.linspace(1, 7, steps)
    populations_car = np.zeros((steps,2))
    min_pop = 0
    out = twod_predator_prey_dyn(Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=car_caps[0], minimal_pops=min_pop, par = par)
    strategies_car = np.zeros((steps, tot_points, 2))
    for i in range(steps):
        out = twod_predator_prey_dyn(Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=car_caps[i],
                                       minimal_pops=min_pop, warmstart_info=out, par = par)
        populations_car[i] = out['x0'][-4:-2]-min_pop
        strategies_car[i, :, 0] = out['x0'][0:tot_points][::-1]
        strategies_car[i, :, 1] = out['x0'][tot_points:2*tot_points][::-1]

    qs = np.linspace(1, 6, steps)
    populations_ref = np.zeros((steps,2))
    min_pop = 0
    out = twod_predator_prey_dyn(Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=car_cap, minimal_pops=min_pop, par = par)
    populations_ref[0] = out['x0'][-4:-2] - min_pop
    strategies_ref = np.zeros((steps, tot_points, 2))
    for i in range(0, steps):
        par['q'] = qs[i]
        out = twod_predator_prey_dyn(Mx=Mx, fixed_point=True, warmstart_out=True, car_cap=car_cap, minimal_pops=min_pop, warmstart_info=out, par = par)
        populations_ref[i] = out['x0'][-4:-2]-min_pop
        strategies_ref[i, :, 0] = out['x0'][0:tot_points][::-1]
        strategies_ref[i, :, 1] = out['x0'][tot_points:2*tot_points][::-1]
    par['q'] = 5
    print(par)
    populations_comp = np.zeros((steps,2))
    competition = np.linspace(0, 1, steps)

    out = twod_predator_prey_dyn(Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=car_cap, minimal_pops=min_pop, par = par)
    strategies_comp = np.zeros((steps, tot_points, 2))

    for i in range(steps):
        par['competition'] = competition[i]
        out = twod_predator_prey_dyn(Mx = Mx, fixed_point=True, warmstart_out=True, car_cap=car_cap,
                                       minimal_pops=min_pop, warmstart_info=out, par = par)
        populations_comp[i] = out['x0'][-4:-2]-min_pop
        strategies_comp[i, :, 0] = out['x0'][0:tot_points][::-1]
        strategies_comp[i, :, 1] = out['x0'][tot_points:2*tot_points][::-1]


    plt.plot(competition, populations_comp[:,0])
    plt.plot(competition, populations_comp[:,1])

    plt.plot(qs, populations_ref[:,0])
    plt.plot(qs, populations_ref[:,1])

    heatmap_plotter([strategies_ref[:,:,0].T, strategies_ref[:,:,1].T], "increasing_refuge_quality"+file_append, [qs[0], qs[-1], 1, 0], xlab = "Refuge quality ($q$)", ylab = "Refuge")
    heatmap_plotter([strategies_car[:,:,0].T, strategies_car[:,:,1].T], "increasing_car_cap"+file_append, [car_caps[0], car_caps[-1], 1, 0], xlab = "Carrying capacity ($K$)", ylab="Refuge")
    heatmap_plotter([strategies_comp[:,:,0].T, strategies_comp[:,:,1].T], "increasing_competition"+file_append, [competition[0], competition[-1], 1, 0], xlab = "Competition ($c$)", ylab="Refuge")

    fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
    fig.set_size_inches((15 / 2.54, 5 / 2.54))
    ax[0].plot(qs, populations_ref[:,0], c=tableau20[0])
    ax[0].plot(qs, populations_ref[:,1], c=tableau20[6])
    ax[0].set_xlabel('Refuge quality ($q$)')
    ax[0].text(1.05, 0.9, 'A', transform=ax[0].transAxes)

    ax[1].plot(car_caps, populations_car[:,0], c=tableau20[0])
    ax[1].plot(car_caps, populations_car[:,1], c=tableau20[6])
    ax[1].set_xlabel('Carrying capacity ($K$)')
    ax[1].text(1.05, 0.9, 'B', transform=ax[1].transAxes)

    ax[2].plot(competition, populations_comp[:,0], c=tableau20[0])
    ax[2].plot(competition, populations_comp[:,1], c=tableau20[6])
    ax[2].set_xlabel('Competition ($c$)')
    ax[2].text(1.05, 0.9, 'C', transform=ax[2].transAxes)

    ax[0].set_ylabel('Biomass')

    fig.tight_layout()
    plt.savefig("./results/plots/pop_levels" + file_append + ".pdf")

giant_simulator(method=simple_method, file_append='_c', steps = 40, layers=3)

#giant_simulator(method=discrete_patches, file_append='_p')