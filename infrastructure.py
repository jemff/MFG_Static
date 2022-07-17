import numpy as np

class simple_method:
    def __init__(self, depth, total_points):
        tot_points = total_points

        self.x = np.linspace(0, depth, tot_points)

        #self.M = depth / (tot_points - 1) * 0.5 * (np.identity(tot_points) + np.diag(np.ones(tot_points - 1), -1))
        self.M =  2/3*np.identity(tot_points) + 1/6*np.diag(np.ones(tot_points - 1), -1) + 1/6*np.diag(np.ones(tot_points - 1), 1)
        self.M[0,0] = 1/3
        self.M[-1,-1] = 1/3
        h = (tot_points-1)/depth

        self.M = self.M/h
        self.D = h/(2)*self.fin_diff_mat(tot_points)

    def fin_diff_mat(self, N):
        D = np.zeros((N, N))
        D[0, 0] = -3
        D[-1, -1] = 3
        D[0, 2] = -1
        D[-1, -3] = 1
        D = D - np.diag(np.ones(N - 1), -1)
        D = D + np.diag(np.ones(N - 1), 1)
        D[0, 1] += 3
        D[-1, -2] -= 3

        return D


class spectral_method:
    def __init__(self, depth, layers, segments=1):

        self.n = layers
        self.x = self.JacobiGL(0, 0, layers - 1)

        D_calc = lambda n: np.matmul(np.transpose(self.vandermonde_dx()),
                                     np.linalg.inv(np.transpose(self.vandermonde_calculator()))) * (depth / 2)

        self.D = D_calc(layers)
        M_calc = lambda n: np.dot(np.linalg.inv(self.vandermonde_calculator()),
                                  np.linalg.inv(np.transpose(self.vandermonde_calculator()))) * (depth / 2)

        self.M = M_calc(layers)
        self.x = ((self.x + 1) / 2) * depth
        self.segments = segments

        if segments > 1:
            M_T = np.zeros((layers * segments, layers * segments))
            D_T = np.zeros((layers * segments, layers * segments))
            x_T = np.zeros(layers * segments)
            s_x = depth / segments
            x_n = np.copy(self.x) / segments

            for k in range(segments):
                M_T[k * layers:(k + 1) * layers, k * layers:(k + 1) * layers] = self.M / segments
                D_T[k * layers:(k + 1) * layers, k * layers:(k + 1) * layers] = self.D / segments
                x_T[k * layers:(k + 1) * layers] = x_n + k * s_x

            self.D = D_T
            self.M = M_T
            self.x = x_T


    def JacobiGL(self, a, b, n):
        alpha = a + 1
        beta = b + 1
        N = n - 2
        if N == 0:
            x = np.array([(alpha - beta) / (alpha + beta + 2)])
            w = 2
        else:
            h1 = 2 * np.arange(0, N + 1) + alpha + beta
            J1 = np.diag(-1 / 2 * (alpha ** 2 - beta ** 2) / (h1 + 2) / h1)
            J2 = np.diag(2 / (h1[0:N] + 2) * np.sqrt(np.arange(1, N + 1) * (np.arange(1, N + 1) + alpha + beta) *
                                                     (np.arange(1, N + 1) + alpha) * (np.arange(1, N + 1) + beta) * (
                                                             1 / (h1[0:N] + 1)) * (1 / (h1[0:N] + 3))), 1)
            J = J1 + J2
            J = J + J.T
            x, w = np.linalg.eig(J)

        return np.array([-1, *np.sort(x), 1])

    def JacobiP(self, x, alpha, beta, n):
        P_n = np.zeros((n, x.shape[0]))
        P_n[0] = 1
        P_n[1] = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
        for i in range(1, n - 1):
            an1n = 2 * (i + alpha) * (i + beta) / ((2 * i + alpha + beta + 1) * (2 * i + alpha + beta))
            ann = (alpha ** 2 - beta ** 2) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta))
            anp1n = 2 * (i + 1) * (i + alpha + beta + 1) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta + 1))

            P_n[i + 1] = ((ann + x) * P_n[i] - an1n * P_n[i - 1]) / anp1n

        return P_n

    def JacobiP_n(self, x, alpha, beta, n):
        P_n = self.JacobiP(x, alpha, beta, n)
        if alpha == 1 and beta == 1:
            gamma = lambda alpha, beta, m: 2 ** (3) * (m + 1) / (m + 2) * 1 / ((2 * m + alpha + beta + 1))
        elif alpha == 0 and beta == 0:
            gamma = lambda alpha, beta, m: 2 / ((2 * m + alpha + beta + 1))

        for i in range(n):
            d = np.sqrt(gamma(alpha, beta, i))
            P_n[i] = P_n[i] / d

        return P_n

    def GradJacobi_n(self, x, alpha, beta, n):
        P_diff = np.zeros((n, x.shape[0]))
        JacobiPnorma = self.JacobiP_n(x, alpha + 1, beta + 1, n)
        for i in range(1, n):
            P_diff[i] = JacobiPnorma[i - 1] * np.sqrt(i * (i + alpha + beta + 1))
        return P_diff

    def vandermonde_calculator(self):
        n = self.n
        x = self.x
        return (self.JacobiP_n(x, 0, 0, n))

    def vandermonde_dx(self):
        n = self.n
        x = self.x
        return (self.GradJacobi_n(x, 0, 0, n))

    def expander(self, old_spectral, transform_vec):
        new_spectral = self
        length = old_spectral.x[-1]
        coeffs = np.linalg.inv(old_spectral.JacobiP_n(2 * old_spectral.x / length - 1, 0, 0, old_spectral.n).T) @ transform_vec
        transformer = new_spectral.JacobiP_n(2 * new_spectral.x / new_spectral.x[-1] - 1, 0, 0, old_spectral.n).T
        return transformer @ coeffs

    def interpolater(self, new_points, transform_vec):
        length = self.x[-1]
        coeffs = np.linalg.inv(self.JacobiP_n(2 * self.x / length - 1, 0, 0, self.n).T) @ transform_vec
        transformer = self.JacobiP_n(2 * new_points / new_points[-1] - 1, 0, 0, self.n).T

        return transformer @ coeffs

class discrete_patches:
    def __init__(self, depth, total_points):
        self.x = np.linspace(0, depth, total_points)

        self.M = depth/total_points * np.identity(total_points)
        self.M[0,0] = 1/2*self.M[0,0]
        self.M[-1, -1] = 1 / 2 * self.M[-1, -1]

def heat_kernel(Mx, t, k):
    gridx, gridy = np.meshgrid(Mx.x, Mx.x)
    ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * k * t)) + np.exp(-(-y - x) ** 2 / (4 * k * t)) + np.exp(-(2*Mx.x[-1] - x - y) ** 2 / (4 * k * t))
    out = (4 * t * k * np.pi) ** (-1 / 2) * ker(gridx, gridy)
    return Mx.M @ out


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=10)

def heatmap_plotter(data, image_name, ext, xlab, ylab):

    """
    The function to create heatmaps
    :param data: The data to plot, given as a list
    :param image_name: The name of the resulting heatmap
    :param ext: The range of the heatmap
    :return: None, the function saves the heatmaps
    """
    fig = plt.figure(figsize=(12/2.54, 12/2.54))

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1,len(data)),
                     axes_pad=0.2,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.2,
                     )

    # Add data to image grid
    type = [" Consumer distribution", " Predator distribution"]
    i = 0
    lets = ["A", "B"]

    for ax in grid:
        im = ax.imshow(data[i], vmin=0, cmap='viridis', extent=ext, origin = 'lower')
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(np.abs((x1 - x0) / (y1 - y0)))
        ax.set_xlabel(xlab + " \n" + type[i % 2])
        ax.set_ylabel(ylab)
        ax.text(1.025, 0.9, lets[i], transform=ax.transAxes)

        i += 1
    import matplotlib as matplotlib
    # Colorbar
    #ax.cax.colorbar(im)
    #ax.cax.toggle_label(True)
    ax.cax.cla()
    thecb = matplotlib.colorbar.Colorbar(ax.cax, im)
    thecb.set_label("\% of biomass", rotation=270, labelpad=12)
    ax.cax.toggle_label(True)

    plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible

    #fig.colorbar(im, cax=cax)

    plt.savefig('results/plots/'+image_name+".pdf", bbox_inches='tight')


def heatmap_plotter_big(data, image_name, ext, xlab, ylab):

    """
    The function to create heatmaps
    :param data: The data to plot, given as a list
    :param image_name: The name of the resulting heatmap
    :param ext: The range of the heatmap
    :return: None, the function saves the heatmaps
    """
    data = []
    fig = plt.figure(figsize=(12 / 2.54, 12 / 2.54))

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(2, 2),
                     axes_pad=0.2,
                     share_all=False,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.2,

                     )

    # Add data to image grid
    #ext = [-12, 12, 90, 0]
    i = 0

    lets = ["A", "C", "B", "D"]

    type = [" Consumer ", " Predator"]
    for val, ax in zip(data, grid):
        im = ax.imshow(val, vmin=0, cmap='viridis', extent=ext)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))
        ax.set_aspect(np.abs((x1 - x0) / (y1 - y0)))
        ax.set_xlabel(xlab + " " + type[i % 2])
        ax.set_ylabel(ylab)
        ax.set_ylabel("Refuge")
        ax.text(1.025, 0.9, lets[i], transform=ax.transAxes)

        i += 1

    ax.cax.cla()
    thecb = matplotlib.colorbar.Colorbar(ax.cax, im)
    thecb.set_label("\% of biomass", rotation=270, labelpad=12)
    ax.cax.toggle_label(True)

    plt.savefig("total_heatmap" + ".pdf", bbox_inches='tight')

    return ax