import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from clothoids.model import clothoids

def example1(save_figure=False):

    #### USER INPUT
    c0 = [0, 0]
    a0 = numpy.deg2rad(45)
    sf = 2
    phi = numpy.array([
        [5, 3],
        [-5, -2],
        [-2, -3],
        [-1, -3]
    ])
    N = 5000
    #### USER INPUT


    phi = numpy.asarray(phi)
    if phi.ndim == 1:
        Nseg = 1
    else:
        Nseg = phi.shape[0]
    s, _, _, c = clothoids(c0, a0, sf, phi, N)
    c_fun = interp1d(s, c.T, kind='cubic')

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(c[:, 0], c[:, 1], '-r')
    s_segs = numpy.linspace(s[0], s[-1], Nseg+1)
    for i in range(Nseg+1):
        c_seg = c_fun(s_segs[i])
        ax.plot(c_seg[0], c_seg[1], 'or')
    ax.set_aspect('equal')
    ax.axis('off')
    if save_figure:
        fig.savefig('example1.pdf')

if __name__ == '__main__':
    example1(save_figure=False)
    plt.show()
