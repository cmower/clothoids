import numpy
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from clothoids.model import clothoids

def _rot(a, deg=False):
    if deg:
        a = numpy.deg2rad(a)
    return numpy.array([[
        [cos(a), -sin(a)],
        [sin(a), cos(a)]
    ]])

def example1(save_figure=False):
    """Plots a number of clothoid segments."""

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

def example2(save_figure=False):
    """Plots the same example as in example1 but with a sine wave on top of the clothoid."""

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
    p = [0.05, 5, 2]
    N = 10000
    #### USER INPUT

    phi = numpy.asarray(phi)
    if phi.ndim == 1:
        Nseg = 1
    else:
        Nseg = phi.shape[0]
    s, _, cd, c = clothoids(c0, a0, sf, phi, N)
    c_fun = interp1d(s, c.T, kind='cubic')
    sin_overlay = numpy.zeros((s.shape[0], 2))
    R = _rot(90, True)
    for i in range(s.shape[0]):
        scale = p[0]*sin(2*pi*p[1]*s[i]+p[2])
        sin_overlay[i, :] = c[i, :] + scale*R @ cd[i, :]

    sin_overlay_fun = interp1d(s, sin_overlay.T)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(c[:, 0], c[:, 1], '-r')
    s_segs = numpy.linspace(s[0], s[-1], Nseg+1)
    for i in range(Nseg+1):
        c_seg = c_fun(s_segs[i])
        s_ov_seg = sin_overlay_fun(s_segs[i])
        ax.plot(c_seg[0], c_seg[1], 'or')
        ax.plot(s_ov_seg[0], s_ov_seg[1], 'ob')
    ax.plot(sin_overlay[:,0], sin_overlay[:,1], '-b')
    ax.set_aspect('equal')
    ax.axis('off')
    if save_figure:
        fig.savefig('example2.pdf')

def example3(save_figure=False):
    """Plots a similar example as in example2 but with a cycloid on top of the clothoid."""

    #### USER INPUT
    c0 = [0, 0]
    a0 = numpy.deg2rad(45)
    sf = 2
    phi = numpy.array([
        [5, 3],
        [-5, -2],
        [-2, -3],
        [-1, -3],
        [10, 2],
        [-12, 12]
    ])
    phi = -phi
    p = [0.05, 10, 2]
    N = 10000
    #### USER INPUT

    phi = numpy.asarray(phi)
    if phi.ndim == 1:
        Nseg = 1
    else:
        Nseg = phi.shape[0]
    s, _, cd, c = clothoids(c0, a0, sf, phi, N)
    c_fun = interp1d(s, c.T, kind='cubic')
    cycloid_overlay = numpy.zeros((s.shape[0], 2))

    for i in range(s.shape[0]):
        scale = p[0]
        R = _rot(2*pi*p[1]*s[i]+p[2])
        cycloid_overlay[i, :] = c[i, :] + scale*R @ cd[i, :]

    sin_overlay_fun = interp1d(s, cycloid_overlay.T)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(c[:, 0], c[:, 1], '-r')
    s_segs = numpy.linspace(s[0], s[-1], Nseg+1)
    for i in range(Nseg+1):
        c_seg = c_fun(s_segs[i])
        s_ov_seg = sin_overlay_fun(s_segs[i])
        ax.plot(c_seg[0], c_seg[1], 'or')
        ax.plot(s_ov_seg[0], s_ov_seg[1], 'ob')
    ax.plot(cycloid_overlay[:,0], cycloid_overlay[:,1], '-b')
    ax.set_aspect('equal')
    ax.axis('off')
    if save_figure:
        fig.savefig('example3.pdf')

if __name__ == '__main__':
    example1(save_figure=False)
    example2(save_figure=False)
    example3(save_figure=False)
    plt.show()
