import numpy
from numpy import sin, cos
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

def _clothoid_segment(c0, phi0, phi1, a0, s0, sf, N):

    # Compute s and a and cd
    s = numpy.linspace(s0, sf, N)
    a = numpy.array([a0 + phi0*(s[i]-s0) + 0.5*phi1*(s[i]-s0)**2 for i in range(N)])
    cd = numpy.array([[cos(a[i]), sin(a[i])] for i in range(N)])

    # Interpolate cd
    cd_interp_fun = interp1d(s, cd.T, kind='cubic')
    def cd_fun(s, c):
        return cd_interp_fun(s)

    # Integrate cd
    res = solve_ivp(cd_fun, t_span=(s0, sf), y0=c0, t_eval=s)
    c = res.y.T

    return s, a, cd, c

def clothoids(c0, a0, sf, phi, N):
    """Generate a number of stitched-together clothoid segments.


    Input
    -----

    c0 [array-like with 2 elements]
        The start position for the path.

    a0 [float]
        The start orientation for the path (in radians).

    sf [float]
        The final value for s. The range of s values will be [0, sf].

    phi [array-like with shape Nseg-by-2]
        Linear parameters for each clothoid segment. Each row corresponds to a
        clothoid segment. If you want only one segment, then pass a list with 2
        elements.

    N [int]
        Discretization of each clothoid segment.


    Output
    ------

    Note, in the following n = N + N*(Nseg-1) (for N/Nseg, see above).

    s [numpy.ndarray with shape (n,)]
        An n-element linear discretization of the range [0, sf].

    a [numpy.ndarray with shape (n,)]
        The corresponding orientation path for the corresponding values in s.

    cd [numpy.ndarray with shape (n, 2)]
        The unit directional gradient for the clothoid segment for the
        correspinding values in s.

    c [numpy.ndarray with shape (n, 2)]
        The clothoid path.

    """

    phi = numpy.asarray(phi)
    if phi.ndim == 1:
        phi = phi.reshape(1, 2)

    Nseg = phi.shape[0]
    s_points = numpy.linspace(0, sf, Nseg+1)

    s_segments = []
    a_segments = []
    cd_segments = []
    c_segments = []
    a0_old = a0
    c0_old = c0
    for i in range(Nseg):
        phi0 = phi[i, 0]
        phi1 = phi[i, 1]
        s0 = s_points[i]
        sf = s_points[i+1]
        s, a, cd, c = _clothoid_segment(c0_old, phi0, phi1, a0_old, s0, sf, N)

        a0_old = a[-1]
        c0_old = c[-1, :]

        if i > 0:
            s = s[1:]
            a = a[1:]
            cd = cd[1:, :]
            c = c[1:, :]

        s_segments += s.flatten().tolist()
        a_segments += a.flatten().tolist()
        cd_segments.append(cd)
        c_segments.append(c)

    s_all = numpy.array(s_segments)
    a_all = numpy.array(a_segments)
    cd_all = numpy.vstack(cd_segments)
    c_all = numpy.vstack(c_segments)

    return s_all, a_all, cd_all, c_all
