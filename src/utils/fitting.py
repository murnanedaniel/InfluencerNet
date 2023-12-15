import numpy as np


def conformal_mapping(x, y, z):
    """
    x, y, z: np.array([])
    return:
    """
    # ref. 10.1016/0168-9002(88)90722-X
    r = x**2 + y**2
    u = x / r
    v = y / r
    # assuming the imapact parameter is small
    # the v = 1/(2b) - u x a/b - u^2 x epsilon x (R/b)^3
    pp, vv = np.polyfit(u, v, 2, cov=True)
    b = 0.5 / pp[2]
    a = -pp[1] * b
    R = np.sqrt(a**2 + b**2)
    e = -pp[0] / (R / b) ** 3  # approximately equals to d0
    dev = 2 * e * R / b**2

    magnetic_field = 2.0
    pT = 0.3 * magnetic_field * R  # in MeV
    # print(a, b, R, e, pT)

    p_rz = np.polyfit(np.sqrt(r), z, 2)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(abs(e))

    r3 = np.sqrt(r + z**2)
    p_zr = np.polyfit(r3, z, 2)
    cos_val = p_zr[0] * z0 + p_zr[1]
    theta = np.arccos(cos_val)
    eta = -np.log(np.tan(theta / 2.0))
    # phi = np.atan2(b, a)
    phi = np.arctan2(b, a)

    return e, z0, eta, phi, pT, dev
