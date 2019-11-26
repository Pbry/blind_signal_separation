import numpy as np
from math import *


def tr(M):
    N = len(M)
    return(1/N * sum([M[i][i] for i in range(N)]))


def off(M):
    N = len(M)
    o = 0
    for i in range(N):
        for j in range(N):
            if i !=j:
                o += M[i][j]
    return(1/(N*(N-1)) * o)


def intercor(x1, x2, w):
    #calculates the average intercorrelation between x1 and x2
    # w size of the window on x1 and x2
    cov = np.dot(x1[:w].reshape(-1,1), x2[:w].reshape(1,-1)) # initialisation
    n = len(x1)
    for  i in range(1, int(n/w)-1):
        cov += np.dot(x1[i*w:(i+1)*w].reshape(-1,1), x2[i*w:(i+1)*w].reshape(1,-1))
    return(cov/int(n/w))


def SOBI(x,f):
    w = 10  # size of the window used to average intercorrelation matrices

    y = x.transpose()

    Rx1x1 = intercor(y[0], y[0], f)
    F1 = off(Rx1x1)
    T1 = tr(Rx1x1)

    Rx2x2 = intercor(y[1], y[1], f)
    F2 = off(Rx2x2)
    T2 = tr(Rx2x2)

    Rx1x2 = intercor(y[0], y[1], f)
    F12 = off(Rx1x2)
    T12 = tr(Rx1x2)

    alpha = 2 * F12 * T12 - (F1 * T2 + F2 * T1)
    beta = 2 * (T12 ** 2 - T1 * T2)
    gamma = sqrt((F1 * T2 - F2 * T1) ** 2 + 4 * (F12 * T2 - T12 * F2) * (F12 * T1 - T12 * F1))

    d1 = alpha - gamma
    d2 = alpha + gamma

    A_est = np.zeros((2, 2))
    # estimate of matrix A such that x=As, where x is the observation and s the signals we wish to retrieve
    A_est[0, 0] = beta * F1 - T1 * d1
    A_est[0, 1] = beta * F12 - T12 * d2
    A_est[1, 0] = beta * F12 - T12 * d1
    A_est[1, 1] = beta * F2 - T2 * d2

    s = np.dot(np.linalg.inv(A_est), x.transpose())
    return(s)