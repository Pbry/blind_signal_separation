import numpy as np


def g(y):
    return (np.tanh(y))


def g_p(y):
    return (1 - np.tanh(y) ** 2)


# g_p is g's derivative

def process(x, w):
    # x is a (2,n) vector containing the two signals of length n, from which we want to extract two source signals
    # x has to be whitened
    # this function calculates the new axis based on the previous one
    n = len(x[0])
    g_wx = g(np.dot(w, x))
    # we perform a term by term multiplication between x and g_wx
    t1 = 1 / n * (x * np.array([g_wx, g_wx])).sum(axis=1)

    t2 = 1 / n * g_p(np.dot(w, x)).sum(axis=0)

    w2 = t1 - [t2 * w[0], t2 * w[1]]

    return (w2 / np.linalg.norm(w2))


def ICA(x, e):
    """x is a (2,n) vector containing the two signals of length n, from which we want to extract two source signals
    x has to be whitened
    This function gives one axis for the decomposition
    In ICA, we consider the algorithm converges when the sine of the angle (w, w+)
    is smaller than some precision e"""
    # initialisation
    nb_iter = 0
    w = [np.random.random_sample(), np.random.random_sample()]
    w = w / np.linalg.norm(w)
    print("w init : ", w, '\n')
    w1 = process(x, w)

    while ((1 - np.dot(w, w1) ** 2 > e) and nb_iter < 10 ** 3):
        w = w1
        w1 = process(x, w1)
        nb_iter += 1

    print("Angular distance w to w+ : ", 1 - np.dot(w, w1) ** 2)
    return (w)

def ICA_all(x,e):
    """This function returns the two axis given by the ICA, on which we can project the observations x
    to obtain the two signals that concern us"""
    w1 = ICA(x,e)
    w2 = ICA(x,e)
    proj = np.dot(w2,w1)
    w2 = w2 - proj*w1
    w2 = w2/np.linalg.norm(w2)
    return(w1,w2)

