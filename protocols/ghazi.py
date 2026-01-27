import numpy as np
import random
import math
from protocols.bell import build_A
from typing import List, Tuple
import scipy

def vrand(x, eps) -> np.array:
    """
    Algorithm VRand (Theorem 9 of "On computing Pairwise Statistics with Local Differential Privacy" by 
    Ghazi et al. ). 
    Implementation from Definition 5.5 of "Towards Instance-Optimal Private Query Release" by Blasiok et al.
    
    :param x: Expected value of the result
    :param eps: epsilon-DP parameter

    :return: np.array res:
    """
    norm = np.linalg.norm(x, 2)

    assert norm <= 1
    p1 = (norm + 1) / 2 
    Ux = random.choices((x / p1, -x / p1), weights=(p1 * 100, 100 - p1 * 100))

    Z = np.random.normal(0, 1, x.shape)

    sign = 2 * (np.dot(Ux, Z) > 0) - 1
    
    p2 = (eps * p1 * sign ) / 6 + 1/2
    Sx = random.choices((1, -1), weights=(100 * p2, 1 - 100*p2))

    res = ((3 / (math.sqrt(math.pi * eps)) ) * Sx[0]) * Z
    return res

def build_one_hot(l, bins, x) -> np.array:
    """
    Returns a one_hot of length bins
    
    :param l: Values corresponding to each bin
    :param bins: Number of bins
    :param x: Value to consider

    :return: np.array res: 
    """
    res = np.zeros(bins)
    for i in range(bins):
        if l[i] <= x <= l[i + 1]:
            res[i] = 1.0
            return res
        
def max_col_norm(A: np.array) -> float:
    """
    Returns the maximum norm on the columns
    
    :param A: Matrix to consider
    :type A: np.array

    :return: float m: maximal norm
    """
    n = A.shape[1]
    m = max([np.linalg.norm(A[:, i]) for i in range(n)])
    return m
        
def decompose(W: np.array, eps: float, n: int) -> Tuple:
    """
    Returns a decomposition of the matrix W from 
    
    :param np.array W: Matrix to decompose
    :param float eps: epsilon-DP parameter
    :param int n: Number of parties

    :return: 
        - np.array L: Left matrix
        - np.array R: Right matrix
    """
    k = W.shape[1]
    beta = 1 / (eps * math.sqrt(n))
    l = int(math.log(k) * beta**(-1) )

    A = np.random.normal(loc=0.0, scale=1.0, size=(k, k))
    for i in range(k):
        norm = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i] / norm

    A = A[0:l]

    X, Y = np.identity(k), W

    L = np.dot(A, X)
    R = np.dot(A, Y)

    return L, R


def ghazi(n: int, eps: float, data: np.array, bounds: Tuple, bins: int, funct) -> Tuple:
    """
    Ghazi's non interactive algorithm (Ghazi)
    
    :param int n: Number of parties
    :param float eps: DP parameter
    :param data: Data
    :param bounds: (min, max) of the input space \mathbb{X}
    :param bins: Number of bin
    :param funct: Function to consider

    :return: 
        - float res: Result
        - np.array L: Left decomposition of A
        - np.array R: Right decomposition of A
        - int C: Variable C in the paper "On computing pairwise statistics 
        with local differential privacy."
        - int m: Variable \ell in the paper "On computing pairwise 
        statistics with local differential privacy."
        - List l: Values corresponding to each bin
    """
    eps = eps / 2
    l, A = build_A(bounds, bins, funct)
    L, R = decompose(A, eps, n)
    m = L.shape[0]

    C = int(math.sqrt(max_col_norm(L) * max_col_norm(R)) * math.sqrt(bins) )

    ll = np.zeros(m)
    rr = np.zeros(m)

    for i in range(n):
        x = build_one_hot(l, bins, data[i])
        left = np.dot(L, x) / C
        right = np.dot(R, x) / C

        ll += vrand(left, eps) * C 
        rr += vrand(right, eps) * C

    res = np.dot(ll / n, rr / n)
    return res, L, R, m, l


def ghazi_shuffle(n, eps, data, bins, L, R, m, l) -> float:
    """
    Ghazi's algorithm under the shuffled model implemented via 
    the article "Private summation in the multi-message shuffle model" from 
    Balle et al., Section 5
    
    :param n: Number of parties
    :param eps: DP parameter
    :param data: Data
    :param bins: Number of bins
    :param L: Left decomposition of A
    :param R: Right decomposition of A
    :param m: Variable \ell in the paper "On computing pairwise 
        statistics with local differential privacy."
    :param l: Values corresponding to each bin

    :return: float result: GhaziSM protocol's result 
    """
    p = math.sqrt(n)
    q = int(2 * n * p)
    eps = eps / 2
    alpha = math.exp(-eps / p)
    
    def bernoulli(probability):
        if random.random() <= probability:
            return 1
        return 0

    def encode(x):
        fl = math.floor(x * p)
        return fl + bernoulli(x * p - fl)

    ll = np.zeros(m)
    rr = np.zeros(m)

    for i in range(n):
        x = build_one_hot(l, bins, data[i])
        left = np.dot(L, x)
        right = np.dot(R, x)

        lll = np.array([encode(tmp) for tmp in left])
        rrr = np.array([encode(tmp) for tmp in right])

        ll += lll
        rr += rrr

    lnoise = scipy.stats.dlaplace.rvs(alpha, size=m) / n
    rnoise = scipy.stats.dlaplace.rvs(alpha, size=m) / n

    return np.dot(ll / (p * n) + lnoise, rr / (p * n) + rnoise)





if __name__ == "__main__":

    def f(a, b):
        return abs(a - b)
    
    dl = scipy.stats.dlaplace(0.8)
    print(dl.rvs(size=10))
    print(ghazi(2, 0.1, [0.1, 0.5], [0.0, 1.0], 50, f))