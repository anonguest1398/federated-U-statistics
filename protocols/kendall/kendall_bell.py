import numpy as np
import random
from typing import List, Tuple
import math

def sign(x):
    if x < 0:
        return -1
    if x == 0:
        return 0
    return 1


def f(a, b):
    return sign(a[0] - b[0]) * sign(a[1] - b[1])

def build_A_kendall(r: Tuple, t: int, f) -> Tuple:
    """
    Returns the matrix of size (t^2, t^2) representing the function f.
    In Kendall's tau, the input z_i = (x_i, y_i) for i \in [n]. 
    A is of the form:
    
    [[(x_1, y_1)=(x_1, y_1)   (x_1, y_1)=(x_1, y_1)    ...    (x_1, y_1)=(x_1, y_t)    (x_1, y_1)=(x_2, y_1)    ...     (x_1, y_1)=(x_t, y_t)]  
    [(x_1, y_2)=(x_1, y_1)    (x_1, y_2)=(x_1, y_2)    ...    (x_1, y_2)=(x_1, y_t)    (x_1, y_2)=(x_1, y_2)    ...     (x_1, y_2)=(x_t, y_t)]  
    ...
    [(x_1, y_t)=(x_1, y_1)    (x_1, y_t)=(x_2, y_1)    ...    (x_1, y_t)=(x_1, y_t)    (x_1, y_t)=(x_2, y_1)    ...    (x_1, y_t)=(x_t, y_t)]
    [(x_2, y_1)=(x_1, y_1)    (x_2, y_1)=(x_2, y_1)    ...    (x_2, y_1)=(x_1, y_t)    (x_2, y_1)=(x_1, y_2)    ...    (x_2, y_1)=(x_t, y_t)]
    ...
    [(x_2, y_t)=(x_1, y_1)    (x_2, y_t)=(x_2, y_1)    ...    (x_2, y_t)=(x_1, y_t)    (x_2, y_t)=(x_2, y_1)    ...    (x_2, y_t)=(x_t, y_t)]
    ...
    [(x_t, y_t)=(x_1, y_1)    (x_t, y_t)=(x_2, y_1)    ...    (x_t, y_t)=(x_1, y_t)    (x_t, y_t)=(x_2, y_1)    ...    (x_t, y_t)=(x_t, y_t)]]

    
    :param Tuple r: (min, max) of the input space \mathbb{X}
    :param int t: Number of bins
    :param f: Kernel function to consider

    :return:
        - List l: Values for each bin
        - np.array A: Matrix representing f
    """
    hope = r[1] / (t)
    l = []
    A = np.zeros((t**2, t**2))
    for i in range(t + 1):
        l.append(i * hope)

    # take the middle of each bin
    ll = []
    for i in range(t):
        ll.append(( l[i] + l[i + 1]) / 2 ) 

    m = t 
    for i in range(m):
        for j in range( m ):
            for k in range(m):
                for v in range(m):
                    # print((ll[i], ll[j]), (ll[k], ll[v] ) )
                    A[i + j, k + v] = f( (ll[i], ll[j]), (ll[k], ll[v] ))
    return l, A

def get_proba_list(x, beta, t, l) -> List:
    """
    Returns a list of length t**2 with probability res[i] = 1 - \beta when l[i] \leq x \leq l[i+1] 
    and res[i] = \beta / t otherwise.
    
    :param x: Input
    :param beta: Probability
    :param t: Number of bins
    :param l: Values corresponding to each bin

    :return: List res: List of probabilities for generating the random one-hot vector
    """
    res = [0.0]
    cumul = beta / (t**2)
    for i in range(t):
        for j in range(t):
            if l[i] <= x[0] < l[i+1] and l[j] <= x[1] < l[j + 1]:
                cumul += 1 - beta
            else:
                cumul += beta / (2 * t)
            res.append(cumul)

    return res

def build_rand_one_hot(x, beta, t, l) -> np.array:
    """
    Returns the randomized one-hot vector of size t**2
    
    :param x: Value to consider
    :param beta: Probability
    :param t: Number of bins
    :param l: Values corresponding to each bin

    :return: np.array res: Randomized one-hot vector
    """
    proba = get_proba_list(x, beta, t, l)

    rand = random.random()

    res = np.zeros(t**2)

    for i in range(t**2):
        if proba[i] <= rand < proba[i + 1]:
            
            res[i] = 1.0
            return res

    res[t - 1] = 1.0
    return res 

def calculate_f_hat(A, beta, t, l, x_1, x_2) -> float:
    """
    Returns \hat{f}_A(R(x_1), R(x_2)) for Kendall's tau

    :param np.array A: Matrix representation of \hat{f}
    :param beta: Probability  
    :param t: Number of bins
    :param l: Values corresponding to each bin
    :param x_1: Data point 1 of the form (y_1, z_1)
    :param x_2: Data point 2 of the form (y_2, z_2)

    :return: float res: 
    """
    e_1 = build_rand_one_hot(x_1, beta, t, l)
    e_2 = build_rand_one_hot(x_2, beta, t, l)

    b = np.ones(t**2) * (beta / t)

    res = (1 / (1 - beta)**2) * np.dot(np.dot((e_1 - b), A ), (e_2 - b))
    return res

def compute_U_hat(data: List[int], beta: float, n: int, t: int, r: List[int], f):
    """
    data (List[int]): data
    beta (float):
    n: (int): number of data points
    t (int): number of bins
    r: (List[int]): (min, max) of data
    f (function)
    """
    
def kendall_bell(n_parties, bins, data, r, eps, func) -> float:
    """
    Algorithm 1 from Private Protocols for U -Statistics in the Local Model and Beyond
    by Bell et al. for Kendall's tau
    
    :param int n_parties: Number of parties
    :param int bins: Number of bins
    :param np.array data: Input data
    :param Tuple bounds: (min, max) of the input space \mathbb{X}
    :param float eps: Epsilon-DP parameter
    :param f: Kernel function to consider

    :return: float result: \hat{U}_f
    """
    beta = bins / (bins + math.exp(eps) - 1)

    res = 0
    l, A = build_A_kendall(r, bins, f)

    for i in range(n_parties):
        for j in range(i, n_parties):
            res += calculate_f_hat(A, beta, bins, l, data[i], data[j])
    return (2 * res) / (n_parties * (n_parties - 1))


if __name__ == "__main__":
    pass    