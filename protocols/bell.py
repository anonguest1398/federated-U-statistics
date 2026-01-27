import math
import random
import numpy as np
from typing import List, Tuple


def build_A(r: Tuple, t: int, f) -> Tuple:
    """
    Returns the matrix representing the function f
    
    :param Tuple r: (low, up) where low is the lower bound and up is the upper bound of \mathbb{X}
    :param int t: Number of bins
    :param f: Kernel function

    :return:
        - List l: A list l that specify the values for each bin
        - np.array A: A numpy matrix A representing f
    """

    hope = r[1] / t
    l = []
    A = np.zeros((t, t))
    for i in range(t + 1):
        l.append(i * hope)

    # take the middle of each bin
    ll = []
    for i in range(t):
        ll.append(( l[i] + l[i + 1]) / 2 ) 

    for i in range(t):
        for j in range(t):
            # for now f(x, y) = x + y
            A[i][j] = f(ll[i], ll[j])

    return l, A
    

def get_proba_list(x, beta, t, l) -> List:
    """
    Returns a list of length t with probability res[i] = 1 - \beta when l[i] \leq x \leq l[i+1] 
    and res[i] = \beta / t otherwise.
    
    :param x: Input
    :param beta: Probability
    :param t: Number of bins
    :param l: Values corresponding to each bin

    :return: List res: List of probabilities for generating the random one-hot vector
    """
    res = [0.0]
    cumul = beta / t
    for i in range(t):
        if l[i] <= x < l[i+1]:
            cumul += 1 - beta
        else:
            cumul += beta / t
        res.append(cumul)
    
    return res
            
def build_rand_one_hot(x, beta, t, l) -> np.array:
    """
    Returns the randomized one-hot vector 
    
    :param x: Value to consider
    :param beta: Probability
    :param t: Number of bins
    :param l: Values corresponding to each bin

    :return: np.array res: Randomized one-hot vector
    """
    proba = get_proba_list(x, beta, t, l)
    rand = random.random()
    res = np.zeros(t)

    for i in range(t):
        if proba[i] <= rand < proba[i + 1]:
            
            res[i] = 1.0
            return res

    res[t - 1] = 1.0
    return res

def calculate_f_hat(A, beta, t, l, x_1, x_2) -> float:
    """
    Returns \hat{f}_A(R(x_1), R(x_2))

    :param np.array A: Matrix representation of \hat{f}
    :param beta: Probability  
    :param t: Number of bins
    :param l: Values corresponding to each bin
    :param x_1: Data point 1
    :param x_2: Data point 2

    :return: float res: 
    """
    e_1 = build_rand_one_hot(x_1, beta, t, l)
    e_2 = build_rand_one_hot(x_2, beta, t, l)

    b = np.ones(t) * (beta / t)

    res = ( (1 - beta)**(-2)) * np.dot(np.dot((e_1 - b), A ), (e_2 - b))
    return res
    

def bell_method(n_parties: int, bins: int, data: List[float], bounds: Tuple,  eps: float, f) -> float:
    """
    Algorithm 1 from "Private Protocols for U -Statistics in the Local Model and Beyond"
    by Bell et al. 
    
    :param int n_parties: Number of parties
    :param int bins: Number of bins
    :param List data: Input data
    :param Tuple bounds: (min, max) of the input space \mathbb{X}
    :param float eps: Epsilon-DP parameter
    :param f: Kernel function to consider

    :return: float \hat{U}_f:
    """
    beta = bins / (bins + math.exp(eps) - 1)

    res = 0
    l, A = build_A(bounds, bins, f)

    for i in range(n_parties):
        for j in range(i, n_parties):
            res += calculate_f_hat(A, beta, bins, l, data[i], data[j])
    return (2 * res) / (n_parties * (n_parties - 1))


if __name__ =="__main__":
    pass