import random
from itertools import combinations
import numpy as np
from typing import Tuple, List

# We consider here Additive secret sharing over the ring Z_{2^l} for an
# easy implementation. Other secret sharing schemes could be possible.

def balanced_sample(n: int, m: int) -> List[Tuple[int, int]]:
    """
    BalancedSampling
    
    :param int n: Number of parties
    :param int m: Size of E

    :return: List E: A list of edges
    """
    rng = np.random.default_rng()
    k = 2

    M = int(np.ceil(k*m*1.0/n))

    E = []
    restart_count = 0
    while len(E) < m:
        E = []
        p = n*[M]
        ptot = n*M
        while len(E) < m:
            ptotj = ptot
            e = []
            for j in range(k):
                r = rng.integers(ptotj)
                i = 0
                while (r >= p[i]) or (i in e):
                    if not(i in e):  # no parallel edges
                        r -= p[i]
                    i += 1
                e.append(i)
                ptotj -= p[i]
                if (j < k-1) and (ptotj == 0):
                    break # restart
            if len(e) < k:
                #print("RESTART -------------")
                restart_count += 1
                break  # restart
            # print("add: ",e)
            E.append(e)
            ptot -= k
            for v in e:
                p[v] -= 1
    return E


def ours_balanced(n: int, size_E: int, data, epsilon: float, funct, l=40, c=14, secret_shared=False) -> float:
    """
    Computes the estimator Umpc using BalancedSampling
    
    :param int n: Number of parties
    :param int size_e: Size of E
    :param np.array data: Data
    :param float epsilon: DP parameter
    :param funct: Kernel function to consider
    :param int l: Parameter \ell of the paper
    :param int c: Parameter c of the paper
    :param bool secret_shared: Boolean value to specify if secret sharing is happening or not

    :return: float result: the estimator \hat{U}_{f, E} of U_f under epsilon DP using 
    """
    tuples = balanced_sample(n, size_E)

    # To compute \delta_G^{max}
    max_table = np.zeros(n)

    if secret_shared == False:
        res = 0

        for (i, j) in tuples:
            max_table[i] += 1
            max_table[j] += 1
            # For ease of conversion, let us write 
            # A) Convert(f(x_i, x_j)),
            # B) f(Convert(x_i), Convert(x_j)).
            # For GMD, A is equivalent to B), but for kernel functions with signs and equality tests, A \neq B.
            # B outputs 1 while A outputs 2**c in Z_^{2^l}. 
            res += convert(funct(data[i], data[j]), c)

        noise = convert(np.random.laplace(0, max(max_table) / epsilon), c)
        return invert(res + noise, c) / size_E

    shares = np.zeros(n)

    for (i, j) in tuples:
        max_table[i] += 1
        max_table[j] += 1

        r = convert(funct(data[i], data[j]), c)
        sh0, sh1 = share(r, 2, 2**l)
        shares[i] = (shares[i] + sh0) % 2**l
        shares[j] = (shares[j] + sh1) % 2**l 
    
    noise_b = np.random.laplace(0, max(max_table) / epsilon )
    noise = convert(noise_b, c)

    noise_shares = np.array(share(noise, n, 2**l))

    res = reconstruct( (shares +  noise_shares), 2**l)
    return invert(res, c) / size_E



def ours_wo_repl(n: int, size_e: int, data, epsilon: float, funct, l=40, c=14) -> float:
    """
    Computes the estimator Umpc using sampling without replacement

    :param int n: Number of parties
    :param int size_e: Size of E
    :param np.array data: Data
    :param float epsilon: DP parameter
    :param funct: Function to consider
    :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)

    :return: float result: the estimator \hat{U}_{f, E} of 
        U_f under epsilon DP using 
    """

    r = range(n)
    res = 0

    # To compute \delta_G^{max}
    max_table = np.zeros(n)

    tuples = random.choices(list(combinations(r, 2)), k = size_e)
    for (i, j) in tuples:
        res += convert(funct(data[i], data[j]), c)
        cpt += 1

        max_table[i] += 1
        max_table[j] += 1

    noise = convert(np.random.laplace(0, max(max_table) / epsilon), c)

    return invert(res + noise, c) / size_e



def ours_bernoulli(n: int, alpha: float, data, epsilon: float, funct, l=40, c=14) -> float:
    """
    Computes the estimator Umpc using Bernoulli sampling
    
    :param int n: Number of parties
    :param float alpha: Probability of picking a specific tuple
    :param np.array data: Data
    :param float epsilon: DP parameter
    :param funct: Function to consider
    :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)

    :return: float result: the estimator \hat{U}_{f, E} 
        of U_f under epsilon DP using Bernoulli sampling
    """

    comb = combinations([i for i in range(0,n)], 2)
    res = 0
    cpt = 0

    # To compute \delta_G^{max}
    max_table = np.zeros(n)

    for (i, j) in comb:
        r = random.random()
        if r < alpha:
            cpt += 1
            res += convert(funct(data[i], data[j]), c)

            max_table[i] += 1
            max_table[j] += 1

    noise = convert(np.random.laplace(0, max(max_table) / epsilon), c)

    return invert(res + noise, c) / cpt

def convert(x: float, c: int, l=40) -> int:
    """
    Converts a float to Z_{2^\ell} with scaling factor h=2^{-c}
    
    :param float x: Float to convert
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)
    :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)

    :return: int result: Reprensation of x in Z_{2^\ell}
    """
    h = 2**c
    repr = x * h
    return repr

def invert(y: int, c:int, l=40) -> float:
    """
    Corresponds to the inverse of convert.
    
    :param int y: Integer in Z_{2^l} to invert
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)
    :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)

    :return float result: convert^{-1}(y)
    """
    h = 2**(-c)
    x = y * h
    return x

def share(secret: int, n: int, ring_size: int) -> List:
    """
    Additive secret sharing Share algorithm
    
    :param int secret: Secret to share
    :param int n: Number of parties
    :param int ring_size: Ring size (a power of 2)

    :return: List result: Shares of secret
    """
    shares = [random.randrange(ring_size) for _ in range(n-1)]
 
    # Append final share by subtracting all shares from secret
    shares.append((secret - sum(shares)) % ring_size )
    return shares

def reconstruct(shares: List, ring_size: int):
    """
    Reconstruct the secret from shares
    
    :param List shares: List of shares 
    :param int ring_size: Ring size (a power of 2)

    :return: int result: Return the secret from shares
    """
    return sum(shares) % ring_size


if __name__ == "__main__": 
    pass