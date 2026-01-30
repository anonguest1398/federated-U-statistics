import math

from costs.utils import JL_dim, r


def Umpc_comp(n, p, nbits, kernel="gini", security_lambda=128) -> int:
    """
    Returns the number of operations per party necessary to complete Umpc protocol
    
    :param int n: Number of parties
    :param int p: Number of neighbors per party in protocol Umpc
    :param int nbits: Number of bits to represent one element
    :param str kernel: Name of the kernel function to consider
    :param int security_lambda: Security parameter for FSS

    :return: int result: Number of operations
    """
    # Sharing phase + Noise addition phase + Computation phase
    if kernel == "gini":
        return int(2*nbits*p + n + nbits * security_lambda)
    elif kernel =="kendall":
        return int(2*nbits*p + n + 2 * nbits * security_lambda)
    elif kernel == "duplicate":
        return int(2*nbits*p + n + nbits * security_lambda)

def Bell_comp(bins) -> int:
    """
    Returns the number of operations per party necessary to complete the Bell protocol
    
    :param int bins: Number of bins

    :return: int result: Number of operations
    """
    return bins

def Ghazi_comp(n, bins, eps) -> int:
    """
    Returns the number of operations per party necessary to complete the Ghazi protocol.
    The number of operations is computed via the article [19] "Approximate nearest 
    neighbors and the fast johnson-lindenstrauss transform." by Ailon and Chazelle.
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param float eps: DP parameter

    :return: int result: Number of operations
    """
    return int(2*(bins * math.log(bins) + bins * JL_dim(n, bins, eps)))

def GhaziSM_comp(n, bins, eps) -> int:
    """
    Returns the number of operations per party necessary to complete the 
    Ghazi protocol under the Shuffled model implemented via [18].
    The number of operations is computed via the article [19] "Approximate nearest 
    neighbors and the fast johnson-lindenstrauss transform." by Ailon and Chazelle.
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param float eps: DP parameter

    :return: int result: Number of operations
    """
    return int(2*(bins * math.log(bins) + bins * JL_dim(n, bins, eps)) + bins * (r(n) + math.log(n)))
