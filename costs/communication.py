from costs.utils import JL_dim, r


def fss_keys(n, p, nbits, security_lambda=128) -> int:
    """
    Returns the number of bits associated with the FSS keys required 
    to perform a secure comparison via [21].
    
    :param int n: Number of parties
    :param int p: Number of neighbors per party in Umpc protocol
    :param int nbits: Number of bits to represent one element
    :param int security_lambda: Security parameter for FSS

    :returns: int result: Number of bits
    """
    return security_lambda * nbits * n * p

def Umpc_comm(n, p, nbits, kernel="gini") -> int:
    """
    Returns the number of bits necessary to complete the Umpc protocol
    by using Protocol B.1 for Noise generation
    
    :param int n: Number of parties
    :param int p: Number of neighbors per party in Umpc Protocol
    :param int nbits: Number of bits to represent one element
    :param str kernel: Name of the kernel function

    :returns: int result: Number of bits
    """

    # This corresponds to the bits for:
    # - the Sharing Phase (alpha * n^2),
    # - the Aggregation phase (n * nbits),
    # - the Noise addition phase (nbits * n^2) implemented via the Protocol B.1
    
    bits = (p * n + n) * nbits + nbits * n**2 

    if kernel == "gini":
        # For Gini Mean Difference, 1 product, 1 comparisons = 3 opens
        return int((3 * n * p) * nbits + fss_keys(n, p, nbits) + bits)
    elif kernel == "kendall":
        # For Kendall's tau, 1 product, 2 comparisons = 4 opens
       return int((4 * n * p) * nbits + 2 * fss_keys(n, p, nbits) + bits)
    elif kernel == "duplicate":
        # For Duplicate Pair Ratio, 1 comparison = 1 open
        return int(n * p * nbits + fss_keys(n, p, nbits) + bits)
    
def Ghazi_comm(n, bins, nbits, eps) -> int:
    """
    Returns the number of bits necessary to complete Ghazi et al.'s protocol
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param int nbits: Number of bits to represent one element
    :param float eps: DP parameter

    :returns: int result: Number of bits
    """
    return JL_dim(n, bins, eps) * 2 * nbits * n

def GhaziSM_comm(n, bins, nbits, eps):
    """
    Returns the number of bits necessary to complete Ghazi et al.'s protocol
    under the Shuffled model implemented via [18]
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param int nbits: Number of bits to represent one element
    :param float eps: DP parameter

    :returns: int result: Number of bits
    """
    return JL_dim(n, bins, eps) * 2 * r(n) * nbits * n

def Bell_comm(n, bins, nbits) -> int:
    """
    Returns the number of bits necessary to complete Bell et al.'s protocol
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param int nbits: Number of bits to represent one element

    :returns: int result: Number of bits
    """
    return bins * n * nbits 
