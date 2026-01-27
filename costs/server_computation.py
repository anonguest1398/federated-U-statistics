import math

from costs.utils import JL_dim, r

def Umpc_serv_comp(n) -> int:
    """
    Returns the number of operations on the server side 
    for the Umpc protocol
    
    :param int n: Number of parties

    :return: int result: Number of operations
    """
    return n

def Bell_serv_comp(n, bins) -> int:
    """
    Returns the number of operations on the server side 
    for the Bell protocol
    
    :param int n: Number of parties
    :param int bins: Number of bins

    :return: int result: Number of operations
    """
    return math.comb(n, 2) * bins**2

def Ghazi_serv_comp(n, bins, eps) -> int:
    """
    Returns the number of operations on the server side 
    for the Ghazi protocol
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param float eps: DP parameter

    :return: int result: Number of operations
    """
    return 2 * JL_dim(n, bins, eps) * n

def GhaziSM_serv_comp(n, bins, eps) -> int:
    """
    Returns the number of operations on the server side 
    for the Ghazi protocol under the Shuffled model via [18]
    
    :param int n: Number of parties
    :param int bins: Number of bins
    :param float eps: DP parameter

    :return: int result: Number of operations
    """
    return 2 * JL_dim(n, bins, eps) * r(n) * n

