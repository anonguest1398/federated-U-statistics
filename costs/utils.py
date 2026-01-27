import math

def beta(n, eps) -> float:
    """
    Returns the variable \beta which is set to 1 / eps * \sqrt(n)
    according to [7]
    
    :param int n: Number of parties
    :param float eps: DP parameter

    :return: float result: beta
    """
    return 1 / (eps * math.sqrt(n))

def r(n, kappa=80) -> int:
    """
    Returns the number of shuffling servers needed to achieve
    kappa statistical security in [18, Sect. 5]
    
    :param int n: Number of parties
    :param float kappa: Security parameter in [18, Sect. 5]

    :return: int result: Number of servers
    """
    q = 2 * n * math.sqrt(n)
    return int((5 / 2) * math.log(q) + math.log(n - 1) + 0.25 * math.log(math.log(q) + math.log(n - 1))) + kappa

def JL_dim(n, bins, eps) -> int:
    """
    Returns the reduced dimension d returned bythe JL theorem. 
    We use here the bound found by Dasgupta and Gupta in
    "An Elementary Proof of a Theorem of Johnson and Lindenstrauss".
    
    :param n: Description
    :param bins: Description
    :param eps: Description

    :return: int result: Reduced dimension
    """
    return int(4 * ( (beta(n, eps)**2) / 2 - (beta(n, eps)**3)/3)**(-1) * math.log(bins))
