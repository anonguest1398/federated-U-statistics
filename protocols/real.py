import numpy as np

def compute_real_U(data: np.array, n: int, f) -> float:
    """
    Returns the U-statistic U_{f, C^n_2}

    :param np.array data: Data
    :param int n: Number of parties
    :param f: Function to consider

    :return: float result:
    """

    res = 0
    for i in range(n):
        for j in range(i+1, n):
            res += f(data[i], data[j])

    return (2 * res) / (n * (n - 1))