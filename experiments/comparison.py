import numpy as np
from math import comb

from protocols.our import ours_bernoulli, ours_balanced, ours_wo_repl
from protocols.bell import bell_method
from protocols.ghazi import ghazi, ghazi_shuffle
from experiments.load_data import uniform_data, get_kendall_bank_data, get_duplicate_bank_data
from protocols.real import compute_real_U
from protocols.kendall.kendall_bell import kendall_bell
from protocols.kendall.kendall_ghazi import kendall_ghazi, kendall_ghazi_shuffle

def g(a, b) -> float:
    """
    Gini Mean Difference kernel function
    
    :param float a: 
    :param float b:

    :return: float result:
    """
    return abs(a - b)

def h(a, b) -> int:
    """
    Duplicate Pair Ratio
    
    :param int a:
    :param int b:

    :return: int result:
    """
    return a == b

def gini_main(n_parties=[15, 20, 25, 30, 100, 1000, 5000, 7000],
              funct=g,
              output_file="results/gini.txt",
              nb_test=7,
              bins=256,
              l=40,
              c=14,
              secret_shared=False,
              epsilons=[1]
              ) -> None:
    """
    MSE of Gini mean difference
    for the 4 different protocols Umpc, Ghazi, GhaziSM and Bell.
    Data is uniformly distributed over X = [0, 1].
    
    :param int n_parties: Number of parties
    :param funct: Kernel function to consider
    :param str output_file: Output file
    :param int nb_test: Number of tests
    :param int bins: Number of bins
    :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)
    :param bool secret_shared: Boolean value to specify if secret sharing is happening or not
    :param List epsilons: DP parameters to consider
    """

    r = [0.0, 1.0]

    with open(output_file, "a") as f:
        f.write("epsilon\tn\to20\tbell\tghazi\tghazi_shuffle\n")
        for epsilon in epsilons:
            for n in n_parties:
                data = uniform_data(n, r)
                for _ in range(nb_test):

                    print(f"epsilon: {epsilon}, Nb of bins: {bins}, n: {n}")
                    real = compute_real_U(data, n, funct)
                    bell = bell_method(n, bins, data, r, epsilon, funct)
                    gh, L, R, m, ll = ghazi(n, epsilon, data, r, bins, funct)
                    gh_shuffle = ghazi_shuffle(n, epsilon, data, bins, L, R, m, ll)
                    our20 = ours_balanced(n, 2*n, data, epsilon, funct, l, c, secret_shared)

                    f.write(f"{epsilon}\t{n}\t{(real - our20)**2}\t{(real - bell)**2}\t{(real - gh)**2}\t{(real - gh_shuffle)**2}\n")

def MSE_kendall(data,
                r=[0.0, 1.0],
                epsilons=[0.1, 1],
                bins=[2**2, 2**3, 2**4, 2**5, 2**6],
                l=40,
                c=14,
                nb_test=5,
                output_file="results/kendall.txt"
                  ) -> None:
    """
    Docstring for MSE_kendall
    
    :param np.array data: Input data of shape (n, 2)
    :param Tuple r: (min, max) of \mathbb{X}
    :param List epsilons: List of epsilons to consider
    :param int bins: Number of bins
        :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)
    :param int nb_test: Number of tests to perform
    :param str output_file: Output file
    """
    n = len(data)

    def sign(x):
        if x < 0:
            return -1
        if x == 0:
            return 0
        return 1

    def h(a, b):
        (x, y) = a
        (x_, y_) = b
        return sign(x - x_) * sign(y - y_)

    funct = h

    data = np.reshape(data, (n, 2))
    with open(output_file, "a") as f:
        f.write("epsilon\tn\tbins\to20\tbell\tghazi\tghazi_shuffle\n")
        for epsilon in epsilons:
            for t in bins:
                for _ in range(nb_test):
                    print(f"{epsilon}, {n}, {t**2}")
                    real = compute_real_U(data, n, funct)
                    bell = kendall_bell(n, t, data, r, epsilon, funct)
                    gh, L, R, m, ll = kendall_ghazi(n, epsilon, data, r, t, funct)
                    gh_shuffle = kendall_ghazi_shuffle(n, epsilon, data, t, L, R, m, ll)
                    our_ = ours_balanced(n, 2*n, data, epsilon, funct, l, c, secret_shared=False)

                    f.write(f"{epsilon}\t{n}\t{t**2}\t{(real - our_)**2}\t{(real - bell)**2}\t{(real - gh)**2}\t{(real - gh_shuffle)**2}\n")


def compare_sampling_main(data,
                          funct=h,
                          epsilons=[1],
                          ratios=[0.1, 0.2, 0.5, 0.75, 1],
                          l=40,
                          c=14,
                          nb_test = 5,
                          output_file="results/duplicate.txt") -> None:
    """
    Compare the different sampling methods for Umpc
    
    :param np.array data: Input data of size n
    :param funct: Kernel function to consider
    :param List epsilons: List of epsilons to consider
    :param List[float] ratios: 
    :param int l: Parameter \ell in the paper (Total number 
        of bits to represent one element in fixed-point representation)
    :param int c: Parameter c in the paper (Number of bits for
        fractional part)
    :param int nb_test: Number of tests to perform
    :param str output_file: Output file
    """

    n = len(data)
    comb_n = (n * (n - 1)) // 2 
    data = np.reshape(data, n)

    with open(output_file, "a") as f:
        f.write("ratio\tn\tbal1\tbal5\two1\two5\tbern1\tbern5\n")
        for ratio in ratios:
            for _ in range(nb_test):

                size_E = int(comb_n * ratio)

                print(f"ratio: {ratio}, n: {n}, size_E: {size_E}")
                real = compute_real_U(data, n, funct)

                f.write(f"{ratio}\t{n}\t")

                for eps in epsilons:
                    bal = ours_balanced(n, size_E, data, eps, funct)
                    f.write(f"{(real - bal)**2}\t")
                    
                for eps in epsilons:
                    wo = ours_wo_repl(n, size_E, data, eps, funct)
                    f.write(f"{(real - wo)**2}\t")
                    
                for eps in epsilons:
                    bern = ours_bernoulli(n, ratio, data, eps, funct)
                    f.write(f"{(real - bern)**2}\t")
                
                f.write("\n")



if __name__ == "__main__":
    # compare_sampling_main(data=uniform_data(1000))
    gini_main()
    # MSE_kendall(data=uniform_data((100, 2)))