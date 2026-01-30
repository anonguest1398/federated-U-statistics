from math import comb
import argparse
import pandas as pd
import termtables as tt

from protocols.our import ours_balanced
from protocols.bell import bell_method
from protocols.ghazi import ghazi, ghazi_shuffle
from protocols.real import compute_real_U

from costs.communication import Umpc_comm, Ghazi_comm, GhaziSM_comm, Bell_comm
from costs.per_party_computation import Umpc_comp, Ghazi_comp, GhaziSM_comp, Bell_comp
from costs.server_computation import Umpc_serv_comp, Ghazi_serv_comp, GhaziSM_serv_comp, Bell_serv_comp

from experiments.load_data import uniform_data

def main(
        n=100,
        epsilon=1.0,
        bins=128,
        p=5,
        nbits=20,
        c=8,
        dataset=None,
        r=[0.0, 1.0],
        nb_tests=5,
        kernel="gini"
):
    
    if dataset == None:
        data = uniform_data(n, r=[0.0, 1.0])
    else:
        data = pd.read_csv(dataset)

    if kernel == "gini":
        funct = lambda x, y: abs(x - y)
    elif kernel == "duplicate":
        funct = lambda x, y: x == y

    umpc, g, g_sm, bell = [], [], [], []

    for _ in range(nb_tests):
        real = compute_real_U(data, n, funct)
        b = bell_method(n, bins, data, r, epsilon, funct)
        gh, L, R, m, l = ghazi(n, epsilon, data, r, bins, funct)
        gh_shuffle = ghazi_shuffle(n, epsilon, data, bins, L, R, m, l)
        our20 = ours_balanced(n, n*p, data, epsilon, funct, l=nbits, c=c)

        mse_umpc = (real - our20)**2
        mse_ghazi = (real - gh)**2
        mse_ghazi_sm = (real - gh_shuffle)**2
        mse_bell = (real - b)**2

        umpc.append(mse_umpc)
        g.append(mse_ghazi)
        g_sm.append(mse_ghazi_sm)
        bell.append(mse_bell)

    
    header = ["Protocol", "MSE", "Communication", "Party computation", "Server computation"]

    sum_umpc = ["Umpc", sum(umpc) / nb_tests, Umpc_comm(n, p, nbits, kernel), 
         Umpc_comp(n, p, nbits, kernel), Umpc_serv_comp(n)]
    sum_ghazi = ["Ghazi",  sum(g) / nb_tests, Ghazi_comm(n, bins, nbits, epsilon), 
         Ghazi_comp(n, bins, epsilon), Ghazi_serv_comp(n, bins, epsilon)]
    sum_ghazism = ["GhaziSM",  sum(g_sm) / nb_tests, GhaziSM_comm(n, bins, nbits, epsilon), 
         GhaziSM_comp(n, bins, epsilon), GhaziSM_serv_comp(n, bins, epsilon)]
    sum_bell = ["Bell", sum(bell) / nb_tests,Bell_comm(n, bins, nbits),
                Bell_comp(bins), Bell_serv_comp(n, bins)]

    string = tt.to_string(
        [sum_umpc, sum_ghazi, sum_ghazism, sum_bell],
        header=header,
        style=tt.styles.ascii_thin_double
    )

    print(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run small preview of the different protocols"
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=100,
        help="Number of parties.",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1.0,
        help="Epsilon for differential privacy.",
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=128,
        help="Number of bins for the protocols Ghazi and Bell.",
    )
    parser.add_argument(
        "-p",
        "--p",
        type=int,
        default=5,
        help="Number of neighbors per party for the protocol Umpc.",
    )
    parser.add_argument(
        "-nbits",
        "--nbits",
        type=int,
        default=20,
        help="Number of bits to represent one element of X.",
    )
    parser.add_argument(
        "-c",
        "--c",
        type=int,
        default=8,
        help="Number of bits to represent the fractional part.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Path to the dataset CSV file. If None, random data will be created.",
    )
    parser.add_argument(
        "-ntests",
        "--nbtests",
        type=int,
        default=5,
        help="Number of replays.",
    )
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        default="gini",
        help="Name of the kernel function.",
    )

    args = parser.parse_args()

    main(
        n=args.n,
        epsilon=args.epsilon,
        bins=args.bins,
        p=args.p,
        nbits=args.nbits,
        c=args.c,
        dataset=args.dataset,
        nb_tests=args.nbtests,
        kernel=args.function
    )