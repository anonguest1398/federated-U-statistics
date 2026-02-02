import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import List

from costs.communication import Umpc_comm, Bell_comm, Ghazi_comm, GhaziSM_comm
from costs.per_party_computation import Umpc_comp, Bell_comp, Ghazi_comp, GhaziSM_comp
from costs.server_computation import Umpc_serv_comp, Bell_serv_comp, Ghazi_serv_comp, GhaziSM_serv_comp

def communication_mse_per_party(
        infile="results/gini.txt",
        bins=256,
        eps=1,
        nbits=40,
        p=2
        ) -> None:
    """
    Figure 1 of the paper
    
    :param str infile: File containing data of MSE
    :param int bins: Number of bins for Ghazi and Bell
    :param float eps: DP parameter
    :param int nbits: Number of bits to represent one element
    :param int p: Number of neighbors per party for the protocol Umpc
    """

    df = pd.read_csv(infile, sep='\t')
    schemes = {"o_balanced": ("Umpc", "orange", "-o"), 
               "bell": ("Bell", "blue", "-^"), 
               "ghazi": ("Ghazi", "green", "-s"), 
               "ghazi_shuffle":("GhaziSM", "red", "-x")}
    
    n_parties = df["n"].unique()

    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'$\epsilon = {eps}, t={bins}$')

    ax1.set_yscale("log")
    ax1.set_xscale("log")

    ax1.set_title(r"Total comm. cost")
    ax1.plot(n_parties, [Umpc_comm(n, p, nbits, kernel="gini") for n in n_parties], '-o', label=r"$\mathsf{Umpc}$", color="tab:orange")
    ax1.plot(n_parties, [Bell_comm(n, bins, nbits) for n in n_parties], '-^', label=r"$\mathsf{Bell}$", color="tab:blue")
    ax1.plot(n_parties, [Ghazi_comm(n, bins, nbits, eps)  for n in n_parties], '-s', label=r"$\mathsf{Ghazi}$", color="tab:green")
    ax1.plot(n_parties, [GhaziSM_comm(n, bins, nbits, eps) for n in n_parties], '-x', label=r"$\mathsf{GhaziSM}$", color="tab:red")

    ax1.set_ylabel("Communication cost (bits)")
    ax1.set_xlabel("Number of parties")

    ax1.grid()

    dic = {}
    for s in schemes:
        dic[s] = []

    for (_, bins), d in df.groupby(["n", "bins"]):
        for s in schemes:
            if d[s].any() != math.inf:
                dic[s].append(d[s].mean())

    for s, tupl in schemes.items():
        ax2.plot(n_parties, dic[s], tupl[2], label=f"$\mathsf{{tupl[0]}}$", color=f"tab:{tupl[1]}")

    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_title(r"MSE")
    ax2.set_xlabel("Number of parties")
    ax2.set_ylabel("MSE")
    ax2.grid()

    handles, labels = ax1.get_legend_handles_labels()

    # Put one shared legend below both plots
    fig.legend(handles, labels, loc="lower center", ncol=5, fancybox=True, shadow=True,)

    # Adjust layout so legend fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom

    plt.show()

def communication_computation(
        n=4521,
        t=[2**4, 2**8, 2**12, 2**16, 2**24, 2**28, 2**32],
        nbits=40,
        p=2
) -> None:
    """
    Figure 3 of the paper
    
    :param int n: Number of parties
    :param List t: List of the number of bins
    :param int nbits: Number of bits to represent one element
    :param int p: Number of neighbors per party for the protocol Umpc
    """
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'$n = {n}$')

    ax1.set_xscale("log")
    ax1.set_yscale("log") 

    #ax1.set_title(f"Online communication cost")
    ax1.set_title(f"Total communication cost")

    ax1.plot(t, [Umpc_comm(n, p, nbits, kernel="kendall") for _ in t], '-o', label=r"$\mathsf{Umpc}$", color="tab:orange")
    ax1.plot(t, [Bell_comm(n, x, nbits) for x in t], '-^', label="$\mathsf{Bell}$")

    eps = 1
    ax1.plot(t, [Ghazi_comm(n, x, nbits, eps) for x in t], '-s', label=r"$\mathsf{Ghazi}, \epsilon=1$", color="tab:green")
    ax1.plot(t, [GhaziSM_comm(n, x, nbits, eps) for x in t], '-x', label=r"$\mathsf{GhaziSM}, \epsilon=1$", color="tab:red")

    eps = 0.1
    ax1.plot(t, [Ghazi_comm(n, x, nbits, eps) for x in t], '--s', label=r"$\mathsf{Ghazi}, \epsilon=0.1$", color="tab:green")
    ax1.plot(t, [GhaziSM_comm(n, x, nbits, eps)for x in t], '--x', label=r"$\mathsf{GhaziSM}, \epsilon=0.1$", color="tab:red")

    ax1.grid()
    ax1.set_ylabel("Total Communication cost (bits)")
    ax1.set_xlabel("Number of discretization bins $t$")

    ax2.set_xscale("log")
    ax2.set_yscale("log") 

    ax2.set_title(f"Total computation cost")

    eps = 1
    ax2.plot(t, [n * Ghazi_comp(n, x, eps) for x in t], '-s', label=r"$\mathsf{Ghazi}, \epsilon=1$", color="tab:green")
    ax2.plot(t, [n * GhaziSM_comp(n, x, eps) for x in t], '-x', label=r"$\mathsf{GhaziSM}, \epsilon=1$", color="tab:red")

    eps = 0.1
    ax2.plot(t, [n * Umpc_comp(n, p, nbits, kernel="kendall") for _ in t], '-o', label=r"$\mathsf{Umpc}$ ($\alpha \approx 0.02$)", color="tab:orange")
    ax2.plot(t, [n * Bell_comp(x) for x in t], '-^', label="$\mathsf{Bell}$")
    ax2.plot(t, [n * Ghazi_comp(n, x, eps) for x in t], '--s', label=r"$\mathsf{Ghazi}, \epsilon=0.1$", color="tab:green")
    ax2.plot(t, [n * GhaziSM_comp(n, x, eps) for x in t], '--x', label=r"$\mathsf{GhaziSM}, \epsilon=0.1$", color="tab:red")

    ax2.grid()
    ax2.set_ylabel("Computation cost (# operations)")
    ax2.set_xlabel("Number of discretization bins $t$")

    # Get handles & labels from one axis
    handles, labels = ax1.get_legend_handles_labels()

    # Put one shared legend below both plots
    fig.legend(handles, labels, loc="lower center", ncol=4, fancybox=True, shadow=True,)

    # Adjust layout so legend fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom
    plt.show()

def communication_server_computation(
        n_parties = [30, 100, 1000, 5000, 7000],
        bins = 256,
        eps=1,
        nbits=40,
        p=2
        ) -> None:
    """
    Figure 2 of the paper

    :param List n_parties: List of numbers of parties
    :param int bins: Number of bins for Ghazi and Bell
    :param float eps: DP parameter
    :param int nbits: Number of bits to represent one element
    :param int p: Number of neighbors per party in protocol Umpc
    """
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'$\epsilon = {eps}, t={bins}$')

    ax2.set_yscale("log") 

    ax2.set_title(f"Server computation cost")

    ax2.plot(n_parties, [Umpc_serv_comp(n) for n in n_parties], '-o', label=r"$\mathsf{Umpc}$", color="tab:orange")
    ax2.plot(n_parties, [Bell_serv_comp(n, bins) for n in n_parties], '-^', label=r"$\mathsf{Bell}$")
    ax2.plot(n_parties, [Ghazi_serv_comp(n, bins, eps) for n in n_parties], '-s', label=r"$\mathsf{Ghazi}$", color="tab:green")
    ax2.plot(n_parties, [GhaziSM_serv_comp(n, bins, eps) for n in n_parties], '-x', label=r"$\mathsf{GhaziSM}$", color="tab:red")

    ax2.grid()
    ax2.set_ylabel("Computation cost (# operations)")
    ax2.set_xlabel(r"Number of parties $n$")
    ax1.set_yscale("log") 
    ax1.set_title(f"Computation cost per party")

    ax1.plot(n_parties, [Umpc_comp(n, p, nbits, kernel="gini") for n in n_parties], '-o', label=r"$\mathsf{Umpc}$", color="tab:orange")
    ax1.plot(n_parties, [Bell_comp(bins) for _ in n_parties], '-^', label=r"$\mathsf{Bell}$")

    ax1.plot(n_parties, [Ghazi_comp(n, bins, eps) for n in n_parties], '-s', label=r"$\mathsf{Ghazi}$", color="tab:green")
    ax1.plot(n_parties, [GhaziSM_comp(n, bins, eps) for n in n_parties], '-x', label=r"$\mathsf{GhaziSM}$", color="tab:red")

    ax1.grid()
    ax1.set_ylabel("Computation cost (# operations)")
    ax1.set_xlabel(r"Number of parties $n$")
    
    # Get handles & labels from one axis
    handles, labels = ax1.get_legend_handles_labels()

    # Put one shared legend below both plots
    fig.legend(handles, labels, loc="lower center", ncol=5, fancybox=True, shadow=True,)

    # Adjust layout so legend fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom
    plt.show()


def mse_mse_vs_comm(
        infile="results/kendall.txt",
        epsilons=[0.1, 1.0],
        nbits=40,
        p=2
        ) -> None:
    """
    Figure 5 of the paper
    
    :param str infile: File containing data of MSE
    :param List epsilons: List of DP parameters
    :param int nbits: Number of bits to represent one element
    :param int p: Number of neighbors per party in Umpc
    """

    plt.rcParams.update({'font.size': 13})
    fig, (ax1, ax2) = plt.subplots(1, 2)

    schemes = {"o20_1.0": ("Umpc, $\epsilon=1$", "orange", "o"), 
               "bell_1.0": ("Bell, $\epsilon=1$", "blue", "^"), 
               "ghazi_1.0": ("Ghazi, $\epsilon=1$", "green", "s"), 
               "ghazi_shuffle_1.0":("GhaziSM, $\epsilon=1$", "red", "x"),
               "o20_0.1": ("Umpc, $\epsilon=0.1$", "orange", "-o"), 
               "bell_0.1": ("Bell, $\epsilon=0.1$", "blue", "-^"), 
               "ghazi_0.1": ("Ghazi, $\epsilon=0.1$", "green", "-s"), 
               "ghazi_shuffle_0.1":("GhaziSM, $\epsilon=0.1$", "red", "-x"),
               }
    
    cols = ["o20", "bell", "ghazi", "ghazi_shuffle"]
    epsilons=[0.1, 1.0]


    dic={}
    comm={}
    for s in cols:
        for eps in epsilons:
            name = s + "_" + str(eps)
            dic[name] = []
            comm[name] = []
    
    df = pd.read_csv(infile, sep='\t')
    ts = df["bins"].unique()
    n = df["n"].unique()[0]

    fig.suptitle(r'$n = 4521$')

    for (eps, bins), d in df.groupby(["epsilon", "bins"]):
        for s in cols:
            name = s + "_" + str(eps)
            dic[name].append(d[s].mean())
            if s == "o20":
                m = Umpc_comm(n, p, nbits, kernel="kendall")
            elif s == "bell":
                m = Bell_comm(n, bins, nbits)
            elif s == "ghazi":
                m = Ghazi_comm(n, bins, nbits, eps)
            elif s == "ghazi_shuffle":
                m = GhaziSM_comm(n, bins, nbits, eps)
            comm[name].append(m)

    for s, tupl in schemes.items():
        ax1.plot(ts, dic[s], f"-{tupl[2]}", label=f"{tupl[0]}", color=f"tab:{tupl[1]}")
        ax2.plot(comm[s], dic[s], f"-{tupl[2]}", label=f"{tupl[0]}", color=f"tab:{tupl[1]}")

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.grid()
    #ax1.set_title(f"MSE comparison for $\epsilon = 0.1$ and n={n}")
    ax1.set_xlabel("Number of discretization bins")
    ax1.set_ylabel("MSE")
    
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.grid()
    ax2.set_xlabel("Communication cost (bits)")
    ax2.set_ylabel("MSE")

    handles, labels = ax2.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # Put one shared legend below both plots
    fig.legend(handles, labels, loc="lower center", ncol=5, fancybox=True, shadow=True, prop={'size': 10})

    # Adjust layout so legend fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom

    plt.show()


def four_plots(
        infile="results/duplicate_appendix.txt",
        ratios = [0.05, 0.1, 0.25, 0.5, 0.75, 1],
        epsilon=1,
        nbits=40,
        n= 834,
        p=2,
        bins=64,
        kernel="duplicate",
):
    """
    Plots in the appendix of the paper
    
    :param str infile: File containing data of MSE
    :param List ratios: Ratios considered in infile
    :param float epsilon: DP parameter considered in infile
    :param int nbits: Number of bits to represent one element
    :param int n: Number of parties considered in the experiment of infile
    :param int p: Number of neighbors per party
    :param int bins: Number of bins
    :param str kernel: Name of the kernel function considered
    """
    
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(2, 2, figsize=(12, 11))


    n_parties = [int(n * r) for r in ratios]

    fig.suptitle(f'$n = {n}$, $\epsilon = {epsilon}$, $t={bins}$')
    
    axs[0, 0].set_yscale("log")

    axs[0, 0].set_title(r"Total comm. cost")
    
    axs[0, 0].plot(n_parties, [Umpc_comm(n, p, nbits, kernel=kernel) for n in n_parties], '-o', label=r"$\mathsf{Umpc}$", color="tab:orange")
    axs[0, 0].plot(n_parties, [Bell_comm(n, bins, nbits) for n in n_parties], '-^', label="$\mathsf{Bell}$")
    axs[0, 0].plot(n_parties, [Ghazi_comm(n, bins, nbits, epsilon) for n in n_parties], '-s', label=r"$\mathsf{Ghazi}$", color="tab:green")
    axs[0, 0].plot(n_parties, [GhaziSM_comm(n, bins, nbits, epsilon) for n in n_parties], '-x', label=r"$\mathsf{GhaziSM}$", color="tab:red")

    axs[0, 0].set_ylabel("Communication cost (bits)")
    axs[0, 0].set_xlabel("Number of parties")

    axs[0, 0].grid()
    
    df = pd.read_csv(infile, sep='\t')
    schemes = ["o20", "bell", "ghazi", "ghazi_shuffle"]
    n_parties = df["n"].unique()
    dic = {}
    comm = {}
    for s in schemes:
        dic[s] = []
        comm[s] = []

    for (n, bins), d in df.groupby(["n", "bins"]):
        for s in schemes:
            if d[s].any() != math.inf:
                dic[s].append(d[s].mean())
            if s == "o20":
                m = Umpc_comm(n, p, nbits, kernel=kernel)
            elif s == "bell":
                m = Bell_comm(n, bins, nbits)
            elif s == "ghazi":
                m = Ghazi_comm(n, bins, nbits, epsilon)
            elif s == "ghazi_shuffle":
                m = GhaziSM_comm(n, bins, nbits, epsilon)
            comm[s].append(m)

    for s in schemes:
        if s == "o20":
            axs[0, 1].plot(comm[s], dic[s], '-o', label=r"$\mathsf{Umpc}$ ($\alpha \approx 0.02$)", color="tab:orange")
        elif s =="ghazi" :
            axs[0, 1].plot(comm[s], dic[s], '-s', label=r"$\mathsf{Ghazi}$", color="tab:green")
        elif s == "ghazi_shuffle":
            axs[0, 1].plot(comm[s], dic[s], '-x', label=r"$\mathsf{GhaziSM}$ ($r = 20$)", color="tab:red")
        elif s == "bell":
            axs[0, 1].plot(comm[s], dic[s], '-^', label=r"$\mathsf{Bell}$", color="tab:blue")

    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_title(r"MSE")
    axs[0, 1].set_xlabel("Communication cost (bits)")
    axs[0, 1].set_ylabel("MSE")
    axs[0, 1].grid()


    # party computation

    axs[1, 0].set_title("Party computation")

    axs[1, 0].plot(n_parties, [Umpc_comp(n, p, nbits, kernel=kernel) for n in n_parties], '-o', label=r"$\mathsf{Umpc}$ ($\alpha \approx 0.02$)", color="tab:orange")
    axs[1, 0].plot(n_parties, [Bell_comp(bins) for n in n_parties], '-^', label="$\mathsf{Bell}$")
    axs[1, 0].plot(n_parties, [Ghazi_comp(n, bins, epsilon)  for n in n_parties], '-s', label=r"$\mathsf{Ghazi}, \epsilon=1$", color="tab:green")
    axs[1, 0].plot(n_parties, [GhaziSM_comp(n, bins, epsilon) for n in n_parties], '-x', label=r"$\mathsf{GhaziSM}, \epsilon=1$", color="tab:red")
    

    axs[1, 0].grid()
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("Number of parties")
    axs[1, 0].set_ylabel("Number of operations")

    # Server side
    axs[1,1].set_title("Server computation")
    axs[1, 1].plot(n_parties, [Umpc_serv_comp(n) for n in n_parties], "-o", color="tab:orange", label=r"$\mathsf{Umpc}$ ($\alpha \approx 0.02$)")
    axs[1, 1].plot(n_parties, [Bell_serv_comp(n, bins) for n in n_parties], "-^", label="$\mathsf{Bell}$")
    axs[1, 1].plot(n_parties, [Ghazi_serv_comp(n, bins, epsilon) for n in n_parties], "-s", label="$\mathsf{Ghazi}$", color="tab:green")
    axs[1, 1].plot(n_parties, [GhaziSM_serv_comp(n, bins, epsilon) for n in n_parties], "-x", label="$\mathsf{GhaziSM}$", color="tab:red")

    axs[1, 1].set_yscale("log")
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("Number of parties")
    axs[1, 1].set_ylabel("Number of operations")

    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Put one shared legend below both plots
    fig.legend(handles, labels, loc="lower center", ncol=5, fancybox=True, shadow=True,)

    # Adjust layout so legend fits
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom

    plt.show()



if __name__ == "__main__":
    #four_plots(infile="results/duplicate_appendix.txt", kernel="duplicate")
    communication_mse_per_party()