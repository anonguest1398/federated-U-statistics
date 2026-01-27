
# Protocols

List of protocols implemented:
- Protocol $\mathsf{Umpc}$ (Algorithm 3.1) in `ours.py`. We implemented our protocol using Additive secret sharing over the ring $\mathbb{Z}_{2^\ell}$. 
For ease of implementation, we did not implement the MPC protocols for computing the kernel functions,
- Protocol $\mathsf{Bell}$ from "Private protocols for u-statistics in the local model and beyond." by Bell et al. in `bell.py`,
- Protocol $\mathsf{Ghazi}$ from "On computing pairwise statistics with local differential privacy." by Ghazi et al. in `ghazi`,
- Protocol $\mathsf{GhaziSM}$ from "On computing pairwise statistics with local differential privacy." by Ghazi et al. under the Shuffle Model implemented via "Private summation in the multi-message shuffle model." by Balle et al. in `ghazi`.

Because Kendall's tau requires data points of dimension 2, i.e., $x_i = (y_i, z_i)$, the implementations changes. 
You can find the adapted protocols in `kendall/`.