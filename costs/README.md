# Costs

We made the choice to measure the communication cost by counting the total number of bits exchanged rather than measuring execution time. This choice avoids dependencies on hardware, network latency, and implementation details, and enables a fair and reproducible comparison of the protocols considered in the paper. You can find the estimation and explanations for the costs in `communication.py`.

For per-party computation, please see `per_party_computation.py`.

For server computation, please see `server_computation.py`.