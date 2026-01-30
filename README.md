
# Accurate, private, secure, federated U-statistics with higher degree

To reproduce the experiments from the paper, please see `experiments/`.

To get the estimation of communication, per-party computation and server computation costs, please see `costs/`.

For the implementation of the protocols, please see `protocols/`.

## Install requirements

Run the following
```bash
# python3.10
pip install -r requirements.txt
```

## Preview

A preview of the experiments can be obtain by running the `preview.py` script.
Most of the arguments can be set using the command line, here is the --help of the script:

```bash
python3 preview.py --help
usage: preview.py [-h] [-n N] [-e EPSILON] [-b BINS] [-a ALPHA] [-nbits NBITS] [-d DATASET] [-ntests NBTESTS] [-f FUNCTION]

Run small preview of the different protocols

options:
  -h, --help            show this help message and exit
  -n N, --n N           Number of parties.
  -e EPSILON, --epsilon EPSILON
                        Epsilon for differential privacy.
  -b BINS, --bins BINS  Number of bins for the protocols Ghazi and Bell.
  -a P, --p P
                        Number of neighbors per party for the protocol Umpc.
  -nbits NBITS, --nbits NBITS
                        Number of bits to represent one element of X.
  -c C, --c C
                        Number of bits to represent the fractional part.
  -d DATASET, --dataset DATASET
                        Path to the dataset CSV file. If not provided, random data will be created.
  -ntests NBTESTS, --nbtests NBTESTS
                        Number of replays.
  -f FUNCTION, --function FUNCTION
                        Name of the kernel function.
```

All arguments are optional and default to:
```bash
python3 preview.py -n 100 -e 1.0 -b 128 -p 5 -nbits 20 -c 8 -ntests 5 -f "gini"
```

## Launch individual Python function

To launch any functions from `experiments/`, `costs/` or `protocols/`, modify the main function of the file containing the function.
Then, execute the Python program. 
For example, to draw the Figure 3 from the paper, modify the following:
```python

if __name__ == "__main__":
  communication_mse_per_party()

```
Then, open a terminal in `clean_code/` and execute the following: 
```bash
python3 -m experiments.plot
```

Another option would be to create a Python file at the root `clean_ustats/` and importing the desired modules.