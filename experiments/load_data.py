import numpy as np
import pandas as pd
from typing import Tuple

def uniform_data(shape:Tuple, r=(0.0, 1.0)) -> np.array:
    """
    Returns a matrix of size shape uniformly distributed
    
    :param int shape: Description
    :param Tuple r: Description

    :return: np.array result: Uniform data of size n
    """

    return np.random.uniform(r[0], r[1], size=shape)
    
def merge_heart_disease() -> np.array:
    """
    Returns the columns (resting blood pressure, 
        serum cholestoral, 
        diagnosis of heart disease) from the Heart Disease Dataset

    :return: np.array normalized_df:
    """
    url1 = "heart+disease/processed.cleveland.data"
    url2 = "heart+disease/processed.va.data"
    url3 = "heart+disease/processed.hungarian.data"
    url4 = "heart+disease/processed.switzerland.data"
    df1 = pd.read_csv(url1, header=None)
    df2 = pd.read_csv(url2, header=None)
    df3 = pd.read_csv(url3, header=None)
    df4 = pd.read_csv(url4, header=None)

    df = pd.concat([df1, df2, df3, df4])

    cols = [3, 4, 13]

    df = df[cols]

    df = df.loc[df[3] != '?']
    df = df.loc[df[4] != '?']

    df[3] = df[3].astype("int64")
    df[4] = df[4].astype("int64")
    df[13] = df[13].astype("int64")

    normalized_df=(df-df.min())/(df.max()-df.min())

    return normalized_df

def get_heart_diseases() -> np.array:
    """
    Returns column (resting blood pressure, serum cholestoral)

    :return: np.array r:
    """

    df = merge_heart_disease()
    col = [3, 4]

    r = df[col].to_numpy()
    return r

def get_heart_diseases_target() -> np.array:
    """
    Returns column diagnosis of heart disease

    :return: np.array r:
    """
    df = merge_heart_disease()
    col = [13]
    
    r = df[col].to_numpy()
    return r


def select_random(data: np.array, nb_points: int) -> np.array:
    """
    Returns nb_points rows uniformly chosen from data
    
    :param np.array data: Input data
    :param int nb_points: Number of points to select

    :return: np.array result:
    """
    m = len(data)
    l = []
    r = np.random.choice(m, nb_points, False)
    for r_ in r:
        l.append(data[r_])

    return np.array(l)

def get_kendall_bank_data() -> np.array:
    """
    Returns columns age and balance from the Bank Dataset

    :return: np.array r:
    """

    url = "bank+marketing/bank/bank.csv"
    df = pd.read_csv(url, sep=";")
    
    col = ["age", "balance"]

    df = df[col]
    normalized_df=(df-df.min())/(df.max()-df.min())
    r = normalized_df[col].to_numpy()
    return r

def get_duplicate_bank_data() -> np.array:
    """
    Returns column job from the Bank dataset 

    :return: np.array r:
    """

    url = "bank+marketing/bank/bank.csv"
    df = pd.read_csv(url, sep=";")

    df = df[["job"]]
    df["job"] = pd.factorize(df["job"])[0]

    normalized_df=(df-df.min())/(df.max()-df.min())
    r = normalized_df[["job"]].to_numpy()

    return r

if __name__ == "__main__":
    pass