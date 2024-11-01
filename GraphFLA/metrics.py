from .distances import mixed_distance
from .algorithms import hill_climb, random_walk
from scipy.stats import spearmanr, pearsonr
from typing import Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import random

def FDC(
        landscape, 
        distance = mixed_distance,
        method: str = "spearman",
    ) -> float:
    """
    Calculate the fitness distance correlation of a landscape. It assesses how likely is it 
    to encounter higher fitness values when moving closer to the global optimum.

    It will add an attribute `fdc` to the landscape object, and also create a "dist_go"
    column to both `data` and `data_lo`.

    The distance measure here is based on a combination of Hamming and Manhattan distances,
    to allow for mixed-type variables. See `Landscape._mixed_distance`.

    Parameters
    ----------
    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure used to assess FDC.

    Returne
    -------
    fdc : float
        An FDC value ranging from -1 to 1. A value close to 1 indicates positive correlation
        between the fitness values of a configuration and its distance to the global optimum.
    """

    configs = np.array(landscape.data['config'].tolist())
    go_config = np.array(landscape.go["config"])
    _data_types = {i:landscape.data_types[var] for i, var in enumerate(landscape.data_types.keys())}
    distances = distance(configs, go_config, _data_types)

    landscape.data["dist_go"] = distances
    if method == "spearman":
        landscape.fdc = spearmanr(distances, landscape.data["fitness"].rank())
    elif method == "pearson":
        landscape.fdc = pearsonr(distances, landscape.data["fitness"].rank())        
    else:
        raise ValueError(f"Invalid method {method}. Please choose either 'spearman' or 'pearson'.")    
    landscape.data["dist_go"] = distances
    landscape.data_lo = landscape.data.loc[landscape.data_lo.index]

    return landscape.fdc

def FFI(
        landscape, 
        frac: float = 1, 
        min_len: int = 3, 
        method: str = "spearman"
    ) -> float:
    """
    Calculate the fitness flatenning index (FFI) of the landscape. It assesses whether the 
    landscape tends to be flatter around the global optimum. It operates by identifying
    (part of, controled by `frac`) adaptive paths leading to the global optimum, and 
    checks whether the fitness gain in each step decreases as approaching the global peak. 

    Parameters
    ----------
    frac : float, default=1
        The fraction of adapative paths to be assessed. 

    min_len : int, default=3
        Minimum length of an adaptive path for it to be considered in evaluation. 
    
    method : str, one of {"spearman", "pearson"}, default="spearman"
        The correlation measure used to assess FDC.

    Returns
    -------
    ffi : float
        An FFI value ranging from -1 to 1. A value close to 1 indicates that the landscape
        is very likely to be flatter around the global optimum. 
    """

    def check_diminishing_differences(data, method):
        data.index = range(len(data))
        differences = data.diff().dropna()
        index = np.arange(len(differences))
        if method == "pearson":
            correlation, _ = pearsonr(index, differences)
        if method == "spearman":
            correlation, _ = spearmanr(index, differences)
        else:
            raise ValueError("Invalid method. Please choose either 'spearman' or 'pearson'.") 
        return correlation
    
    node_attributes = dict(landscape.graph.nodes(data=True))
    fitness = pd.DataFrame.from_dict(node_attributes, orient='index')["fitness"]
    if landscape.maximize:
        go_id = fitness.idxmax()
    else:
        go_id = fitness.idxmin()
    ffi_list = []
    total = int(nx.number_of_nodes(landscape.graph) * frac)
    for i in range(total):
        lo, steps, trace = hill_climb(landscape.graph, i, "delta_fit", verbose=0, return_trace=True)
        if len(trace) >= min_len:
            if lo == go_id:
                fitnesses = fitness.loc[trace]
                ffi = check_diminishing_differences(fitnesses, method)
                ffi_list.append(ffi)

    landscape.ffi = pd.Series(ffi_list).mean()
    return landscape.ffi

def autocorrelation(
        landscape,
        walk_length: int = 20, 
        walk_times: int = 1000,
        lag: int = 1
    ) -> Tuple[float, float]:
    """
    A measure of landscape ruggedness. It operates by calculating the autocorrelation of 
    fitness values over multiple random walks on a graph.

    Parameters:
    ----------
    walk_length : int, default=20
        The length of each random walk.

    walk_times : int, default=1000
        The number of random walks to perform.

    lag : int, default=1
        The distance lag used for calculating autocorrelation. See pandas.Series.autocorr.

    Returns:
    -------
    autocorr : Tuple[float, float]
        A tuple containing the mean and variance of the autocorrelation values.
    """

    corr_list = []
    nodes = list(landscape.graph.nodes())
    for _ in range(walk_times):
        random_node = random.choice(nodes)
        logger = random_walk(landscape.graph, random_node, "fitness", walk_length)
        autocorrelation = pd.Series(logger["fitness"]).autocorr(lag=lag)
        corr_list.append(autocorrelation)

    landscape.autocorr = pd.Series(corr_list).median()

    return landscape.autocorr, pd.Series(corr_list).var()