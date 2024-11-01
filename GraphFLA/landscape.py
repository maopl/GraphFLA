import pandas as pd
import networkx as nx
import numpy as np 
import copy 
import matplotlib.pyplot as plt
import umap.umap_ as umap
import concurrent.futures

from karateclub import HOPE
from palettable.lightbartlein.diverging import BlueOrangeRed_3
from typing import List, Any, Dict, Tuple, Set, Union, Optional
from itertools import product, combinations
from functools import lru_cache
from collections import defaultdict
from tqdm import tqdm
from .algorithms import hill_climb, random_walk
from .utils import get_mlon, trim_lon, add_network_metrics
from .distances import mixed_distance
from .metrics import FDC, FFI, autocorrelation
from .visualization import draw_neighborhood, draw_landscape_2d, draw_landscape_3d

import warnings
warnings.filterwarnings('ignore')

class Landscape():
    """
    Class implementing the fitness landscape object

    Parameters
    ----------
    X : pd.DataFrame or np.array
        The data containing the configurations to construct the landscape. Note 
        that for numerical variables, we currently favor data produced by grid search. Using
        random-sampled data points may lead to unexpected issues. For categorical variables,
        values in either strings or numbers are acceptable. 

    f : pd.Series or list or np.array
        The fitness values associated with the configurations. 

    graph : nx.DiGraph
        Initialize the landscape with precomputed data as networkx directed graph. 

    maximize : bool
        Indicates whether the fitness is to be maximized. 

    data_types : dictionary
        A dictionary specifying the data type of each variable in X. Each variable can 
        be {"boolean", "categorical", "ordinal"}. If

        - X is pd.DataFrame, then the keys of data_types should match with columns of X.
        - X is np.array, the keys of data_types can be in arbitrary format, but the order 
          of the keys should be the same as in X. 

    neutrality : bool, default=False
        Whether to allow neutrality in the landscape. Note that the framework is currently
        deisgned on non-neutral landscapes, and allowing neutrality may lead to unexpected
        bahaviors. 

    verbose : bool
        Controls the verbosity of output.

    configs : list of tuples
        A list of tuples representing the configurations in the landscape.
    
    lo_index : list of int
        A list of indices specifying the indices of local optima in the landscape.

    Attributes
    ----------
    data : pd.DataFrame
        A pandas dataframe containing all tabular information regarding each configuration.

    data_lo : pd.DataFrame
        A pandas dataframe containing all tabular information regarding local optima 
        configurations in the landscape. 

    graph : nx.DiGraph
        A networkx directed graph representing the landscape. Fitness values and other 
        calculated information are available as node attributes. Fitness differences between
        each pair of nodes (configurations) are stored as edge weights 'delta_fit'. The 
        direction of the edge always points to fitter configurations. 

    n_configs : int
        Number of total configurations in the constructed landscape.

    n_vars : int
        Number of variables in the constructed landscape.

    n_edges : int
        Number of total connections in the constructed landscape.

    n_lo : int
        Number of local optima in the constructed landscape.

    Examples
    --------
    Below is an example of how to create a `Landscape` object using a dataset of hyperparameter 
    configurations and their corresponding test accuracy.

    ```python

    # Define the data types for each hyperparameter
    data_types = {
        "learning_rate": "ordinal",
        "max_bin": "ordinal",
        "max_depth": "ordinal",
        "n_estimators": "ordinal",
        "subsample": "ordinal",
    }

    >>> df = pd.read_csv("hpo_xgb.csv", index_col=0)

    >>> X = df.iloc[:, :5]  # Assuming the first five columns are the configuration parameters
    >>> f = df["acc_test"]  # Assuming 'acc_test' is the column for test accuracy

    # Create a Landscape object
    >>> landscape = Landscape(X, f, maximize=True, data_types=data_types)

    # General information regarding the landscape
    >>> landscape.describe()
    ```
    """
    # @profile
    def __init__(
            self, 
            X: pd.DataFrame = None, 
            f: pd.Series = None,
            graph: nx.DiGraph = None,
            maximize: bool = True, 
            data_types: Dict[str, str] = None,
            neutrality: bool = False,
            verbose: bool = True
        ) -> None:

        self.maximize = maximize
        self.verbose = verbose
        self.neutrality = neutrality
        self.data_lo = None
        self.has_lon = False
        if graph is None:
            if self.verbose:
                print("Creating landscape from scratch with X and f...")
            assert X is not None and f is not None, "X and f cannot be None if graph is not provided."
            assert len(X) == len(f), "X and f must have the same length."
            assert data_types is not None, ("data_types cannot be None if graph is not provided.") 
            self.n_configs = X.shape[0]
            self.n_vars = X.shape[1]
            self.data_types = data_types
            data, self.config_dict_ = self._validate_and_prepare_data(X, f, data_types)
            del X, f
            edge_list = self._parallel_create_edges(data, n_edit=1)
            self.graph = self._construct_landscape(data, edge_list)
            self.graph = self._add_network_metrics(self.graph, weight="delta_fit")
            self._determine_local_optima()
            self._calculate_basin_of_attraction()
            self._determine_global_optimum()
        else:
            if self.verbose:
                print("Loading landscape from precomputed graph")
            self.graph = graph
        if self.verbose:
            print("Landscape constructed!\n")
        
    def _validate_and_prepare_data(
            self, 
            X: pd.DataFrame, 
            f: pd.Series, 
            data_types: Dict[str, str]
        ) -> Tuple[pd.DataFrame, dict]:
        """Preprocess the input data and generate domain dictionary for X"""
        if self.verbose:
            print("# Preparing data...")
        if not isinstance(f, pd.Series):
            f = pd.Series(f)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"x{i}" for i in range(X.shape[1])]
            self.data_types = {f"x{i}": value for i, (key, value) in enumerate(data_types.items())}
        
        X = X[list(data_types.keys())]
        X.index = range(len(X))
        f.index = range(len(f))
        f.name = "fitness"

        X_raw = copy.deepcopy(X)

        for column in X.columns:
            dtype = data_types[column]
            if dtype == "boolean":
                X[column] = X[column].astype(bool)
                X_raw[column] = X_raw[column].astype(bool)
            elif dtype == "categorical":
                X[column] = pd.Categorical(X[column]).codes
            elif dtype == "ordinal":
                X[column] = pd.Categorical(X[column], ordered=True).codes

        self.configs = pd.Series(X.apply(tuple, axis=1))
        config_dict = self._generate_config_dict(data_types, X)

        data = pd.concat([X_raw, f], axis=1)
        data.index = range(len(data))

        return data, config_dict
    
    def _generate_config_dict(
            self, 
            data_types: Dict[str, str], 
            data: pd.DataFrame
        ) -> Dict[Any, Any]:
        """Generate a dictionary specifying the domain of x"""
        max_values = data[list(data_types.keys())].max()
        config_dict = {}
        for idx, (key, dtype) in enumerate(data_types.items()):
            config_dict[idx] = {'type': dtype, 'max': max_values[key]}
        return config_dict
    
    def _batched_create_edges(
            self, 
            data,
            n_edit: int = 1
        ) -> List[List[Tuple]]:
        """Finding the neighbors for a list of configurations"""
        config_to_index_mapping = dict(zip(self.configs, data.index))
        config_to_fit_mapping = dict(zip(self.configs, data["fitness"]))
        edge_list = []
        for config in tqdm(
            self.configs,
            total = self.n_configs,
            desc = "# Determining edges"
        ):
            current_fit = config_to_fit_mapping[config]
            current_id = config_to_index_mapping[config]
            neighbors = self._generate_neighbors(config, self.config_dict_, n_edit)
            for neighbor in neighbors:
                try:
                    neighbor_fit = config_to_fit_mapping[neighbor]
                    delta_fit = current_fit - neighbor_fit
                    if (self.maximize and delta_fit < 0) or (not self.maximize and delta_fit > 0):
                        edge_list.append((
                            current_id, 
                            config_to_index_mapping[neighbor], 
                            abs(delta_fit)
                        ))
                except:
                    pass
  
        return edge_list

    def _parallel_create_edges(
                self, 
                data,
                n_edit: int = 1,
                max_workers: int = 8
            ) -> List[List[Tuple]]:
        """Finding the neighbors for a list of configurations"""
        config_to_index_mapping = dict(zip(self.configs, data.index))
        config_to_fit_mapping = dict(zip(self.configs, data["fitness"]))
        
        def process_config(config):
            current_fit = config_to_fit_mapping[config]
            current_id = config_to_index_mapping[config]
            neighbors = self._generate_neighbors(config, self.config_dict_, n_edit)
            local_edge_list = []
            
            for neighbor in neighbors:
                if neighbor in config_to_fit_mapping:
                    neighbor_fit = config_to_fit_mapping[neighbor]
                    delta_fit = current_fit - neighbor_fit
                    if (self.maximize and delta_fit < 0) or (not self.maximize and delta_fit > 0):
                        local_edge_list.append((
                            current_id, 
                            config_to_index_mapping[neighbor], 
                            abs(delta_fit)
                        ))
            
            return local_edge_list
        
        edge_list = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {executor.submit(process_config, config): config for config in self.configs}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_config),
                total=self.n_configs,
                desc="# Determining edges"
            ):
                try:
                    local_edges = future.result()
                    edge_list.extend(local_edges)
                except Exception as exc:
                    print(f"Config generated an exception: {exc}")
        
        return edge_list
    
    def _batched_find_neighbors(
            self, 
            configs: List[Tuple[Any, ...]], 
            n_edit: int = 1
        ) -> List[List[Tuple]]:
        """Finding the neighbors for a list of configurations"""
        neighbor_list = []
        for config in tqdm(
            configs,
            total = len(configs),
            desc = "# Calculating neighborhoods"
        ):
            neighbor_list.append(self._generate_neighbors(config, self.config_dict_, n_edit))
        
        return neighbor_list

    def _generate_neighbors(
            self, 
            config: Tuple[Any, ...], 
            config_dict: Dict[Any, Any],
            n_edit: int = 1, 
        ) -> List[Tuple[Any, ...]]:
        """Finding the neighbors of a given configuration"""
        @lru_cache(None)  
        def get_neighbors(index, value):
            config_type = config_dict[index]['type']
            config_max = config_dict[index]['max']
            
            if config_type == 'categorical':
                return [i for i in range(config_max + 1) if i != value]
            elif config_type == 'ordinal':
                neighbors = []
                if value > 0:
                    neighbors.append(value - 1)
                if value < config_max - 1:
                    neighbors.append(value + 1)
                return neighbors
            elif config_type == 'boolean':
                return [1 - value]
            else:
                raise ValueError(f"Unknown variable type: {config_type}")

        def k_edit_combinations():
            original_config = config
            for indices in combinations(range(len(config)), n_edit):
                current_config = list(original_config)  
                possible_values = [get_neighbors(i, current_config[i]) for i in indices]
                for changes in product(*possible_values):
                    for idx, new_value in zip(indices, changes):
                        current_config[idx] = new_value
                    yield tuple(current_config)
        
        return list(k_edit_combinations())

    def _construct_landscape(
            self, 
            data,
            edge_list: List[Tuple], 
        ) -> nx.DiGraph:
        """Constructing the fitness landscape"""
        if self.verbose:
            print("# Constructing landscape...")

        graph = nx.DiGraph()
        graph.add_weighted_edges_from(edge_list, "delta_fit")    

        if self.verbose:
            print(" - Adding node attributes...")
        for column in data.columns:
            nx.set_node_attributes(graph, data[column].to_dict(), column)

        self.n_edges = graph.number_of_edges()

        return graph
    
    def _add_network_metrics(
            self, 
            graph: nx.DiGraph, 
            weight: str = "delta_fit"
        ) -> nx.DiGraph:
        """Calculate basic network metrics for nodes"""
        if self.verbose:
            print("# Calculating network metrics...")

        graph = add_network_metrics(graph, weight=weight)

        return graph 

    def _determine_local_optima(self):
        """Determine the local optima in the landscape."""
        if self.verbose:
            print("# Determining local optima...")

        out_degrees = dict(self.graph.out_degree())
        is_lo = {node: out_degrees[node] == 0 for node in self.graph.nodes}
        nx.set_node_attributes(self.graph, is_lo, 'is_lo')
        self.n_lo = pd.Series(is_lo.values()).sum()

        is_lo = pd.Series(nx.get_node_attributes(self.graph, 'is_lo'))
        self.lo_index = list(is_lo[is_lo].sort_index().index)

    def _calculate_basin_of_attraction(self):
        """Determine the basin of attraction of each local optimum."""
        if self.verbose:
            print("-Calculating basins of attraction...")
        
        basin_index = defaultdict(int)
        dict_size = defaultdict(int)
        dict_diameter = defaultdict(list)

        for i in tqdm(range(self.n_configs), total=self.n_configs, desc=" -Local searching from each config"):
            lo, steps = hill_climb(self.graph, i, "delta_fit",)
            basin_index[i] = lo
            dict_size[lo] += 1
            dict_diameter[lo].append(steps)
        
        nx.set_node_attributes(self.graph, basin_index, "basin_index")
        nx.set_node_attributes(self.graph, dict_size, "size_basin")
        nx.set_node_attributes(self.graph, {k: max(v) for k, v in dict_diameter.items()}, "max_radius_basin")

        self.basin_index = basin_index

    def get_data(self, lo_only: bool=False) -> pd.DataFrame:
        """
        Get tabular landscape data as pd.DataFrame.
        
        Parameters
        ----------
        lo_only : bool, default=False
            Whether to return only local optima configurations.

        Returns
        -------
        pd.DataFrame : A pandas dataframe containing all information regarding each configuration.
        """
        if lo_only:
            if not self.has_lon:
                graph_lo_ = self.graph.subgraph(self.lo_index)
                data_lo = pd.DataFrame.from_dict(dict(graph_lo_.nodes(data=True)), orient='index').sort_index()
                return data_lo.drop(columns=["is_lo", "out_degree", "in_degree", "basin_index"])
            else:
                data_lo = pd.DataFrame.from_dict(dict(self.lon.nodes(data=True)), orient='index').sort_index()
                return data_lo
        else:
            data = pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index').sort_index()
            return data.drop(columns=["size_basin", "max_radius_basin"])

    def _determine_global_optimum(self):
        """Determine global optimum of the landscape."""
        if self.verbose:
            print("# Determining global peak...")

        fitness_list = pd.Series(nx.get_node_attributes(self.graph, 'fitness'))

        if self.maximize:
            self.go_index = fitness_list.idxmax()
        else:
            self.go_index = fitness_list.idxmin()

        self.go = self.graph.nodes[self.go_index]

    def describe(self):
        """Print the basic information of the landscape."""
        print("---")
        print(f"number of variables: {self.n_vars}")
        print(f"number of configurations: {self.n_configs}")
        print(f"number of connections: {self.n_edges}")
        print(f"number of local optima: {self.n_lo}")

    def FDC(
            self, 
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
        float : An FDC value ranging from -1 to 1. A value close to 1 indicates positive correlation
            between the fitness values of a configuration and its distance to the global optimum.
        """
        return FDC(self, distance=distance, method=method)

    def FFI(
            self, 
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
        float : An FFI value ranging from -1 to 1. A value close to 1 indicates that the landscape
            is very likely to be flatter around the global optimum. 
        """
        return FFI(self, frac=frac, min_len=min_len, method=method)
    
    def fitness_assortativity(self):
        
        if self.graph.number_of_nodes() > 100000:
            warnings.warn("The number of nodes in the graph is greater than 100,000.")

        self.assortativity = nx.numeric_assortativity_coefficient(self.graph, "fitness")
        return self.assortativity

    def autocorrelation(
            self,
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
        return autocorrelation(self, walk_length=walk_length, walk_times=walk_times, lag=lag)
    
    def get_lon(
            self,
            mlon: bool = True,
            min_edge_freq: int = 3,
            trim: int = None,
            verbose: bool = True    
        ) -> nx.DiGraph:
        """
        Construct the local optima network (LON) of the fitness landscape 

        Parameters
        ----------
        mlon : bool, default=True
            Whether to use monotonic-LON (M-LON), which will only have improving edges, 
            i.e., the fitness of the target local optimum is also superiror than the source.

        min_edge_freq : int, default = 3
            Minimal escape frequency needed to construct an edge between two local optima. 
            This will be used to mask the adjacency matrix. 

        trim : int, default=None
            The number of edges with the highest transition probability to retain for each node. 
            If None, all transitions between local optima are retained.

        Returns
        -------
        nx.DiGraph : The constructed local optimum network (LON).
        """
    
        if verbose:
            print("Constructing local optima network...")

        lo_neighbors_list = self._batched_find_neighbors(
            self.configs[self.lo_index], 
            n_edit = 2)
        
        lo_adj = self._calculate_lon_adj(lo_neighbors_list, min_edge_freq)
        self.lon = self._create_lon(lo_adj)
        self._calculate_escape_rate(lo_adj)
        self.lo_adj = lo_adj
        self.lo_neighbors_list = lo_neighbors_list
        self._calculate_improve_rate()
        
        if mlon:
            self._get_mlon()

        if trim:
            self._trim_lon(trim)

        self._calculate_lo_accessibility()

        if self.verbose:
            print("# Adding further node attributes...")
        self.lon = self._add_network_metrics(self.lon, weight="weight")
        self.has_lon = True

        return self.lon

    def _calculate_lon_adj(self, neighbors_list, min_edge_freq = 3) -> np.array:
        """
        Calculate the adjacency matrix for LON. 
        """
        basin_index = pd.Series(nx.get_node_attributes(self.graph, 'basin_index')).sort_index()
        lo_to_index_mapping = dict(zip(self.lo_index, range(self.n_lo)))
        basin_index = basin_index.map(lo_to_index_mapping)

        config_to_basin_mapping = dict(zip(
            self.configs, 
            basin_index
        ))

        lo_adj = np.zeros((self.n_lo, self.n_lo), dtype=np.int16)
        for i, lo_neighbors in tqdm(
            enumerate(neighbors_list), total=self.n_lo, 
            desc=" - Creating adjacency matrix"
        ):
            try:
                for neighbor in lo_neighbors:
                    lo_adj[i, config_to_basin_mapping[neighbor]] += 1
            except:
                pass

        if self.verbose:
            print(f" - Masking positions with transition frequency <= {min_edge_freq}")        
        lo_adj = np.where(lo_adj <= min_edge_freq, 0, lo_adj)
        
        return lo_adj
    
    def _create_lon(self, lo_adj: np.array) -> nx.DiGraph:
        """
        Create LON based on adjacency matrix. 
        """
        if self.verbose:
            print("# Creating LON from adjacency matrix...")
            
        index_to_lo_mapping = dict(zip(range(self.n_lo), self.lo_index))

        lon = nx.DiGraph(lo_adj)
        lon = nx.relabel_nodes(lon, index_to_lo_mapping)

        graph_lon_ = self.graph.subgraph(self.lo_index)
        for attribute in ['fitness', 'size_basin', 'max_radius_basin', 'config']:
            dic = nx.get_node_attributes(graph_lon_, attribute)
            nx.set_node_attributes(lon, dic, attribute)

        self_loops = list(nx.selfloop_edges(lon))
        lon.remove_edges_from(self_loops)

        return lon

    def _calculate_escape_rate(self, lo_adj: np.array):
        """
        Calculate the probability of escaping from a local optimum.
        """
        column_sums = np.sum(lo_adj, axis=1) - np.diag(lo_adj)
        escape_difficulty = np.zeros(lo_adj.shape[0])
        for i in tqdm(range(self.n_lo), total=self.n_lo, desc="# Calculating escape probability"):
            if column_sums[i] != 0:  
                escape_difficulty[i] = lo_adj[i, i] / (column_sums[i] + lo_adj[i, i])
            else:
                escape_difficulty[i] = 1
        dic = dict(zip(self.lo_index, escape_difficulty))
        nx.set_node_attributes(self.lon, dic, 'escape_difficulty')

    def _calculate_improve_rate(self):
        if self.verbose:
            print("# Calculating improve rate...")
        improvement_measure = {}

        for node in self.lon.nodes():
            total_outgoing_weight = 0
            improving_moves_weight = 0
            current_fitness = self.lon.nodes[node]['fitness']
            
            for target in self.lon.successors(node):
                edge_data = self.lon.get_edge_data(node, target)
                edge_weight = edge_data['weight']
                total_outgoing_weight += edge_weight
                target_fitness = self.lon.nodes[target]['fitness']  
                if self.maximize:   
                    if target_fitness > current_fitness:
                        improving_moves_weight += edge_weight
                else:       
                    if target_fitness < current_fitness:
                        improving_moves_weight += edge_weight
            
            if total_outgoing_weight > 0:
                improvement_measure[node] = improving_moves_weight / total_outgoing_weight
            else:
                improvement_measure[node] = 0
        
        nx.set_node_attributes(self.lon, improvement_measure, 'improve_rate')

    def _get_mlon(self) -> None:
        """
        Transforms the Local Optimum Network (LON) into a Monotonic LON (M-LON).
        The M-LON is created by keeping only the improving edges based on the fitness values.
        """
        self.lon = get_mlon(self.lon, self.maximize, "fitness")
        if self.verbose:
            print(" - The LON has been reduced to M-LON by keeping only improving edges")

    def _trim_lon(self, k: int) -> None:
        """
        Trim the LON to keep only k out-goging edges from each local optiam with the largest 
        transition probability.

        k : int, default=10
            The number of edges to retain for each node. Default is 10.
        """
        self.lon = trim_lon(self.lon, k, "fitness")
        if self.verbose:
            print(f" - The LON has been trimmed to keep only {k} edges for each node.")

    def _calculate_lo_accessibility(self):
        """
        Calculate the accessibility of each local optima (LO) in the LON. This is done by assessing 
        how many LOs are able to reach a given LO via improving escape moves.
        """
        access_lon = []
        for node in tqdm(self.lon.nodes, total=self.n_lo, desc="# Calculating accessibility of LOs:"):
            access_lon.append(len(nx.ancestors(self.lon, node)))

        dic = dict(zip(self.lon.nodes, access_lon))
        nx.set_node_attributes(self.lon, dic, 'accessibility')

    def draw_neighborhood(
            self, 
            node: Any, 
            radius: int = 1, 
            node_size: int = 300, 
            with_labels: bool = True, 
            font_weight: str = 'bold', 
            font_size: int = 12,
            font_color: str = 'black', 
            node_label: str = "fitness", 
            node_color: Any = "fitness",
            edge_label: str = "delta_fit",
            colormap = plt.cm.RdBu_r, 
            alpha: float = 1.0
        ):
        """
        Visualizes the neighborhood of a node in a directed graph within a specified radius.

        Parameters
        ----------
        G : nx.DiGraph
            The directed graph.
        
        node : Any
            The target node whose neighborhood is to be visualized.
        
        radius : int, optional, default=1
            The radius within which to consider neighbors.
        
        node_size : int, optional, default=300
            The size of the nodes in the visualization.
        
        with_labels : bool, optional, default=True
            Whether to display node labels.
        
        font_weight : str, optional, default='bold'
            Font weight for node labels.

        font_size : str, optional, default=12
            Font size for labels.
        
        font_color : str, optional, default='black'
            Font color for node labels.
        
        node_label : str, optional, default=None
            The node attribute to use for labeling, if not the node itself.
        
        node_color : Any, optional, default=None
            The node attribute to determine node colors.

        edge_label : str, optional, default="delta_fit"
            The edge attribute to use for labeling edges. If None, then no edge labels 
            are displayed.
        
        colormap : matplotlib colormap, optional, default=plt.cm.Blues
            The Matplotlib colormap to use for node coloring.
        
        alpha : float, optional, default=1.0
            The alpha value for node colors.
        """
        draw_neighborhood(
            G=self.graph, 
            node=node, 
            radius=radius, 
            node_size=node_size, 
            with_labels=with_labels, 
            font_weight=font_weight, 
            font_size=font_size,
            font_color=font_color, 
            node_label=node_label, 
            node_color=node_color, 
            edge_label=edge_label,
            colormap=colormap, 
            alpha=alpha
        )
    
    def draw_landscape_2d(
        self,
        fitness: str="fitness",
        embedding_model: Any = HOPE(),
        reducer: Any = umap.UMAP(n_neighbors=15, n_epochs=500, min_dist=1),
        rank: bool = True,
        n_grids: int = 100,
        cmap: Any = BlueOrangeRed_3
    ) -> None:
        """
        Draws a 2D visualization of a landscape by plotting reduced graph embeddings and coloring them 
        according to the fitness values.

        Parameters
        ----------
        landscape : Any
            The landscape object that contains the graph and data for visualization.

        fitness : str, default="fitness"
            The name of the fitness column in the landscape data that will be visualized on the contour plot.

        embedding_model : Any, default=HOPE()
            The model used to generate embeddings from the landscape's graph. It should implement fit and 
            get_embedding methods.

        reducer : Any, default=umap.UMAP(...)
            The dimensionality reduction technique to be applied on the embeddings.
        rank : bool, default=True
            If True, ranks the metric values across the dataset.

        n_grids : int, default=100
            The number of divisions along each axis of the plot grid. Higher numbers increase the 
            resolution of the contour plot.

        cmap : Any, default=BlueOrangeRed_3
            The color map from 'palettable' used for coloring the contour plot.
        """
        draw_landscape_2d(
            self,
            metric=fitness,
            embedding_model=embedding_model,
            reducer=reducer,
            rank=rank,
            n_grids=n_grids,
            cmap=cmap
        )

    def draw_landscape_3d(
        self,
        fitness: str="fitness",
        embedding_model: Any = HOPE(),
        reducer: Any = umap.UMAP(n_neighbors=15, n_epochs=500, min_dist=1),
        rank: bool = True,
        n_grids: int = 100,
        cmap: Any = BlueOrangeRed_3
    ) -> None:
        """
        Draws a 3D interactive visualization of a landscape by plotting reduced graph embeddings and coloring 
        them according to a specified metric. 

        Parameters
        ----------
        landscape : Any
            The landscape object that contains the graph and data for visualization.

        fitness : str, default="fitness"
            The name of the fitness score in the landscape data that will be visualized on the contour plot.

        embedding_model : Any, default=HOPE()
            The model used to generate embeddings from the landscape's graph. It should implement fit and 
            get_embedding methods.

        reducer : Any, default=umap.UMAP(...)
            The dimensionality reduction technique to be applied on the embeddings. 

        rank : bool, default=True
            If True, ranks the metric values across the dataset.

        n_grids : int, default=100
            The number of divisions along each axis of the plot grid. Higher numbers increase the 
            resolution of the contour plot.

        cmap : Any, default=BlueOrangeRed_3
            The color map from 'palettable' used for coloring the contour plot.
        """
        draw_landscape_3d(
            self,
            metric=fitness,
            embedding_model=embedding_model,
            reducer=reducer,
            rank=rank,
            n_grids=n_grids,
            cmap=cmap
        )
        
        



