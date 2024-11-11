import pandas as pd
import networkx as nx
import copy 
import matplotlib.pyplot as plt
import umap.umap_ as umap
import concurrent.futures
import palettable

from karateclub import HOPE
from typing import List, Any, Dict, Tuple
from itertools import product, combinations
from collections import defaultdict
from tqdm import tqdm

from .lon import get_lon
from .algorithms import hill_climb
from .utils import add_network_metrics
from .distances import mixed_distance
from .metrics import *
from .visualization import *

import warnings
warnings.filterwarnings('ignore')

class Landscape():
    """
    Class implementing the fitness landscape object

    Parameters
    ----------
    X : pd.DataFrame or np.array
        The data containing the configurations to construct the landscape. 

    f : pd.Series or list or np.array
        The fitness values associated with the configurations. 

    graph : nx.DiGraph
        If provided, initialize the landscape with precomputed data as networkx directed graph. 

    maximize : bool
        Indicates whether the fitness is to be maximized or minimized. 

    data_types : dictionary
        A dictionary specifying the data type of each variable in X. Each variable can 
        be {"boolean", "categorical", "ordinal"}. If

        - X is pd.DataFrame, then the keys of data_types should match with columns of X.
        - X is np.array, the keys of data_types can be in arbitrary format, but the order 
          of the keys should be the same as in X. 

    allow_neutrality : bool, default=False
        Whether to allow neutrality in the landscape. Note that the framework is currently
        deisgned on non-neutral landscapes, and allowing neutrality may lead to unexpected
        bahaviors. 

    verbose : bool
        Controls the verbosity of output.
    
    Attributes
    ----------
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
    def __init__(
        self, 
        X: pd.DataFrame = None, 
        f: pd.Series = None,
        graph: nx.DiGraph = None,
        maximize: bool = True, 
        data_types: Dict[str, str] = None,
        allow_neutrality: bool = False,
        verbose: bool = True
        ) -> None:

        self.maximize = maximize
        self.verbose = verbose
        self.allow_neutrality = allow_neutrality
        self.has_lon = False
        self.graph = None
        self.configs = None
        self.config_dict = None
        self.basin_index = None
        self.data_types = None
        self.lo_index = None
        self.n_configs = None
        self.n_edges = None
        self.n_lo = None
        self.n_vars = None

        if graph is None:
            if self.verbose:
                print("Creating landscape from scratch with X and f...")
            if X is None or f is None:
                raise ValueError("X and f cannot be None if graph is not provided.")
            if len(X) != len(f):
                raise ValueError("X and f must have the same length.")
            if data_types is None:
                raise ValueError("data_types cannot be None if graph is not provided.")
            self.n_configs = X.shape[0]
            self.n_vars = X.shape[1]
            self.data_types = data_types
            data, self.config_dict = self._validate_and_prepare_data(X, f, data_types)
            del X, f
            edge_list = self._batched_create_edges(data, n_edit=1)
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
            neighbors = self._generate_neighbors(config, self.config_dict, n_edit)
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
            neighbors = self._generate_neighbors(config, self.config_dict, n_edit)
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
            neighbor_list.append(self._generate_neighbors(config, self.config_dict, n_edit))
        
        return neighbor_list

    def _generate_neighbors(
            self, 
            config: Tuple[Any, ...], 
            config_dict: Dict[Any, Any],
            n_edit: int = 1, 
        ) -> List[Tuple[Any, ...]]:
        """Finding the neighbors of a given configuration"""

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

    def _determine_local_optima(self) -> None:
        """Determine the local optima in the landscape."""

        if self.verbose:
            print("# Determining local optima...")

        out_degrees = dict(self.graph.out_degree())
        is_lo = {node: out_degrees[node] == 0 for node in self.graph.nodes}
        nx.set_node_attributes(self.graph, is_lo, 'is_lo')
        self.n_lo = sum(is_lo.values())

        is_lo = pd.Series(nx.get_node_attributes(self.graph, 'is_lo'))
        self.lo_index = list(is_lo[is_lo].sort_index().index)

    def _calculate_basin_of_attraction(self) -> None:
        """Determine the basin of attraction of each local optimum."""

        if self.verbose:
            print("# Calculating basins of attraction...")
        
        basin_index = defaultdict(int)
        dict_size = defaultdict(int)
        dict_diameter = defaultdict(list)

        for i in tqdm(range(self.n_configs), total=self.n_configs, desc=" - Local searching from each config"):
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

    def _determine_global_optimum(self) -> None:
        """Determine global optimum of the landscape."""

        if self.verbose:
            print("# Determining global peak...")

        fitness_list = pd.Series(nx.get_node_attributes(self.graph, 'fitness'))

        if self.maximize:
            self.go_index = fitness_list.idxmax()
        else:
            self.go_index = fitness_list.idxmin()

        self.go = self.graph.nodes[self.go_index]

    def describe(self) -> None:
        """Print the basic information of the landscape."""

        print("---")
        print(f"number of variables: {self.n_vars}")
        print(f"number of configurations: {self.n_configs}")
        print(f"number of connections: {self.n_edges}")
        print(f"number of local optima: {self.n_lo}")

    def fdc(
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

        return fdc(self, distance=distance, method=method)

    def ffi(
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

        return ffi(self, frac=frac, min_len=min_len, method=method)
    
    def fitness_assortativity(self) -> float:
        """
        Calculate the assortativity of the landscape based on fitness values.

        Returns
        -------
        float : The assortativity value of the landscape.
        """
        
        if self.graph.number_of_nodes() > 100000:
            warnings.warn("The number of nodes in the graph is greater than 100,000.")

        assortativity = nx.numeric_assortativity_coefficient(self.graph, "fitness")
        return assortativity

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
    
    def neutrality(self, threshold: float = 0.01) -> float:
        """
        Calculate the neutrality index of the landscape. It assesses the proportion of neighbors 
        with fitness values within a given threshold, indicating the presence of neutral areas in 
        the landscape.

        Parameters
        ----------
        threshold : float, default=0.01
            The fitness difference threshold for neighbors to be considered neutral.

        Returns
        -------
        neutrality : float
            The neutrality index, which ranges from 0 to 1, where higher values indicate more 
            neutrality in the landscape.
        """

        return neutrality(self, threshold=threshold)

    def ruggedness(self) -> float:
        """
        Calculate the ruggedness index of the landscape. It is defined as the ratio of the number
        of local optima to the total number of configurations.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        Returns
        -------
        float
            The ruggedness index, ranging from 0 to 1.
        """

        return ruggedness(self)

    def basin_size_fit_corr(self, method: str = "spearman") -> tuple:
        """
        Calculate the correlation between the size of the basin of attraction and the fitness of local optima.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        method : str, one of {"spearman", "pearson"}, default="spearman"
            The correlation measure to use.

        Returns
        -------
        tuple
            A tuple containing the correlation coefficient and the p-value.
        """

        return basin_size_fit_corr(self, method=method)

    def gradient_intensity(self) -> float:
        """
        Calculate the gradient intensity of the landscape. It is defined as the average absolute
        fitness difference (delta_fit) across all edges.

        Parameters
        ----------
        landscape : Landscape
            The fitness landscape object.

        Returns
        -------
        float
            The gradient intensity.
        """

        return gradient_intensity(self)

    def single_mutation_effects(
            self, 
            position: str, 
            test_type: str = 'positive', 
            n_jobs: int = 1
        ) -> pd.DataFrame:
        """
        Assess the fitness effects of all possible mutations at a single position across all genetic backgrounds.

        Parameters
        ----------
        position : str
            The name of the position (variable) to assess mutations for.
        
        test_type : str, default='positive'
            The type of significance test to perform. Must be 'positive' or 'negative'.
        
        n_jobs : int, default=1
            The number of parallel jobs to run.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing mutation pairs, median absolute fitness effect, 
            p-values, and significance flags.
        """

        return single_mutation_effects(
            landscape=self, 
            position=position, 
            test_type=test_type, 
            n_jobs=n_jobs
        )

    def all_mutation_effects(
            self, 
            test_type: str = 'positive', 
            n_jobs: int = 1
        ) -> pd.DataFrame:
        """
        Assess the fitness effects of all possible mutations across all positions in the landscape.

        Parameters
        ----------
        test_type : str, default='positive'
            The type of significance test to perform. Must be 'positive' or 'negative'.
        
        n_jobs : int, default=1
            The number of parallel jobs to run.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing, for each position and mutation pair, the median absolute fitness effect, 
            p-values, and significance flags.
        """

        return all_mutation_effects(
            landscape=self, 
            test_type=test_type, 
            n_jobs=n_jobs
        )
    
    def pairwise_epistasis(
            self, 
            pos1: str, 
            pos2: str
        ) -> pd.DataFrame:
        """
        Assess the pairwise epistasis effects between all unique unordered mutations 
        at two specified positions within the landscape.

        This method leverages the `pairwise_epistasis` function to automatically enumerate all 
        possible mutations at the given positions, compute epistatic interactions, and return 
        the results in a structured DataFrame.

        Parameters
        ----------
        pos1 : str
            The name of the first genetic position to assess mutations for.
        
        pos2 : str
            The name of the second genetic position to assess mutations for.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the epistasis results for all mutation pairs between 
            the two positions.

        Raises
        ------
        ValueError
            If either `pos1` or `pos2` is not a valid column in the landscape's genotype matrix.

        Examples
        --------
        ```python
        # Assuming you have a Landscape object named 'landscape'

        # Define the two positions to assess for epistasis
        position1 = 'position_3'
        position2 = 'position_5'

        # Compute pairwise epistasis between position1 and position2
        epistasis_results = landscape.pairwise_epistasis(pos1=position1, pos2=position2)

        # View the results
        print(epistasis_results)
        ```
        """

        data = self.get_data()
        X = data.iloc[:, :len(self.data_types)]
        f = data["fitness"]

        if pos1 not in X.columns:
            raise ValueError(f"Position '{pos1}' is not a valid column in the genotype matrix.")
        if pos2 not in X.columns:
            raise ValueError(f"Position '{pos2}' is not a valid column in the genotype matrix.")

        epistasis_df = pairwise_epistasis(X, f, pos1, pos2)

        return epistasis_df
    
    def all_pairwise_epistasis(
            self,
            n_jobs: int = 1
        ) -> pd.DataFrame:
        """
        Compute epistasis effects between all unique pairs of positions within the landscape using parallel execution.

        This method leverages the `all_pairwise_epistasis` function to iterate over all possible 
        pairs of genetic positions, compute their epistatic interactions in parallel, and compile the 
        results into a comprehensive DataFrame.

        Parameters
        ----------
        n_jobs : int, default=1
            The number of parallel jobs to run. -1 means using all available cores.

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing epistasis results for all position pairs.
            Each row corresponds to a specific mutation pair between two positions.

        Raises
        ------
        ValueError
            If the genotype matrix or fitness data is not properly initialized.

        Examples
        --------
        ```python
        # Assuming you have a Landscape object named 'landscape'

        # Compute epistasis between all pairs of positions using 4 cores
        all_epistasis_results = landscape.all_pairwise_epistasis(n_jobs=4)

        # View the results
        print(all_epistasis_results)
        ```
        """

        data = self.get_data()
        X = data.iloc[:, :len(self.data_types)]
        f = data["fitness"]

        if X.empty:
            raise ValueError("Genotype matrix X is empty.")
        if f.empty:
            raise ValueError("Fitness data f is empty.")
        if len(X) != len(f):
            raise ValueError("Mismatch between number of genotypes and fitness values.")

        all_epistasis_df = all_pairwise_epistasis(X, f, n_jobs=n_jobs)

        return all_epistasis_df

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
        ) -> None:
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
        cmap: Any = palettable.lightbartlein.diverging.BlueOrangeRed_3
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

        cmap : Any, default=palettable.lightbartlein.diverging.BlueOrangeRed_3
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
            cmap: Any = palettable.lightbartlein.diverging.BlueOrangeRed_3
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

        cmap : Any, default=palettable.lightbartlein.diverging.BlueOrangeRed_3
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

    def draw_epistasis(
        self,
        epistasis_df=None, 
        p_threshold=0.05, 
        cohen_d_threshold=0.5, 
        figsize=(5, 5),  
        node_color='#f2f2f2',  
        label_font_size=10,  
        node_size=500, 
        legend_loc='upper right',  
        edge_width_scale=2  
        ) -> None:
        """
        Calls the external draw_epistasis function to visualize epistatic interactions.
        
        Parameters
        ----------
        aggregated_epistasis_df : pd.DataFrame
            Aggregated epistasis results for all position pairs.

        p_threshold : float, default=0.05
            p-value threshold for significance.
        
        cohen_d_threshold : float, default=0.5
            Threshold for Cohen's d to define strong interactions.
        
        figsize : tuple, default=(8, 8)
            Size of the plot figure.
        
        node_color : str, default='#f2f2f2'
            Color of the nodes in the plot.
        
        label_font_size : int, default=10
            Font size for the node labels.
        
        node_size : int, default=500
            Size of the nodes in the plot.
        
        legend_loc : str, default='upper right'
            Location of the legend.
        
        edge_width_scale : float, default=2
            Scale factor for edge width based on `average_cohen_d`.

        Returns
        -------
        None
            Displays the epistasis plot.
        """
        if epistasis_df is None:
            epistasis_df = self.all_pairwise_epistasis()

        draw_epistasis(
            epistasis_df=epistasis_df, 
            p_threshold=p_threshold, 
            cohen_d_threshold=cohen_d_threshold, 
            figsize=figsize, 
            node_color=node_color, 
            label_font_size=label_font_size, 
            node_size=node_size, 
            legend_loc=legend_loc, 
            edge_width_scale=edge_width_scale
        )

    def get_lon(
            self,
            mlon: bool = True,
            min_edge_freq: int = 3,
            trim: int = None,
            verbose: bool = True    
        ) -> nx.DiGraph:
        """
        Construct the local optima network (LON) of the fitness landscape.

        Parameters
        ----------
        mlon : bool, default=True
            Whether to use monotonic-LON (M-LON), which will only have improving edges.

        min_edge_freq : int, default=3
            Minimal escape frequency needed to construct an edge between two local optima.

        trim : int, default=None
            The number of edges with the highest transition probability to retain for each node.

        Returns
        -------
        nx.DiGraph : The constructed local optimum network (LON).
        """

        if verbose:
            print("Constructing local optima network...")

        self.lon = get_lon(
            graph=self.graph,
            configs=self.configs,
            lo_index=self.lo_index,
            config_dict=self.config_dict,
            maximize=self.maximize,
            mlon=mlon,
            min_edge_freq=min_edge_freq,
            trim=trim,
            verbose=verbose
        )
        self.has_lon = True
        return self.lon