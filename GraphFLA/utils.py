import networkx as nx
import pandas as pd
from typing import Any

def add_network_metrics(graph: nx.DiGraph, weight: str) -> nx.DiGraph:
    """
    Calculate basic network metrics for nodes in the graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph for which the network metrics are to be calculated.

    weight : str, default='delta_fit'
        The edge attribute key to be considered for weighting. Default is 'delta_fit'.

    Returns
    -------
    nx.DiGraph
        The directed graph with node attributes added.
    """
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, weight=weight)

    nx.set_node_attributes(graph, in_degree, "in_degree")
    nx.set_node_attributes(graph, out_degree, "out_degree")
    nx.set_node_attributes(graph, pagerank, "pagerank")

    return graph

def get_mlon(
        graph: nx.DiGraph, 
        maximize: bool = True, 
        attribute: str = 'fitness'
    ) -> nx.DiGraph:
    """
    Generates a Monotonic Local Optima Network (M-LON) from a given directed graph.
    
    Parameters
    ----------
    G : nx.DiGraph
        The LON to be trimmed.

    maximize : bool
        Whether the fitness is to be optimized.
    
    attribute : str, default = "weight"
        The edge attribute key based on which the edges are sorted. Default is 'weight'.

    Return
    ------
    nx.DiGraph: The resulting M-LON
    """
    
    if maximize:
        edges_to_remove = [(source, end) for source, end in graph.edges()
                        if graph.nodes[source][attribute] > graph.nodes[end][attribute]]
    else:
        edges_to_remove = [(source, end) for source, end in graph.edges()
                        if graph.nodes[source][attribute] < graph.nodes[end][attribute]]

    graph.remove_edges_from(edges_to_remove)

    return graph

def trim_lon(
        graph: nx.DiGraph,
        k: int = 10, 
        attribute: str = 'weight'
    ) -> nx.DiGraph:
    """
    Trim the LON to keep only k out-goging edges from each local optiam with the largest transition probability.

    Parameters
    ----------
    G : nx.DiGraph
        The LON to be trimmed.

    k : int, default=10
        The number of edges to retain for each node. Default is 10.

    attribute : str, default = "weight"
        The edge attribute key based on which the edges are sorted. Default is 'weight'.
    
    Return
    ------
    nx.DiGraph: The resulting trimmed LON.
    """
    for node in graph.nodes():
        edges = sorted(graph.out_edges(node, data=True), key=lambda x: x[2][attribute], reverse=True)
        edges_to_keep = edges[:k]
        edges_to_remove = [edge for edge in graph.out_edges(node) if edge not in [e[:2] for e in edges_to_keep]]
        graph.remove_edges_from(edges_to_remove)
    
    return graph

def get_embedding(
        graph: nx.Graph,
        data: pd.DataFrame,
        model: Any,
        reducer: Any
    ) -> pd.DataFrame:
    """
    Processes a graph to generate embeddings using a specified model and then reduces the dimensionality
    of these embeddings using a given reduction technique. The function then augments the reduced embeddings
    with additional data provided.

    Parameters
    ----------
    graph : nx.Graph
        The graph structure from which to generate embeddings. This is used as input to the model.

    data : pd.DataFrame
        Additional data to be joined with the dimensionally reduced embeddings.

    model : Any
        The embedding model to be applied on the graph. This model should have fit and get_embedding methods.

    reducer : Any
        The dimensionality reduction model to apply on the high-dimensional embeddings. This model should
        have fit_transform methods.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dimensionally reduced embeddings, now augmented with the additional data.
        Each embedding is represented in two components ('cmp1' and 'cmp2').
    """
    model.fit(graph)
    embeddings = model.get_embedding()
    embeddings = pd.DataFrame(data=embeddings)

    embeddings_low = reducer.fit_transform(embeddings)
    embeddings_low = pd.DataFrame(data=embeddings_low)
    embeddings_low.columns=["cmp1","cmp2"]
    embeddings_low = embeddings_low.join(data)
    
    return embeddings_low

def relabel(graph: nx.Graph) -> nx.Graph:
    """
    Relabels the nodes of a graph to use sequential numerical indices starting from zero. This function
    creates a new graph where each node's label is replaced by a numerical index based on its position
    in the node enumeration.

    Parameters
    ----------
    graph : nx.Graph
        The graph whose nodes are to be relabeled. 

    Returns
    -------
    nx.Graph
        A new graph with nodes relabeled as consecutive integers, maintaining the original graph's structure.
    """
    mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    new_graph = nx.relabel_nodes(graph, mapping)
    return new_graph

def prepare_visualization_data(
        landscape: Any,
        metric: str,
        embedding_model: Any,
        reducer: Any,
        rank: bool = True
    ) -> pd.DataFrame:
    """
    Prepares data for visualization by generating embeddings of the graph, reducing their dimensionality, 
    and optionally ranking the specified metric. The function returns a DataFrame containing the reduced 
    embeddings along with the metric values.

    Parameters
    ----------
    landscape : Any
        The landscape object containing the graph and data to be visualized. The graph's structure is used
        to generate embeddings, and the data is augmented with metric values.

    metric : str
        The name of the fitness column in the landscape data to be included in the visualization.

    embedding_model : Any
        The model used to generate node embeddings from the graph. It should have a fit method and a 
        get_embedding method.

    reducer : Any
        The dimensionality reduction model applied to the high-dimensional embeddings. It should have a 
        fit_transform method.

    rank : bool, default=True
        Whether to rank the metric values. If True, the metric values are ranked in ascending order.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dimensionally reduced embeddings and the metric values, optionally ranked.
    """
    embedding_model.fit(relabel(landscape.graph))
    embeddings = embedding_model.get_embedding()
    embeddings = pd.DataFrame(data=embeddings)
    embeddings_low = reducer.fit_transform(embeddings)
    embeddings_low = pd.DataFrame(data=embeddings_low, columns=["cmp1", "cmp2"])

    data_ = embeddings_low.join(landscape.data[metric])

    df = data_.copy()
    if rank:
        df[metric] = df[metric].rank()

    return df