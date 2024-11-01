
# GraphFLA Package

## Overview
GraphFLA (Python-based Fitness Landscape Analysis) is a Python package designed for general analysis of fitness landscapes of black-box optimization problems (BBOPs). It provides efficient tools to construct fitness landscapes from measured fitness data, providing a playground for researchers to analyze and visualize the landscape properties of their optimization problems.

## Landscape Construction
The package has native support for constructing landscapes from both artificial and real-world data.

### Construction with Synthetic Data

Currently, we implemented the popular Kauffman's NK landscape model as a demo of artifical landscapes. The NK model comes with two tunable parameters: `n` and `k`. 
- `n` determines the number of bits in the configuration, i.e., the dimension of the problem.
- `k` controls the degree of dependence between different positions (i.e., loci). A higher `k` value means more interactions between features, and thus leads to a more rugged landscape. With `k=0`, the landscape is completely unimodal, while with `k=n-1`, the landscape is maximally rugged.

In `GraphFLA`, we can obtain all possible configurations (2^n) for a NK landscape as well as their corresponding fitness values in a compact DataFrame with the following code:

```python
from GraphFLA.problems import NK
from GraphFLA.landscape import Landscape

nk_model = NK(n=10, k=5)
df = nk_model.data()
```

After obtaining the landscape data, we can now create a `Landscape` object based on it. Here, everything is somewhat similar like building a machine learning model: we split the dataset into configurations `X` and their associated fitness values `f`. By additionally specifying whether we want to maximize or minimize the fitness as well as the type of each variable, the `Landscape` object can be created. 

```python
from GraphFLA.landscape import Landscape
X = df["config"].apply(pd.Series)
f = df["fitness"]

data_types = {x: "boolean" for x in X.columns}

landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

After creating the landscape, the `landscape.describe()` method can be used to get a brief overview of the landscape properties.
```python
landscape.describe()
```

In addition to the NK model, other classic landscape models (e.g., Rough Mount Fuji (RMF)) will be supported in later version. Also, combinatorial optimization problems (e.g., TSP, NPP, MAX-SAT), or, customized problems, can be plugged in to this framework. 


### Construction with Real-World Data

In addition to artificial landscapes, `GraphFLA` is also designed to handle various types of real-world optimization problems. Below are some use cases:

#### Case 1: Hyperparameter optimization (HPO)

Here we have a dataset containing 14,960 hyperparameter configurations for a `XGBoostClassifer` based on 5 hyperparameters and their corresponding performance metrics (test accuracy) on an OpenML dataset. `GraphFLA` is able to construct a *hyperparameter loss landscape* from this performance data, which could provides insights into the nature of the HPO problem, and thereby guiding the design of more efficient HPO algorithms.

The search space here contains 5 hyperparameters, taking ordinal (numerical) values. 

```python
df = pd.read_csv("hpo_xgb.csv")
X = df.iloc[:, :5]
f = df["acc_test"]

data_types = {
    "learning_rate": "ordinal",
    "max_bin": "ordinal",
    "max_depth": "ordinal",
    "n_estimators": "ordinal",
    "subsample": "ordinal",
}

landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

#### Case 2: Evolutionary biology (DNA sequences)

This case replicates the experiments of the Science (2023) paper "A rugged yet easily navigable fitness landscape". The authors exhaustively measured the fitness of all possible genotypes derived from 9 positions of the *Escherichia coli folA* gene. This results in 4^9 = 262,144 DNA genotypes, where 135,178 are functional. Here, the search space is categorical, in which each position can take one of the four nucleotides (A, T, C, G). Their findings with landscape analysis answered the long-standing question of relationship between landscape ruggedness and the accessibility of evolutionary pathways.

```python
df = pd.read_csv("dna_landscape.csv", index_col=0)
X = df.iloc[:,:9]
f = df["fitness"]
data_types = {x: "categorical" for x in X.columns}

landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

#### Case 3: Software configuration

This case is based on this present submission to ASE'24, where we measured the runtime of the LLVM compiler on 2^20 = 1,048,576 configurations generated from 20 configurable options. The search space is boolean, where each option can be either enabled or disabled. 

Notably, this case demostrates the scalability of `GraphFLA` in handling large-scale data (e.g., at the scale of millions).

```python
df = pd.read_csv("/home/Arwen/TEVC_LON_2024/data/LLVM_data/2mm.csv")
X = df.iloc[:,:20]
f = df["run_time"]
data_types = {x: "boolean" for x in X.columns}
landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

## Landscape Analysis

After constructing the landscape, `GraphFLA` then provides verstail tools for performing landscape analysis. Here are some examples:

```python
# calculate the autocorrelation
landscape.autocorrelation()
# calculate the fitness distance correlation (FDC)
landscape.FDC()
```

## Landscape Visualization

`GraphFLA` also provides a set of find-grained visualization tools to enable intuitive visualizations of the constructed landscape that are applicable to high-dimensional optimization problems. 

```python
# visualize the neighborhood of a specific configuration
landscape.draw_neighborhood(node=1)
# generate an interactive plot of the landscape in low-dimensions
landscape.draw_landscape_3d(n_grids=50, rank=False)
```

## Advanced Analysis with Local Optima Network (LON)

Beyond the basic landscape analysis, `GraphFLA` also supports the construction of Local Optima Network (LON) from the landscape data. The LON is a compressed graph representation of the landscape, where each node is a local optima and each edge represents a transition between two local optima. Since local optima is one of the main obstacles in optimization, insights into their connectivity can provide valuable information for advancing the understanding of the underlying problem. 

```python
lon = landscape.get_lon(min_edge_freq=2)
print(lon.number_of_nodes(), lon.number_of_edges())
```

# Collected Performance Data

Our collected performance data for over 86M configurations across 32 workloads of LLVM, Apache, and SQLite, is available at this link. 

