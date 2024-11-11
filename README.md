
# GraphFLA Package

## Overview
`GraphFLA` (Graph-based Fitness Landscape Analysis) is a package designed for scalable analysis of fitness landscapes for black-box optimization problems (BBOPs). It provides end-to-end tools for constructing, manipulating, analyzing and visualizing performance data, and thereby enhancing our way for understanding the variable-objective mapping of BBOPs.

## Landscape Construction
The `GraphFLA` package has native support for constructing landscapes from both artificial and real-world data.

### Construction with Synthetic Data

As a demo, here we show how to construct a synthetic landscape using the popular Kauffman's NK model in `Graph`. The NK model comes with two tunable parameters: `n` and `k`. 
- `n` determines the dimension of the problem.
- `k` controls the degree of dependence between different variables. 

We first create an instance of the NK model with `n=10` and `k=5`.

```python
from graphfla.problems import NK

nk_model = NK(n=10, k=5)
```

We now generate the fitness data by calling the `data()` method. This will return a pandas DataFrame containing all the possible configurations and their associated fitness values.

```python
df = nk_model.get_data()
```

After obtaining the fitness data, we can now create a `Landscape` object based on it. This is process is similar to building a machine learning model in `scikit-learn`: we split the dataset into configurations `X` and their associated fitness values `f`. Then, by additionally specifying whether we want to maximize or minimize the fitness as well as the type of each variable, the `Landscape` object can be created. 

```python
from graphfla.landscape import Landscape

X = df["config"].apply(pd.Series)
f = df["fitness"]

data_types = {x: "boolean" for x in X.columns}

landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

After creating the landscape, the `landscape.describe()` method can be used to generate a brief overview of the landscape properties.
```python
landscape.describe()
```

In addition to the NK model, other classic landscape models (e.g., Rough Mount Fuji (RMF)) will be supported in later version. Also, combinatorial optimization problems (e.g., TSP, NPP, MAX-SAT), or, customized problems, can be plugged in to this framework. 

### Construction with Real-World Data

In addition to artificial landscapes, `GraphFLA` is designed to handle various types of real-world BBOPs. Below are some use cases:

#### Case 1: Hyperparameter optimization (HPO)

Efficient tuning of the hyperparameters of machine learning (ML) models has been a critical task concerned in research and applications domains, and understanding the topography of the optimization landscape is crucial for designing effective HPO algorithms.

To illustrate how `GraphFLA` could assist in this understanding, here we use a demo dataset containing 14,960 hyperparameter configurations for a `XGBoostClassifer` across 5 hyperparameters and their corresponding test performance (test accuracy) on an OpenML dataset. 

Similarly, to construct the landscape for this HPO task, we first load the dataset and split it into configurations `X` and their associated fitness values `f`. We then specify the data types of the hyperparameters, which can be ordinal, categorical, or boolean. Once these steps are completed, we can create a `Landscape` object, with the `maximize` parameter set to `True` since we aim to maximize the test accuracy.

```python
df = pd.read_csv("example_data/hpo_landscape.csv", index_col=0)
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

The structure of the fitness landscapes of biological systems (e.g., DNA/RNA sequences or proteins) play a fundamental role in understanding the evolutionary dynamics of organisms.

This case replicates the experiments of the Science (2023) paper "A rugged yet easily navigable fitness landscape". The authors exhaustively measured the fitness of all possible genotypes derived from 9 positions of the *Escherichia coli folA* gene. This results in 4^9 = 262,144 DNA genotypes, where 135,178 are functional. Here, the search space is categorical, where each position can take one of the four nucleotides (A, T, C, G). Their findings with landscape analysis answered the long-standing question of relationship between landscape ruggedness and the accessibility of evolutionary pathways.

```python
df = pd.read_csv("example_data/dna_landscape.csv", index_col=0)
X = df.iloc[:,:9]
f = df["fitness"]
data_types = {x: "categorical" for x in X.columns}

landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

#### Case 3: Software configuration

This case is based on this present submission to ISSTA'24, where we measured the runtime of the LLVM compiler on 2^20 = 1,048,576 configurations generated from 20 configurable options. The search space is boolean, where each option can be either enabled or disabled. 

Notably, this case demostrates the scalability of `GraphFLA` in handling large-scale data (e.g., at the scale of millions). The 

```python
df = pd.read_csv("LLVM_2mm.csv")
X = df.iloc[:,:20]
f = df["run_time"]
data_types = {x: "boolean" for x in X.columns}
landscape = Landscape(X, f, maximize=True, data_types=data_types)
```

## Landscape Analysis

After constructing the landscape, `GraphFLA` then provides versatile tools for characterizing the topography of the landscape. For example:

```python
# calculate the autocorrelation
# it is a measure of landscape ruggedness
landscape.autocorrelation()

# calculate the neutrality index
# it measures the prevalence of fitness plateaus
landscape.neutrality()

# calculate the fitness flattening index (FFI)
# it measures to extend to which the landscape 
# tends to be flatter around the global optimum
landscape.ffi()
```

## Landscape Visualization

`GraphFLA` also provides a set of find-grained visualization tools to enable intuitive understandings of the constructed landscape.

```python
# visualize the neighborhood of a specific configuration
landscape.draw_neighborhood(node=1)

# generate an interactive plot of the landscape in low-dimensions
landscape.draw_landscape_3d()
```

## Advanced Analysis with Local Optima Network (LON)

Beyond the basic landscape analysis, `GraphFLA` also supports the construction of Local Optima Network (LON) from the landscape data. The LON is a compressed graph representation of the landscape, where each node is a local optima and each edge represents a transition between two local optima. Since local optima is one of the main obstacles in optimization, insights into their connectivity can provide valuable information for advancing the understanding of the underlying problem. 

```python
lon = landscape.get_lon(min_edge_freq=2)
print(lon.number_of_nodes(), lon.number_of_edges())
```