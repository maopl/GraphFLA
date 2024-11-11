import random
import itertools
import math
import pandas as pd

class OptimizationProblem:
    """
    Base class for defining optimization problems.

    This class provides a framework for representing optimization problems
    and includes methods for evaluating solutions and generating data for analysis.
    Subclasses should implement specific optimization problem behavior.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    """

    def __init__(self, n):
        """
        Initialize the optimization problem with a given number of variables.
        """
        self.n = n
        self.variables = range(n)
    
    def evaluate(self, config):
        """
        Evaluate the fitness of a given configuration.

        This method should be implemented by subclasses to define the 
        specific evaluation criteria for the optimization problem.

        Parameters
        ----------
        config : tuple
            A configuration representing a potential solution.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_all_configs(self):
        """
        Generate all possible configurations for the problem.

        This method should be implemented by subclasses to provide the
        complete set of possible configurations for the problem.

        Returns
        -------
        iterator
            An iterator over all possible configurations.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        raise NotImplementedError("Subclasses should define this method to generate all configurations.")
    
    def get_data(self):
        """
        Generate a DataFrame containing configurations and their fitness values.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns `config` (list of variables) and `fitness`.
        """

        all_configs = self.get_all_configs()
        config_values = {config: self.evaluate(config) for config in all_configs}
        
        data = pd.DataFrame(list(config_values.items()), columns=["config", "fitness"])
        data['config'] = data['config'].apply(list)
        return data

class NK(OptimizationProblem):
    """
    NK model for fitness landscapes.

    This class represents an NK landscape, a model used to study the complexity of 
    adaptive landscapes based on interactions among components.

    Parameters
    ----------
    n : int
        The number of variables in the problem.

    k : int
        The number of interacting components for each variable.

    exponent : int, default=1
        An exponent used to transform the fitness values.
    """

    def __init__(self, n, k, exponent=1):
        """
        Initialize the NK model with given parameters.
        """

        super().__init__(n)
        self.k = k
        self.exponent = exponent
        self.dependence = [
            tuple(sorted([e] + random.sample(set(self.variables) - set([e]), k)))
            for e in self.variables
        ]
        self.values = {}
    
    def get_all_configs(self):
        """
        Generate all possible binary configurations for the NK model.

        Returns
        -------
        iterator
            An iterator over all binary configurations of length `n`.
        """

        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the NK model.

        Parameters
        ----------
        config : tuple
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """

        total_value = 0.0
        config = tuple(config)  
        for e in self.variables:
            key = (e,) + tuple(config[i] for i in self.dependence[e])
            if key not in self.values:
                self.values[key] = random.random()
            total_value += self.values[key]
        
        total_value /= self.n
        if self.exponent != 1:
            total_value = math.pow(total_value, self.exponent)
        
        return total_value

class Max3Sat(OptimizationProblem):
    """
    Max-3-SAT optimization problem.

    This class represents the Max-3-SAT problem, where the goal is to maximize
    the number of satisfied clauses in a Boolean formula with exactly three literals per clause.

    Parameters
    ----------
    n : int
        The number of Boolean variables.

    alpha : float
        The clause-to-variable ratio, determining the number of clauses.
    """

    def __init__(self, n, alpha):
        """
        Initialize the Max-3-SAT problem with given parameters.
        """

        super().__init__(n)
        self.m = int(alpha * n)
        self.clauses = self._generate_clauses()
        
    def _generate_clauses(self):
        """
        Generate a set of 3-literal clauses.

        Returns
        -------
        list
            A list of randomly generated 3-literal clauses.
        """

        clauses = set()
        while len(clauses) < self.m:
            vars = random.sample(self.variables, 3)
            clause = tuple((var, random.choice([True, False])) for var in vars)
            clauses.add(clause)
        return list(clauses)
    
    def get_all_configs(self):
        """
        Generate all possible configurations for the Max-3-SAT problem.

        Returns
        -------
        iterator
            An iterator over all Boolean configurations of length `n`.
        """

        return itertools.product((True, False), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Max-3-SAT problem.

        Parameters
        ----------
        config : tuple
            A Boolean configuration representing a potential solution.

        Returns
        -------
        int
            The number of satisfied clauses.
        """

        num_satisfied = 0
        for clause in self.clauses:
            if any((config[var] == is_positive) for var, is_positive in clause):
                num_satisfied += 1
        return num_satisfied
