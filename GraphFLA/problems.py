import random
import itertools
import math
import pandas as pd
from functools import lru_cache

class OptimizationProblem:
    def __init__(self, n):
        self.n = n
        self.variables = range(n)
    
    @lru_cache(maxsize=None)
    def evaluate(self, config):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_all_configs(self):
        raise NotImplementedError("Subclasses should define this method to generate all configurations.")
    
    def get_data(self):
        all_configs = self.get_all_configs()
        config_values = {config: self.evaluate(config) for config in all_configs}
        
        data = pd.DataFrame(list(config_values.items()), columns=["config", "fitness"])
        data['config'] = data['config'].apply(list)
        return data

class NK(OptimizationProblem):
    def __init__(self, n, k, exponent=1):
        super().__init__(n)
        self.k = k
        self.exponent = exponent
        self.dependence = [
            tuple(sorted([e] + random.sample(set(self.variables) - set([e]), k)))
            for e in self.variables
        ]
        self.values = {}
    
    def get_all_configs(self):
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
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
    def __init__(self, n, alpha):
        super().__init__(n)
        self.m = int(alpha * n)
        self.clauses = self._generate_clauses()
        
    def _generate_clauses(self):
        clauses = set()
        while len(clauses) < self.m:
            vars = random.sample(self.variables, 3)
            clause = tuple((var, random.choice([True, False])) for var in vars)
            clauses.add(clause)
        return list(clauses)
    
    def get_all_configs(self):
        return itertools.product((True, False), repeat=self.n)

    def evaluate(self, config):
        num_satisfied = 0
        for clause in self.clauses:
            if any((config[var] == is_positive) for var, is_positive in clause):
                num_satisfied += 1
        return num_satisfied
