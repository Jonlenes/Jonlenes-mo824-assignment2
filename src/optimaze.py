import os
import sys
import numpy as np
import gurobipy as gp
import pandas as pd

from gurobipy import GRB
from itertools import combinations
from time import time
from read_instance import load_instance, list_avaliable_instances
from tqdm import tqdm

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(n, model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, n)
        if len(tour) < n:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour)-1)

# Given a tuplelist of edges, find the shortest subtour
def subtour(edges, n):
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

def build_tsp_model(instance):
    num_vertices, _, dist = instance
    dist = {eval(k):v for k, v in dist.items()}

    # Create a new model
    model = gp.Model("tsp")
    model.setParam(gp.GRB.Param.OutputFlag, 0)

    # Variables
    var_edges = model.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='edges')
    for i, j in var_edges.keys():
        var_edges[j, i] = var_edges[i, j]  # edge in opposite direction

    # Objective
    model.setObjective(gp.quicksum(var_edges[i, j] * dist[i, j] for i, j in dist.keys()), GRB.MINIMIZE)
    
    # Constraints
    #   Add degree-2 constraint
    model.addConstrs(var_edges.sum(i, '*') == 2 for i in range(num_vertices))

    model._vars = var_edges
    model.Params.lazyConstraints = 1

    #global n
    # n = num_vertices
    def callback(model, where):
        return subtourelim(num_vertices, model, where)
    return model, callback

def build_2tsp_model(instance):
    # Create a new model
    model = gp.Model("2tsp")
    model.setParam(gp.GRB.Param.OutputFlag, 0)

    # Vars

    # Objective
    
    # Constraints

    return model

def get_optimal_tour(model, n):
    vals = model.getAttr('x', model._vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    tour = subtour(selected, n)
    assert len(tour) == n
    return tour

def main(ins_folder):
    ins_filenames = list_avaliable_instances(os.path.join(ins_folder, "*.json"))
    results = pd.DataFrame(
        columns=[
            "n_vertices",
            "n_vars",
            "tsp_cost",
            "tsp_time",
            "2tsp_cost",
            "2tsp_time",
            'optimal_tour_tsp',
            'optimal_tour_2tsp'
        ]
    )

    print("Starting experiments")
    for filename in tqdm(ins_filenames):
        # Load instance
        instance = load_instance(filename)

        # Save cost and time
        costs, times, optimal_tours = [], [], []

        # Building the models
        for model, callback in [build_tsp_model(instance), build_tsp_model(instance)]:
            start_time = time()
            # Optimize model
            model.optimize(callback)
            # Saving costs and times
            costs.append(model.objVal)
            times.append(time() - start_time)
            optimal_tours.append(get_optimal_tour(model, instance[0]))

        results = results.append(
            {
                "n_vertices": instance[0],
                "n_vars": len(model.getVars()),
                "tsp_cost": costs[0],
                "tsp_time": round(times[0], 3),
                "2tsp_cost": costs[1],
                "2tsp_time": round(times[1], 3),
                'optimal_tour_tsp': optimal_tours[0],
                'optimal_tour_2tsp': optimal_tours[1]
            },
            ignore_index=True,
        )
    results.to_csv("data/results.csv", index=False)


if __name__ == "__main__":
    ins_folder = "data"
    if len(sys.argv) > 1:
        ins_folder = sys.argv[1]
    main(ins_folder)
