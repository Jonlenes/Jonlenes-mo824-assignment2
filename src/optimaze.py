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
        for var in model._vars:
            vals = model.cbGetSolution(var)
            selected = gp.tuplelist((i, j) for i, j in var.keys()
                                    if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected, n)
            if len(tour) < n:
                # add subtour elimination constr. for every pair of cities in tour
                model.cbLazy(gp.quicksum(var[i, j]
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
    var_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name='edges')
    for i, j in var_edges.keys():
        var_edges[j, i] = var_edges[i, j]  # edge in opposite direction

    # Objective
    model.setObjective(gp.quicksum(var_edges[i, j] * dist[i, j] for i, j in dist.keys()), GRB.MINIMIZE)
    
    # Constraints
    #   Add degree-2 constraint
    model.addConstrs(var_edges.sum(i, '*') == 2 for i in range(num_vertices))

    model._vars = [var_edges]
    model.Params.lazyConstraints = 1

    #global n
    # n = num_vertices
    def callback(model, where):
        return subtourelim(num_vertices, model, where)
    return model, callback

def build_2tsp_model(instance):
    num_vertices, _, dist = instance
    dist = {eval(k):v for k, v in dist.items()}

    # Create a new model
    model = gp.Model("2tsp")
    model.setParam(gp.GRB.Param.OutputFlag, 0)

    # Variables
    x_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name='x_edges')
    y_edges = model.addVars(dist.keys(), vtype=GRB.BINARY, name='y_edges')
    for i, j in x_edges.keys():
        x_edges[j, i] = x_edges[i, j]  # edge in opposite direction
        y_edges[j, i] = y_edges[i, j]  # edge in opposite direction

    # Objective
    model.setObjective( gp.quicksum(x_edges[i, j] * dist[i, j] for i, j in dist.keys()) + gp.quicksum(y_edges[i, j] * dist[i, j] for i, j in dist.keys()), GRB.MINIMIZE)
    
    # Constraints
    #   Add degree-2 constraint
    # import pdb; pdb.set_trace()
    model.addConstrs(x_edges.sum(i, '*') == 2 for i in range(num_vertices))
    model.addConstrs(y_edges.sum(i, '*') == 2 for i in range(num_vertices))

    model.addConstrs(gp.quicksum([x_edges[i, j], y_edges[i, j]]) <= 1 for i, j in x_edges.keys())

    model._vars = [x_edges, y_edges]
    model.Params.lazyConstraints = 1

    #global n
    # n = num_vertices
    def callback(model, where):
        return subtourelim(num_vertices, model, where)
    return model, callback

def get_optimal_tour(model, n):
    tours, edges = [], []
    for var in model._vars:
        vals = model.getAttr('x', var)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        edges.append(list(selected))
        tour = subtour(selected, n)
        assert len(tour) == n
        tours.append(tour)
    if len(edges) > 1:
        assert not any([edge in edges[1] for edge in edges[0]])
    return tours, edges

def main(ins_folder):
    ins_filenames = list_avaliable_instances(os.path.join(ins_folder, "*.json"))
    results = pd.DataFrame()

    print("Starting experiments")
    for filename in tqdm(ins_filenames):
        # Load instance
        instance = load_instance(filename)

        # Save cost and time
        costs, times, optimal_tours, selected_edges = [], [], [], []

        # Building the models
        for model, callback in [build_tsp_model(instance), build_2tsp_model(instance)]:
            start_time = time()
            # Optimize model
            model.optimize(callback)
            # Saving costs and times
            costs.append(model.objVal)
            times.append(time() - start_time)

            # Get optimal tour
            tours, edges = get_optimal_tour(model, instance[0])
            optimal_tours.append(tours)
            selected_edges.append(edges)

        results = results.append(
            {
                "n_vertices": instance[0],
                "n_vars": len(model.getVars()),
                "cost_tsp": costs[0],
                "time_tsp": round(times[0], 3),
                "cost_2tsp": costs[1],
                "time_2tsp": round(times[1], 3),
                'optimal_tour_tsp': optimal_tours[0],
                'optimal_tour_2tsp': optimal_tours[1],
                'edges_tsp': selected_edges[0],
                'edges_2tsp': selected_edges[1],
            },
            ignore_index=True,
        )
    results.to_csv("data/results.csv", index=False)


if __name__ == "__main__":
    ins_folder = "data"
    if len(sys.argv) > 1:
        ins_folder = sys.argv[1]
    main(ins_folder)
