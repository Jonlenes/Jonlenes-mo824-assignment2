import os
import sys
import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from time import time
from read_instance import load_instance, list_avaliable_instances
from tqdm import tqdm


def build_tsp_model(instance):
    num_vertices, poses, dists = instance

    # Create a new model
    model = gp.Model("tsp")
    model.setParam(gp.GRB.Param.OutputFlag, 0)

    # Vars

    # Objective
    
    # Constraints

    return model

def build_2tsp_model(instance):
    num_vertices, poses, dists = instance

    # Create a new model
    model = gp.Model("2tsp")
    model.setParam(gp.GRB.Param.OutputFlag, 0)

    # Vars

    # Objective
    
    # Constraints

    return model


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
        ]
    )

    print("Starting experiments")
    for filename in tqdm(ins_filenames):
        # Load instance
        instance = load_instance(filename)

        # Save cost and time
        costs, times = [], []

        # Building the models
        for model in [build_tsp_model(instance), build_2tsp_model(instance)]:
            start_time = time()
            # Optimize model
            model.optimize()
            # Saving costs and times
            costs.append(model.objVal)
            times.append(time() - start_time)

        results = results.append(
            {
                "n_clients": instance[0],
                "n_vars": len(model.getVars()),
                "interger_cost": costs[0],
                "interger_time": round(times[0], 3),
                "relax_cost": costs[1],
                "relax_time": round(times[1], 3),
            },
            ignore_index=True,
        )
    results.to_csv("data/results.csv", index=False)


if __name__ == "__main__":
    ins_folder = "data"
    if len(sys.argv) > 1:
        ins_folder = sys.argv[1]
    main(ins_folder)
