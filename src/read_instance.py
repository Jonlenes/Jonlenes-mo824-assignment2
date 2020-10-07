import json
import numpy as np
from glob import glob


def list_avaliable_instances(regex="data/*.json"):
    return glob(regex)


def read_json(filename):
    with open(filename, "r") as file:
        data = json.loads(file.read())
    return data


def load_instance(filename):
    json_ins = read_json(filename)
    return [
        json_ins["num_vertices"],
        json_ins["points"],
        json_ins["dist"]
    ]
