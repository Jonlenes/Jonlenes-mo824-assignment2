import os
import json
import numpy as np

def generate_instance(num_vertices):
    # Posição do vertex no plano [0, 1]
    poses = np.random.uniform(size=(num_vertices, 2))
    # Compute the distance btw the points
    dists = np.sum(np.abs(poses[:, None, :] - poses[None, :, :]), axis=-1)
    return [num_vertices, poses, dists]

def instance2json(ins):
    names = ['num_vertices', 'poses', 'dists']
    dic_ins = {}
    for name, value in zip(names, ins):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dic_ins[name] = value
    return json.dumps(dic_ins)

def save_json(filename, str_json):
    with open(filename, 'w') as file:
        file.write(str_json)

def generate_and_save_all(save_folder='data'):
    os.makedirs(save_folder, exist_ok=True)

    # Quantidade de vértices
    V = [20, 40, 60, 80, 100]

    for index, v in enumerate(V):
       ins = generate_instance(v)
       json_ins = instance2json(ins)
       save_json(os.path.join(save_folder, f'instancia-{index}.json'), json_ins)

if __name__ == "__main__":
    generate_and_save_all()