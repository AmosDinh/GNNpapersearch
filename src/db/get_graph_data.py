import pickle

with open("graph.pkl", "rb") as f:
    graph = pickle.load(f)

def get_list_of_dicts(name: str):
    data = graph[name].to_dict()
    keys = list(data.keys())
    keys.remove("num_nodes")
    data = list(zip(*[data[key] for key in keys]))
    return [{key: value for key, value in zip(keys, values)} for values in data]

node_types = [
    "paper", 
    "author", 
    "category", 
    "word", 
    "journal"
    ]

for node in node_types:
    data = get_list_of_dicts(node)
    if node == "paper":
        [i.pop("date") for i in data]

    with open(f"{node}_data.pkl", "wb") as f:
        pickle.dump(data, f)