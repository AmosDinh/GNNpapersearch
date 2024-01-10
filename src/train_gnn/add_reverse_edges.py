# %%
import pickle

heterodata = pickle.load(open("hetero_graph_final.pkl", "rb"))

# %%
heterodata

# %%
heterodata['paper']['license'] = [license if license is not None else 'None' for license in heterodata['paper']['license']]

# %%
# idx = 336
# idx2 = 4
# a= heterodata[heterodata.edge_types[idx2]].edge_index
# from_node = heterodata[heterodata.edge_types[idx2][0]].name[a[0,idx]]
# to_node = heterodata[heterodata.edge_types[idx2][2]].name[a[1,idx]]

# print(heterodata.edge_types[idx2][0], from_node)
# print(heterodata.edge_types[idx2][2], to_node)

# %%
# heterodata[('paper', 'in_journal', 'journal')] = heterodata[('paper', 'in_journal', 'journal-ref')]
# del heterodata[('paper', 'in_journal', 'journal-ref')]

# remove isolated nodes
import torch_geometric.transforms as T
transform = T.Compose([
       #T.RemoveIsolatedNodes(),
       T.RemoveDuplicatedEdges(),
       # T.ToUndirected(merge=False) # don't merge reversed edges into the original edge type
])
from torch_geometric.data import HeteroData 

# from copy import deepcopy
# data = deepcopy(heterodata)
# for node_type in heterodata.node_types[-5:]:
       # heterodata[node_type].num_nodes = hetero

# for edge_type in  heterodata.edge_types[-:]:
       # print(edge_type)
       # 
       # del data[edge_type]
       # del data[list(reversed(heterodata.edge_types))[5]]

# for node_type in heterodata.node_types[:6]:
#        print(node_type)
#        del data[node_type]
# test = transform(data)
heterodata = transform(heterodata)

# %%
import torch
# id mapping so we can remove the attributes and then remove isolated nodes, later we can add the attributes back
id_dict = {}
more_mappings = {}
heterodata_dict = heterodata.to_dict()
for nodetype in heterodata.node_types:
    print(nodetype)
    ids = torch.arange(heterodata[nodetype].num_nodes) 
    id_mapping = {i.item():name for i, name in zip(ids, heterodata[nodetype].name)}
    if nodetype =='paper':
        for key in ['license','doi','pages','journal','date','id']:
            more_mappings[key] = {i.item():name for i, name in zip(ids, heterodata[nodetype][key])}
    
    heterodata[nodetype].x = ids.squeeze(-1)
    id_dict[nodetype] = id_mapping
    
    for key in heterodata_dict[nodetype].keys():
        if key != 'x':
            print('del', nodetype, key)
            del heterodata[nodetype][key]

# %%
heterodata

# %%
# check correct edge types:
transform = T.Compose([
       T.RemoveIsolatedNodes(),
       #    T.RemoveDuplicatedEdges(),
       # T.ToUndirected(merge=False) # don't merge reversed edges into the original edge type
])

heterodata = transform(heterodata)

# %%
heterodata


# %%
# from tqdm.auto import tqdm
# import json 
# import pandas as pd 

# def read_first_n_lines(file_path, n=10000):
#     data = []
#     with open(file_path, 'r') as file:
#         for i, line in tqdm(enumerate(file), total=n):
#             if i >= n:
#                 break
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue
#     return pd.DataFrame(data)

# file_path = '../make_the_graph/data/arxiv-metadata-oai-snapshot.json'

# # DataFrame erstellen
# df = read_first_n_lines(file_path, 1000)

# %%
heterodata = T.ToUndirected()(heterodata)


# %%
# map back
for nodetype in heterodata.node_types:
    print(nodetype)
    id_mapping = id_dict[nodetype]
    heterodata[nodetype].name = [id_mapping[i.item()] for i in heterodata[nodetype].x]
    
    heterodata[nodetype].num_nodes = len(heterodata[nodetype].name)
    
    if nodetype =='paper':
        for key in ['license','doi','pages','journal','date','id']:
            heterodata[nodetype][key] = [more_mappings[key][i.item()] for i in heterodata[nodetype].x]
    
    del heterodata[nodetype].x

# %%
# idx = 330
# idx2 = 4
# a= heterodata[heterodata.edge_types[idx2]].edge_index
# from_node = heterodata[heterodata.edge_types[idx2][0]].name[a[0,idx]]
# to_node = heterodata[heterodata.edge_types[idx2][2]].name[a[1,idx]]

# print(heterodata.edge_types[idx2][0], from_node)
# print(heterodata.edge_types[idx2][2], to_node)

# %%
import pickle
import os
if not os.path.exists('arxiv_author_paper_graph_no_features_bare.pkl'):
    pickle.dump(heterodata, open("arxiv_author_paper_graph_no_features_bare.pkl", "wb"))

# %%
# paper nans to median
import math
non_nan = [num for num in heterodata['paper'].pages if not math.isnan(num)]
median = non_nan[len(non_nan)//2]
heterodata['paper'].pages = [median if math.isnan(num) else num for num in heterodata['paper'].pages]


# %%
# categorical 
unique_licenses= list(set(heterodata['paper'].license))
categories = {unique_licenses[i]:i for i in range(len(unique_licenses))}
heterodata['paper'].license = [categories[license] for license in heterodata['paper'].license]
import torch 
# onehot
onehot = torch.nn.functional.one_hot(torch.tensor(heterodata['paper'].license ))
heterodata['paper'].license = onehot

# %%
given_timestamp = heterodata['paper'].date[0]
import datetime
start_timestamp_epoch = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=given_timestamp.tzinfo)

# Calculate the difference in hours divided by 
difference_in_hours_by_4 = (given_timestamp - start_timestamp_epoch).total_seconds() // (3600*4) 

# %%
heterodata['paper'].date = [((timestamp - start_timestamp_epoch).total_seconds() // (3600*4)) for timestamp in heterodata['paper'].date]

# %%
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# skill_sbert_embeddings = embedder.encode(skill_nodes['skill'].tolist(), convert_to_numpy=True)
from tqdm.auto import tqdm
# for each node type create the x attribute from the embeddings of the node names
for node_type in heterodata.node_types:
    print(node_type)
    # do it in batches
    x = torch.zeros(heterodata[node_type].num_nodes, 384).float()
    batch_size = 10000
    for i in tqdm(range(0, heterodata[node_type].num_nodes, 10000)):
        x[i:i+10000] = torch.tensor(embedder.encode(heterodata[node_type].name[i:i+10000], convert_to_numpy=True))
        
        
    heterodata[node_type].x = x
    # heterodata[node_type].x = embedder.encode(heterodata[node_type].name, convert_to_numpy=False)

# %%
heterodata['paper'].x = torch.concatenate((heterodata['paper'].x,heterodata['paper'].license), dim=1)
heterodata['paper'].x  = torch.concatenate((heterodata['paper'].x, torch.tensor(heterodata['paper'].date).unsqueeze(1)), dim=1)

# %%
for node_type in heterodata.node_types:
    del heterodata[node_type].name 
    if node_type == 'paper':
        for key in ['license','doi','pages','journal','date','id']:
            del heterodata[node_type][key]

# %%
# save 
import pickle
import os
if not os.path.exists('arxiv_author_paper_graph_training.pkl'):
    pickle.dump(heterodata, open("arxiv_author_paper_graph_training.pkl", "wb"))

# %%
# for nodetype in heterodata.node_types:
#     print(nodetype, heterodata[nodetype].x.shape, heterodata[nodetype].num_nodes)

# %%
def describe(heterodata):
    heterodata_dict = heterodata.to_dict()
    # print('has isolated nodes:', heterodata.has_isolated_nodes())
    # print('has self loops:', heterodata.has_self_loops())
    # print('is directed', heterodata.is_directed())
    for nodetype in heterodata.node_types:
        print(nodetype)
        print('  ',heterodata[nodetype].num_nodes)
        print('  ',heterodata_dict[nodetype].keys())
        
    for edgetype in heterodata.edge_types:
        print(edgetype)
        print('  ',heterodata[edgetype].num_edges)
        print('  ',heterodata[edgetype])

# describe(heterodata)


