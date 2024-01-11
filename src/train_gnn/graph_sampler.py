# Sampling
# 1. Sample using HGT Sampler as outlined in the paper, using pyg implementations
# 2. The sampling is adapted to link prediction, by first sampling random supervision edges of which the nodes create the supervision nodes
# a. Dataset is divided across multiple dimensions:
#   a.1. Split into Train, Val, Test split (96, 2, 2)
#   a.2. Training only: Edges are split into those which are used solely for message passing and those solely used for supervision (80, 20). 
#        Because an expressive model (HGT) is used, this prevents the model from memorizing supervision edges by their appearance as message passing edges
#   a.3. This means Training consists of 96%*80% Message passing Edges, 96%*20% supervision edges, Val contains 2% Supervision Edges, Test contains 2% supervison Edges
#   a.4. Validation and Test edges use the Training Message passing Edges as well.
# b. For mini-batch sampling in the training phase, first x random edges are sampled as supervision edges. 
#    For the nodes of these supervision edges, we apply batch-wise HGT Sampling. Due to implementation limitations, for each supervision entity type, the hgt sampling is separate. 
#    This limitation does not apply for sampled neighbor entity types
# during sampling, also the reverse edge of the supervision edge is removed to avoid data leakage


# HGT Sampler (See Paper for further reference)
# The probablity of a neighbor node s to be sampled depends on the normalized degree of all its edge types connecting it to all source nodes
# If neighbor node s is connected to a and b by edge type r, and a has 2 neighbors through edge type r and b has 1 neighbor (node s) through edge type r, 
# then the sampling probablity of s is (1/2+1)**2 / 2**2, if it were connected through other edge types to the nodes as well, those degrees would be added to the numerator and denominator.
# Nodes are sampled without replacement.
# This sampling strategy creates more dense mini-batches, because neighbor nodes which are connected to multiple source nodes and by multiple relationship types are sampled more frequently.
# Therefore, training is sped up since less node representations have to be computed. Furthermore, as stated in the paper, the sampling method allows to sample a 
# similar number of neighbors for each edge type, because high-count edge types and low-count edge types are weighted equally. For each neighbor node type T, a fixed number n of nodes is sampled.





import os
import torch
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
import numpy as np
def add_reverse_edge_original_attributes_and_label_inplace(original_edge, reverse_edge):
    # add edge label and index and edge attr to reverse edge 
    if 'edge_attr' in original_edge:
        reverse_edge['edge_attr'] = original_edge['edge_attr']
    reverse_edge['edge_label'] = original_edge['edge_label']
    reverse_edge['edge_label_index'] = original_edge['edge_label_index'].index_select(0, torch.LongTensor([1, 0]))
    for key in original_edge.keys():
        if key not in ['edge_index', 'edge_attr', 'edge_label', 'edge_label_index']:
            reverse_edge[key] = original_edge[key]
                
    return reverse_edge

import pickle 

def get_datasets(get_edge_attr=False, filename=None, filter_top_k=False, top_k=50, remove_text_attr=True):
    if filename is None:
        filename = 'arxiv_author_paper_graph_training_v1.pkl'
    size = os.path.getsize(filename)
    print('size of dataset on disk: ', size/1e9, 'gb')

    if os.path.exists(filename):
        #data = HeteroData.from_dict(torch.load(filename))
        data = pickle.load(open(filename, 'rb'))
        print('loading saved heterodata object')


    
    def top_k_mask(scores, indices, top_k ):
        # Make sure we are using the GPU
        # scores = scores.cuda()
        # indices = indices.cuda()
        
        # Create an empty mask with the same shape as scores
        mask = torch.zeros_like(scores, dtype=torch.bool)
        # Get the unique indices and their counts
        unique_indices, counts = torch.unique(indices, return_counts=True)
    
        # Indices where count > top_k
        large_indices = unique_indices[counts > top_k]
    
        # Set mask for indices where count <= top_k
        mask[~torch.isin(indices,large_indices)] = True
        # For indices where count > 50, we only keep top 50 scores
        # if len(large_indices) > 0:
        #     idx_masks = [(indices == idx) for idx in tqdm(large_indices)]
        #     values_idxs = [scores[idx_mask].topk(top_k) for idx_mask in tqdm(idx_masks)]
        #     for idx_mask, (values, idxs) in tqdm(zip(idx_masks, values_idxs)):
        #         mask[idx_mask][idxs] = True
        
        
        # argsort the scores
        # first sort the numbers by the scores desceding
        # then sort the numbers in descending order such that the same numbers are neighbors but the  corresponding scores are still sorted in descending order
        # then get first top_k indices per number
        sorted_scores, sorted_score_indices = torch.sort(scores, descending=True)
        sorted_numbers = indices[sorted_score_indices]
        
        sorted_numbers_by_score_then_by_number, corresponding_mask = torch.sort(sorted_numbers, descending=True, stable=True)
        
        sorted_score_indices_by_score_then_by_number = sorted_score_indices[corresponding_mask]
        
        _, counts = torch.unique(sorted_numbers_by_score_then_by_number, sorted=True, return_counts=True)
        counts = torch.flip(counts, [0]) # descending
        start = torch.cat((torch.tensor([0]),torch.cumsum(counts, dim=0)[:-1]))
        end = start + top_k
        for s, e in zip(start, end):
            mask[sorted_score_indices_by_score_then_by_number[s:e]] = True
        
        # for idx in tqdm(large_indices):
            # mask[sorted_score_indices[(sorted_indices == idx)][:top_k]] = True
            
        # for idx in tqdm(large_indices):
        #     idx_mask = (sorted_indices == idx)
        #     reverse_map = sorted_score_indices[idx_mask][:top_k]
        #     mask[reverse_map] = True
            
        print('mask trues', torch.sum(mask), 'of total', mask.shape[0])
        return mask.cpu()

    
   
    if filter_top_k:
        def filter_once(edge, rev_e=None, keep_n_foreach_left=True, top_k=top_k):
            e = edge
            #rev_e = (e[2],'rev_'+e[1],e[0])
            
            if data[e].edge_attr.dim() == 1:
                data[e].edge_attr = data[e].edge_attr.unsqueeze(1)
            if rev_e is not None:
                if data[rev_e].edge_attr.dim() == 1:
                    data[rev_e].edge_attr = data[rev_e].edge_attr.unsqueeze(1)
            
            cache_dir = 'cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            mask_path = os.path.join(cache_dir, f'mask{top_k}_{"".join(list(edge))}.pt') 
            
            # if os.path.isfile(mask_path):
                # mask = torch.load(mask_path)
            # else:
            if keep_n_foreach_left:
                mask = top_k_mask(data[e].edge_attr.squeeze(1), data[e].edge_index[0,:], top_k)
            else: # for each right
                mask = top_k_mask(data[e].edge_attr.squeeze(1), data[e].edge_index[1,:], top_k)
                    
                # torch.save(mask, mask_path) 
            print('mask', mask.shape, data[e].edge_attr.shape, data[e].edge_index.shape)
            
            data[e].edge_attr = data[e].edge_attr[mask]
            data[e].edge_index = data[e].edge_index[:,mask]
            if rev_e is not None:
                data[rev_e].edge_attr = data[rev_e].edge_attr[mask]
                data[rev_e].edge_index = data[rev_e].edge_index[:,mask]
                
            print('keep',torch.sum(mask), 'of total',mask.shape[0])

        filter_once(('paper', 'has_word', 'word'), ('word', 'rev_has_word', 'paper'), keep_n_foreach_left=True) # for each paper only the top k words kept 
        filter_once(('word', 'co_occurs_with', 'word'), keep_n_foreach_left=True, top_k=top_k*2)
        filter_once(('word', 'co_occurs_with', 'word'), keep_n_foreach_left=False, top_k=top_k*2)
    
    data[('word','rev_co_occurs_with','word')].edge_index = data['word','co_occurs_with','word'].edge_index.flip(0)
    data[('word','rev_co_occurs_with','word')].edge_attr = data['word','co_occurs_with','word'].edge_attr
    
    
    from torch_geometric import seed_everything
    import torch_geometric.transforms as T
    from torch_geometric.utils import sort_edge_index


    
    edge_types = []
    rev_edge_types = []
    for edge_type in data.edge_types:
        print('len', edge_type, data[edge_type].edge_index.shape[1])
        if edge_type[1].startswith('rev_'):
            rev_edge_types.append(edge_type)
        else:
            edge_types.append(edge_type)

    
    transform = T.RandomLinkSplit(
        is_undirected=True,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types,
        num_val=0.05,
        num_test=0.05,
        add_negative_train_samples=False, # only adds neg samples for val and test, neg train are added by LinkNeighborLoader. This means for each train batch, negs. are different, for val and train they stay the same
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.3, #  training edges are shared for message passing and supervision
        )

    seed_everything(14)
    # sort by col to speed up sampling later (we can sepcify is_sorted=True in link neighbor loader)
    # we actually dont use the sort, because it seems to mess up things, but have not checked if everything works without sorting, so we leave it here
    def sort_edges(data):
        for edge_type in data.edge_types:
            if 'edge_attr' in data[edge_type].keys():
                data[edge_type].edge_index, data[edge_type].edge_attr = sort_edge_index(data[edge_type].edge_index, data[edge_type].edge_attr, sort_by_row=False) 
            else:
                data[edge_type].edge_index = sort_edge_index(data[edge_type].edge_index, sort_by_row=False) 
        return data
    

    def preprocess(data):
        if not get_edge_attr:
            # delete edge_attr of every edge type
            for edge_type in data.edge_types:
                del data[edge_type].edge_attr 

        # delete all keys for every node type except 'x' (e.g. description and title)
        for node_type in data.node_types:
            keys = list(data[node_type].keys())
            for key in keys:
                if key != 'x':
                    del data[node_type][key]
        return data
    
    
    # change all types to float32 and normalize the triangle columns
    for node_type in data.node_types:
        for i in range(data[node_type].x.shape[1]):
            if data[node_type].x[:,i].max()>5:
                #normalize
                print('normalizing column ', i, ' of node type ', node_type)
                data[node_type].x[:,i] = data[node_type].x[:,i]/data[node_type].x[:,i].max()
        
        data[node_type].x = data[node_type].x.to(torch.float32)
        
    
    
    train_data, val_data, test_data = transform(data)
    #train_data = sort_edges(train_data)
    #val_data = sort_edges(val_data)
    #test_data = sort_edges(test_data)
    if remove_text_attr:
        train_data = preprocess(train_data)
        val_data = preprocess(val_data)
        test_data = preprocess(test_data)
    
    return train_data, val_data, test_data



# COMMAND ----------


from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.loader import HGTLoader
from torch_geometric.sampler import NegativeSampling

def get_hgt_linkloader(data, target_edge, batch_size, is_training, sampling_mode, neg_ratio, num_neighbors_hgtloader, num_workers, prefetch_factor, pin_memory):
    # first sample some edges in linkNeighborLoader
    # use the nodes of the sampled edges to sample from hgt loader
    
    
    num_neighbors_linkloader = [0]
    #for edge_type in data.edge_types:
    #    num_neighbors_linkloader[edge_type] = [0,0]
    
    negative_sampling = NegativeSampling(
        mode=sampling_mode, # binary or triplet
        amount=neg_ratio  # ratio, like Graphsage # 10
        #weight=  # "Probabilities" of nodes to be sampled: Node degree follows power law distribution
        )
    
    if sampling_mode == 'triplet':
        data[target_edge].edge_label = None
        

    linkNeighborLoader = LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors_linkloader,
            edge_label_index=(target_edge, data[target_edge].edge_label_index), # if (edge, None), None means all edges are considered
        
            neg_sampling=negative_sampling, # adds negative samples
            batch_size=batch_size,
            shuffle=is_training, #is_training
            subgraph_type='directional', # contains only sampled edges
            #drop_last=True,
            num_workers=num_workers,
            #disjoint=True # sampled seed node creates its own, disjoint from the rest, subgraph, will add "batch vector" to loader output
        
            #num_workers=2,
            #prefetch_factor=2
            is_sorted = False,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
    )
   
   
    def get_hgt(data, input_nodetype, input_mask):
        return next(iter(HGTLoader(
                data,
                # Sample 512 nodes per type and per iteration for 4 iterations
                num_samples=num_neighbors_hgtloader,
                batch_size=input_mask.shape[0],
                input_nodes=(input_nodetype, input_mask),
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )))
        
    
    def add_self_loops(data):
        for node_type in data.node_types:
            data[node_type, 'self_loop', node_type].edge_index = torch.arange(data[node_type].num_nodes).repeat(2,1)
        return data 

            
    def get_hgt_with_selfloops(loader):
        
        
        for batch in loader:   
            if sampling_mode=='triplet':      
                # original edge_label_index from the whole data object
                unmapped_batchids = torch.cat((batch[target_edge[0]].src_index,batch[target_edge[2]].dst_pos_index, batch[target_edge[2]].dst_neg_index.flatten()))
                original_node_ids = batch[target_edge[0]].n_id[unmapped_batchids]
                original_edge_label_nodes = torch.LongTensor(original_node_ids.unique())

                #remapping or sorting is not needed, since nodes are sorted, also in the htg batch, the edges will be the same
                src = batch[target_edge[0]].src_index.unsqueeze(0)
                src_total = src
                for i in range(neg_ratio):
                    src_total = torch.cat((src_total,src), dim=1)
                dst = torch.cat((batch[target_edge[2]].dst_pos_index, batch[target_edge[2]].dst_neg_index.flatten()),dim=0).unsqueeze(0)
                
                local_edge_label_index = torch.cat((src_total, dst),dim=0)
                edge_label = torch.cat((torch.ones(batch[target_edge[2]].dst_pos_index.shape[0]), torch.zeros(batch[target_edge[2]].dst_neg_index.flatten().shape[0])))
                
            elif sampling_mode=='binary':
                unmapped_batchids = batch[target_edge].edge_label_index.flatten()
                original_node_ids = batch[target_edge[0]].n_id[unmapped_batchids]
                original_edge_label_nodes = torch.LongTensor(original_node_ids.unique())

            else:
                raise Exception('binary or triplet sampling mode')
                
                
            hgt_batch = get_hgt(data, target_edge[0], original_edge_label_nodes) # 0,1,3,4,5,6,7,8,9,
          
            if sampling_mode=='triplet':
                
                # return message passing edges, and supervision edges/labels, ignore labels/label_indices in the message passing edges
                yield add_self_loops(hgt_batch), local_edge_label_index, edge_label, batch[target_edge].input_id, original_node_ids
            else: # sampling_mode=='binary':
                # return message passing edges, and supervision edges/labels, ignore labels/label_indices in the message passing edges, as well as original edge indices
                yield add_self_loops(hgt_batch), batch[target_edge].edge_label_index, batch[target_edge].edge_label, batch[target_edge].input_id, original_node_ids
    
    def get_hgt_2types_with_selfloops(loader):
        for batch in loader:
            if sampling_mode=='triplet':   
                original_src_nodes = batch[target_edge[0]].n_id[batch[target_edge[0]].src_index]
                original_edge_label_nodes_class1 = torch.LongTensor(original_src_nodes.unique())
                
                original_dst_nodes = batch[target_edge[2]].n_id[torch.cat((batch[target_edge[2]].dst_pos_index, batch[target_edge[2]].dst_neg_index.flatten()))]
                original_edge_label_nodes_class2 = torch.LongTensor(original_dst_nodes.unique())
                
                
                src = batch[target_edge[0]].src_index.unsqueeze(0)
                src_total = src
                for i in range(neg_ratio):
                    src_total = torch.cat((src_total,src), dim=1)
                
                dst = torch.cat((batch[target_edge[2]].dst_pos_index, batch[target_edge[2]].dst_neg_index.flatten()),dim=0).unsqueeze(0)
                
                local_edge_label_index = torch.cat((src_total, dst),dim=0)
                edge_label = torch.cat((torch.ones(batch[target_edge[2]].dst_pos_index.shape[0]), torch.zeros(batch[target_edge[2]].dst_neg_index.flatten().shape[0])))

            elif sampling_mode=='binary':
                original_src_nodes = batch[target_edge[0]].n_id[batch[target_edge].edge_label_index[0,:]]
                original_edge_label_nodes_class1 = original_src_nodes.unique()
                original_dst_nodes = batch[target_edge[2]].n_id[batch[target_edge].edge_label_index[1,:]]
                original_edge_label_nodes_class2 = original_dst_nodes.unique()

            else:
                raise Exception('binary or triplet sampling mode')

            # batch the start and end supervision nodes separately
            hgt_batch1 = get_hgt(data, target_edge[0], original_edge_label_nodes_class1)
            hgt_batch2 = get_hgt(data, target_edge[2], original_edge_label_nodes_class2)
            
            
            # ** We dont need to remove any edges ** since the supervision edges wont be sampled by hgt
            if sampling_mode=='triplet':
                yield add_self_loops(hgt_batch1), add_self_loops(hgt_batch2), local_edge_label_index, edge_label, batch[target_edge].input_id, original_src_nodes, original_dst_nodes
            else: # sampling_mode=='binary':
                # we can access the corresponding nodes of edge_label_index[0,:] in hgt_batch1[target_edge[0]], those of [1,:] in hgt_batch2...
                yield add_self_loops(hgt_batch1), add_self_loops(hgt_batch2), batch[target_edge].edge_label_index, batch[target_edge].edge_label, batch[target_edge].input_id, original_src_nodes, original_dst_nodes

        
    if target_edge[0] == target_edge[2]:
        # same edge type, only need to sample once
        return get_hgt_with_selfloops(linkNeighborLoader)
    else:
        return get_hgt_2types_with_selfloops(linkNeighborLoader)


import random

def get_minibatch_count(data, batch_size):
    batches = []
    for edge_type in data.edge_types:
        if edge_type[1].startswith('rev_'):
            continue
        batches.extend([edge_type for _ in range((data[edge_type].edge_label_index.shape[1]+batch_size)//batch_size)])
        
    return len(batches)

def get_single_minibatch_count(data, batch_size, edge_type):
    return (data[edge_type].edge_label_index.shape[1]+batch_size)//batch_size

from copy import deepcopy
def uniform_hgt_sampler(data, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory):

    # return batches from all edgetypes with each "edge" being drawn uniformly at random (but we translate the probabilities to batches), last batches of each edge type may be smaller than batch_size
    batches = []
    loaders = {}
    # only the non-reverse edge types for now
    for edge_type in data.edge_types:
        if edge_type[1].startswith('rev_'):
            continue
        batches.extend([edge_type for _ in range((data[edge_type].edge_label_index.shape[1]+batch_size)//batch_size)])
        loaders[edge_type]=get_hgt_linkloader(data, edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)
        
    random.seed(14)
    random.shuffle(batches)
    # set a random random seed again (may affect creating the loaders later for a second epoch)
    random.seed()
    
    print('total batches:', len(batches))
    
    for target_edge_type in batches:
        if target_edge_type[0] == target_edge_type[2]:
            same_nodetype = True
        else:
            same_nodetype = False
        try:
            minibatch = next(loaders[target_edge_type])
        except StopIteration: # "reinit" iterator
            loaders[target_edge_type] = get_hgt_linkloader(data, target_edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)#iter(loaders[target_edge_type])
            minibatch = next(loaders[target_edge_type])
            pass

        yield same_nodetype, target_edge_type, minibatch
        
def sampler_for_init(data, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory):
    batchcount=[]
    batches=[]
    loaders = {}
    for edge_type in data.edge_types:
        if edge_type[1].startswith('rev_'):
            continue
        batchcount.extend([edge_type for _ in range((data[edge_type].edge_label_index.shape[1]+batch_size)//batch_size)])
        batches.append(edge_type)
        loaders[edge_type]=get_hgt_linkloader(data, edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)
        
    print('total batches:', len(batchcount))

    
    # set a random random seed again (may affect creating the loaders later for a second epoch)
    random.seed()
    
    for target_edge_type in batches:
        if target_edge_type[0] == target_edge_type[2]:
            same_nodetype = True
        else:
            same_nodetype = False
        try:
            minibatch = next(loaders[target_edge_type])
        except StopIteration: # "reinit" iterator
            loaders[target_edge_type] = get_hgt_linkloader(data, target_edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)#iter(loaders[target_edge_type])
            minibatch = next(loaders[target_edge_type])
            pass
            
        yield same_nodetype, target_edge_type, minibatch

def equal_edgeweight_hgt_sampler(data, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory):
    batchcount=[]
    batches=[]
    loaders = {}
    for edge_type in data.edge_types:
        if edge_type[1].startswith('rev_'):
            continue
        batchcount.extend([edge_type for _ in range((data[edge_type].edge_label_index.shape[1]+batch_size)//batch_size)])
        batches.append(edge_type)
        loaders[edge_type]=get_hgt_linkloader(data, edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)
        
    print('total batches:', len(batchcount))

    batches = random.choices(batches, k=len(batchcount)) 
    # set a random random seed again (may affect creating the loaders later for a second epoch)
    random.seed()
    
   
    
    for target_edge_type in batches:
        if target_edge_type[0] == target_edge_type[2]:
            same_nodetype = True
        else:
            same_nodetype = False
        try:
            minibatch = next(loaders[target_edge_type])
        except StopIteration: # "reinit" iterator
            loaders[target_edge_type] = get_hgt_linkloader(data, target_edge_type, batch_size, is_training, sampling_mode, neg_sampling_ratio, num_neighbors, num_workers, prefetch_factor, pin_memory)#iter(loaders[target_edge_type])
            minibatch = next(loaders[target_edge_type])
            pass
            
        yield same_nodetype, target_edge_type, minibatch

if __name__ == '__main__':
    import datetime
    import time

    # num neighbors: sample 25 one hop neighbors and for each one hop neighbor again 10 neighbors

    batch_size = 32
    num_relationships = len(train_data.edge_types)
    one_hop_neighbors = (25 * batch_size)//num_relationships # per relationship type
    two_hop_neighbors = (25 * 10 * batch_size)//num_relationships # per relationship type
    num_neighbors = [one_hop_neighbors, two_hop_neighbors]
    print('num_neighbors', num_neighbors)

    sampler = uniform_hgt_sampler(train_data, batch_size, True, 'binary', 1, num_neighbors, num_workers=0)
    start = datetime.datetime.now()
    print(start)
    print()
    for i,(same_nodetype, target_edge_type, batch) in enumerate(sampler):
        
        # batching is different depending on if node types in edge are same or different
        edge_type = batch[-1]
        if same_nodetype:
            minibatch, edge_label_index, edge_label, input_edge_ids = batch
            print(minibatch)
        else:
            minibatchpart1, minibatchpart2, edge_label_index, edge_label, input_edge_id = batch
            print(minibatchpart1)
            
        print(i,target_edge_type)
        
        break
        time.sleep(5)
        
    end = datetime.datetime.now()
    print()
    print(end-start)


