import yaml
from tqdm.auto import tqdm
import numpy as np
import weaviate
import pathlib
from weaviate.util import generate_uuid5
from models.WeightedSkillSAGE import skillsage_388_prelu_batchnorm_edgeweight
from models.get_entity_embedding import get_entity_embedding
from models.get_node_neighbors import get_node_neighbors
import pathlib
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
from copy import deepcopy


def submit_weaviate_batch(batch, data_objects=[], references=[]):
    for data_object in data_objects:
        batch.add_data_object(**data_object)
        
    for reference in references:
        batch.add_reference(**reference)

def generate_uuid(class_name, node_id: str):
    return generate_uuid5((node_id, class_name))


def get_weaviate_client(db_url):
    # ec2-3-125-226-65.eu-central-1.compute.amazonaws.com localhost
    return weaviate.Client(db_url)








def generate_object(class_name, node_id, node_name, node_embedding):
    return {
            'class_name': 'Node',
            'uuid': generate_uuid(class_name, node_id),
            'data_object': {
                'name': node_name,
                'type': class_name
            },
            'vector': node_embedding
        }

def generate_reference_object(from_uuid, to_uuid, from_class_name, to_class_name, edge_name):
    ref = {
            'from_object_uuid': from_uuid,
            'from_object_class_name': from_class_name,
            'from_property_name': edge_name,
            'to_object_uuid': to_uuid,
            'to_object_class_name': to_class_name,
        }
    return ref
    

from typing import List, Dict, Tuple

def upload_batch(client, node_objects, edge_objects):
    with client.batch as batch:
        batch.configure(
            batch_size=64,
            num_workers=4,
            dynamic=True,  # dynamically update the `batch_size` based on import speed
            #callback=f_
        )
        submit_weaviate_batch(batch, node_objects, edge_objects)
        
def fill_db(model:torch.nn.Module, data:HeteroData, node_type_and_node_index_to_name_mappings:Dict[str,Dict[int,str]]): 
    client = get_weaviate_client('http://localhost:8081')
    schema_path = 'embedding_db_schema.yml'
    delete_schema(client)
    create_schema(client, schema_path)
    print('created schema')
    
    for node_type, index_mapping in node_type_and_node_index_to_name_mappings.items():
        print(data[node_type].x.shape[0],len(list(index_mapping.keys())))
        assert data[node_type].x.shape[0] == len(list(index_mapping.keys())), f'index mapping for {node_type} must be as big as supplied data object for that node type'
    
    # remove isolated nodes, for those we can not compute embeddings
    # save the names before that
    
    for node_type, index_mapping in node_type_and_node_index_to_name_mappings.items():
        data[node_type].n_id = torch.arange(len(list(index_mapping.keys())))
        # data[node_type].x = np.concatenate([data[node_type].x, np.array(names).reshape(-1,1)], axis=1) 
        
    data = T.RemoveIsolatedNodes()(data)
    
    temp_mapping = {}
    for node_type, old_mapping in node_type_and_node_index_to_name_mappings.items():
        index_mapping = {}
        for i in range(data[node_type].n_id.shape[0]):
            index_mapping[i] = old_mapping[data[node_type].n_id[i].item()]
            
        temp_mapping[node_type] = index_mapping
    node_type_and_node_index_to_name_mappings = temp_mapping
    
    # create all the nodes
    batch_size_n_nodes = 1000
    for node_type, index_mapping in node_type_and_node_index_to_name_mappings.items():
        print(f'Add nodes: {node_type}')
        mapping = list(zip(index_mapping.keys(), index_mapping.values()))
        while len(mapping):
            print(f'Add nodes: {node_type}, {len(mapping)}')
            node_objects = []
            batch_mapping = mapping[:batch_size_n_nodes]
            mapping = mapping[batch_size_n_nodes:]
            
            node_embeddings = get_entity_embedding(model, data, node_type=node_type, num_neighbors=[5,4], node_ids=[i for i, name in batch_mapping])
            for i, (node_id, name) in enumerate(batch_mapping):
                node_objects.append(generate_object(node_type, node_id, name, node_embeddings[i]))

            upload_batch(client, node_objects=node_objects, edge_objects=[])
            
    # add all the edges
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index.T
        edge_objects = []
        for i in tqdm(range(edge_index.shape[0]), desc=f'Add edges: {str(edge_type)}'):

            edge = edge_index[i]
            edge_objects.append(
                generate_reference_object(
                    # from_uuid=generate_uuid(class_name=edge_type[0], node_id=edge[0]),
                    # to_uuid=generate_uuid(class_name=edge_type[2], node_id=edge[1]),
                    # edge_name=edge_type[1], from_class_name=edge_type[0], to_class_name=edge_type[2]
                    from_uuid=generate_uuid(class_name=edge_type[0], node_id=edge[0].item()), # class names have to be job and skill, to get unique uuid
                    to_uuid=generate_uuid(class_name=edge_type[2], node_id=edge[1].item()),
                    edge_name=edge_type[1].lower(), from_class_name='Node', to_class_name='Node'
                    )
                )
        
        upload_batch(client, node_objects=[], edge_objects=edge_objects)
         
   
    
    
def delete_schema(client):

    schema = client.schema.get()

    # Status READONLY is set when disk is over a certain limit, can bespecified in config
    for _class in schema['classes']:
        # need to set status to ready on all shards, so we can delete
        client.schema.update_class_shard(
            class_name=_class['class'],
            status="READY",
        )
    client.schema.delete_all()
    
def create_schema(client, schema_yaml):

    with pathlib.Path(schema_yaml).open() as stream:
        schema = yaml.safe_load(stream)


    for class_obj in schema['classes']:
        print(class_obj)
        client.schema.create_class(class_obj)

    # add cross references
    for cross_ref in schema['cross_refs']:
        client.schema.property.create(
            cross_ref["forClass"], cross_ref["property"])





def main():
    filename = 'Job_Skill_HeteroData_withdupes_fulldataset_v1.pt' # 
    
    data = HeteroData.from_dict(torch.load('./'+filename))



    model = skillsage_388_prelu_batchnorm_edgeweight()
    checkpoint = torch.load('runs/skillsage_388_prelu_batchnorm_edgeweight_checkpoints/checkpoint_ep3.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    node_mappings = torch.load('Job_Skill_HeteroData_name_mappings_withdupes_fulldataset_v1.pt')
    
    # map titles to the normalized index
    normindex_to_jobtitle = {}
    for key, value in node_mappings['inverted_jobmapping'].items():
        normindex_to_jobtitle[key] = node_mappings['jobmapping_index_to_title_alttile'][value]
        
    normindex_to_skillname = node_mappings['inverted_skillmapping']
    
    
    node_type_and_node_index_to_name_mappings = {}
    node_type_and_node_index_to_name_mappings['Job']= normindex_to_jobtitle
    node_type_and_node_index_to_name_mappings['Skill']= normindex_to_skillname
    fill_db(model, data, node_type_and_node_index_to_name_mappings)
    

if __name__=='__main__':
    print('need to use torch 2.1.0 and cu121! for skillsage 388 prelu batchnorm edgeweight .pt')
    main()
    # start docker, then
    # docker compose up -d
    # to start the weaviate instance
    
    
        
   