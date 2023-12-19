from weaviate import Client
from weaviate.util import generate_uuid5
from typing import List, Dict
from dataclasses import dataclass
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import yaml
import torch_geometric.transforms as T


@dataclass
class NodeObject:
    class_name: str
    uuid: str
    data_object: Dict[str, str]
    vector: List[float]


@dataclass
class Reference:
    from_object_uuid: str
    from_object_class_name: str
    from_property_name: str
    to_object_uuid: str
    to_object_class_name: str


def generate_uuid(class_name: str, node_id: str) -> str:
    return generate_uuid5((node_id, class_name))


def get_weaviate_client(db_url: str = "http://localhost:8081") -> Client:
    return Client(db_url)


def generate_object(class_name, node_id, node_name, node_embedding):
    return NodeObject(
        class_name="Node",
        uuid=generate_uuid(class_name, node_id),
        data_object={"name": node_name, "type": class_name},
        vector=node_embedding,
    )


def generate_reference_object(
    from_uuid, to_uuid, from_class_name, to_class_name, edge_name
):
    return Reference(
        from_object_uuid=from_uuid,
        from_object_class_name=from_class_name,
        from_property_name=edge_name,
        to_object_uuid=to_uuid,
        to_object_class_name=to_class_name,
    )


def upload_batch(
    client: Client,
    node_objects: List[NodeObject] = [],
    edge_objects: List[Reference] = [],
) -> None:
    with client.batch as batch:
        batch.configure(
            batch_size=64,
            num_workers=4,
            dynamic=True,  # dynamically update the `batch_size` based on import speed
            # callback=f_
        )
        for data_object in node_objects:
            batch.add_data_object(**data_object.__dict__)

        for reference in edge_objects:
            batch.add_reference(**reference.__dict__)


def delete_schema(client: Client):
    schema = client.schema.get()

    # Status READONLY is set when disk is over a certain limit, can bespecified in config
    for _class in schema["classes"]:
        # need to set status to ready on all shards, so we can delete
        client.schema.update_class_shard(
            class_name=_class["class"],
            status="READY",
        )
    client.schema.delete_all()


def create_schema(client: Client, schema_yaml: str):
    with Path(schema_yaml).open() as stream:
        schema = yaml.safe_load(stream)

    for class_obj in schema["classes"]:
        print(class_obj)
        client.schema.create_class(class_obj)

    # add cross references
    for cross_ref in schema["cross_refs"]:
        client.schema.property.create(cross_ref["forClass"], cross_ref["property"])


def fill_db(
    model: torch.nn.Module,
    data: HeteroData,
    node_type_and_node_index_to_name_mappings: Dict[str, Dict[int, str]],
):
    client = get_weaviate_client()
    schema_path = "../../schema.yml"
    delete_schema(client)
    create_schema(client, schema_path)
    print("created schema")
