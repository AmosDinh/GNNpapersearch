from weaviate import Client
from weaviate.util import generate_uuid5
from torch import load, Tensor
from pathlib import Path
import yaml
import pickle
from tqdm.auto import tqdm


def generate_uuid(class_name: str, node_id: str) -> str:
    return generate_uuid5((node_id, class_name))


def get_weaviate_client(db_url: str = "http://localhost:8081") -> Client:
    return Client(db_url)


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


def generate_object(
    class_name: str, node_id: int, data_object: dict, node_embedding: Tensor
):
    return dict(
        class_name=class_name,
        uuid=generate_uuid(class_name, node_id),
        data_object={**data_object, "type": class_name},
        vector=node_embedding,
    )


def get_all_nodes(node_type: str, embeddings: dict, names: list) -> list:
    return [
        [
            generate_object(node_type, indice, names[indice], embedding)
            for embedding, indice in zip(embeddings, indices)
        ]
        for embeddings, indices in zip(embeddings["embeddings"], embeddings["indices"])
    ]


FOLDER = "results/results"
NODE_TYPES = [
    "paper",
    "author",
    "category",
    "word",
    "journal"
]

if __name__ == "__main__":
    client = get_weaviate_client()
    delete_schema(client)
    create_schema(client, "schema.yml")
    client.batch.configure(
        batch_size=64,
        num_workers=10,
        dynamic=True,  # dynamically update the `batch_size` based on import speed)
    )
    for node_type in NODE_TYPES:
        with tqdm(range(len(list(Path(FOLDER).glob(f"embeddings_{node_type}*"))))) as tepoch:
            for counter in tepoch:
                tepoch.set_postfix({"node_type": node_type, "counter": counter})
                path = f"{FOLDER}/embeddings_{node_type}_{counter}.pt"
                embeddings = load(path)
                with open(f"data/{node_type}_data.pkl", "rb") as f:
                    names = pickle.load(f)
                items = get_all_nodes(node_type.capitalize(), embeddings, names)

                with client.batch as batch:
                    [[batch.add_data_object(**data_object) for data_object in x] for x in items]
