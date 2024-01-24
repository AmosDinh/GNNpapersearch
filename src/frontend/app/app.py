import streamlit as st
from streamlit import session_state
import pickle
from weaviate import Client
from weaviate.util import generate_uuid5
import torch

# import json
import pandas as pd
import numpy as np


def generate_uuid(class_name: str, node_id: str) -> str:
    return generate_uuid5((node_id, class_name))


def get_weaviate_client(db_url: str = "http://Weaviate:8080") -> Client:
# def get_weaviate_client(db_url: str = "http://localhost:8081") -> Client:
    return Client(db_url)


def get_by_idx(node_type: str, idx: int):
    indice = torch.tensor(idx)
    uuid = generate_uuid(node_type, indice)
    client = get_weaviate_client()

    data_object = client.data_object.get_by_id(
        uuid,
        class_name=node_type,
    )
    return data_object


DATA_OB_DICT = {
    "Paper": "name",
    "Author": "name",
    "Category": "name",
    "Word": "name",
    "Journal": "name",
}


def search_keyword(keyword: str, node_type: str, property: str):
    client = get_weaviate_client()
    response = (
        client.query.get(node_type.capitalize(), property)
        .with_bm25(query=keyword)
        .with_additional("vector")
        .with_limit(15)
        .do()
    )
    result = response["data"]["Get"][node_type.capitalize()]
    return result


def get_near_vector(vector: list, node_type: str, property: str):
    client = get_weaviate_client()
    response = (
        client.query.get(node_type.capitalize(), property)
        .with_near_vector({"vector": vector})
        .with_limit(15)
        .with_additional(["distance"])
        .do()
    )
    return response["data"]["Get"][node_type.capitalize()]


# @st.cache_data
def process_text(text, option1):
    try:
        # This is where you would put your text processing.
        # For the sake of this example we'll just return the text and the selected options.
        property = DATA_OB_DICT[option1]
        result = search_keyword(text, option1, property)
        df = [{property.capitalize(): i[property] for _ in i.items()} for i in result]
        df = pd.DataFrame(df)
        return df, result

    except IndexError:
        st.write("No results found")


def prepare_vector(vector: dict, option1: str, option2: str):
    vector = torch.tensor(vector["_additional"]["vector"])

    if option1 != option2:
        if option2 == "Paper":
            relationship_vector = torch.load(
                f"relation_embeddings/{option2.lower()}_{option1.lower()}.pt"
            )
            vector = vector - relationship_vector["embedding"]
        elif option1 == "Paper":
            relationship_vector = torch.load(
                f"relation_embeddings/{option1.lower()}_{option2.lower()}.pt"
            )
            vector = vector + relationship_vector["embedding"]
        else:
            relationship_vector = torch.load(
                f"relation_embeddings/paper_{option1.lower()}.pt"
            )
            vector = vector - relationship_vector["embedding"]

            relationship_vector = torch.load(
                f"relation_embeddings/paper_{option2.lower()}.pt"
            )
            vector = vector + relationship_vector["embedding"]
    property = DATA_OB_DICT[option2]
    result = [
        {property.capitalize(): i[property] for _ in i.items()}
        for i in get_near_vector(vector, option2, property)
    ]
    result = pd.DataFrame(result)
    return result


if __name__ == "__main__":

    def dataframe_with_selections(df):
        df_with_selections = df.copy()
        df_with_selections.insert(0, "Select", False)
        edited_df = st.data_editor(df_with_selections, hide_index=True, use_container_width=True)
        selected_indices = list(np.where(edited_df.Select)[0])
        return selected_indices

    st.set_page_config(
        page_title="Database Queries", page_icon=":memo:"
    )  # , layout="wide")

    option_list = ["Paper", "Author", "Category", "Word", "Journal"]

    st.title("Database Queries")

    # Text input
    text = st.text_input("Enter some text")

    # Dropdowns
    option1 = st.selectbox("From", option_list)
    option2 = st.selectbox("Search in", option_list)

    st.session_state.s = False

    # Button
    try:
        df, result = process_text(text, option1)
        selection = dataframe_with_selections(df)
    except:
        pass

    try:
        idx = selection[0]
        st.write("Your selection:")
        result1 = prepare_vector(result[idx], option1, option2)
        st.dataframe(result1, use_container_width=True)
    except:
        pass
