import ast
import uuid
import hashlib



import pandas as pd
from pymilvus import MilvusClient, connections, DataType


def create_collection(client):
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        metric_type="COSINE"
    )

    schema.add_field(field_name="created", datatype=DataType.VARCHAR, max_length=25)
    schema.add_field(field_name="uuid", datatype=DataType.VARCHAR,is_primary=True, max_length=50)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )

    client.create_collection(
        collection_name="image_embeddings",
        schema=schema,
        index_params=index_params
    )

def load_dataset(path, client):

    dataset = pd.read_csv(path)
    dataset['vector'] = dataset['vector'].apply(ast.literal_eval)
    dataset = dataset.to_dict(orient='records')

    client.insert(collection_name="image_embeddings", data=dataset)


if __name__ == '__main__':
    csv_path = './dataset.csv'
    connections.connect("default", host='localhost', port='19530')

    client = MilvusClient(
        uri="http://localhost:19530"
    )

    if client.has_collection(collection_name="image_embeddings"):
        client.drop_collection(collection_name="image_embeddings")
    create_collection(client)
    load_dataset(csv_path, client)

