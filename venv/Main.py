import streamlit as st
import time
from datetime import datetime
import sys
import pandas as pd
from google.cloud import aiplatform
from google.cloud import bigquery
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
import numpy as np




PROJECT_ID = "dark-caldron-414803"  
LOCATION = "us-central1"
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint("885911702871212032")
DEPLOYED_INDEX_ID = "embvs_tutorial_deployed_02261247"


aiplatform.init(project=PROJECT_ID, location=LOCATION)
vertexai.init(project=PROJECT_ID, location=LOCATION)

bq_client = bigquery.Client(project=PROJECT_ID)
QUERY_TEMPLATE = """
        select a.*, b.duration from `DummyDataset.Site_Detail` as a
        left outer join `DummyDataset.Duraion_Detail` as b
        ON a.quotation_No = b.quotation
        AND a.Room_Type = b.room;
        """
query = QUERY_TEMPLATE.format()
query_job = bq_client.query(query)
rows = query_job.result()
df = rows.to_dataframe()

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

BATCH_SIZE=5
def get_embeddings_wrapper(texts):
    embs = []
    for i in (range(0, len(texts), BATCH_SIZE)):
        time.sleep(1)  # to avoid the quota error
        result = model.get_embeddings(texts[i : i + BATCH_SIZE])
        embs = embs + [e.values for e in result]
    return embs



my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint("885911702871212032")
DEPLOYED_INDEX_ID = "embvs_tutorial_deployed_02261247"


prompt = st.chat_input("Say something")
if prompt:
    test_embeddings = get_embeddings_wrapper([prompt])

    response = my_index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_INDEX_ID,
    queries=test_embeddings,
    num_neighbors=5,
    )
    arr=[]
    i=0
    # show the result


    for idx, neighbor in enumerate(response[0]):
        id = np.int64(neighbor.id)
        arr.append([])
        similar = df.query("id == @id", engine="python")
        arr[i].append(similar.Location.values[0])
        arr[i].append(similar.Tier.values[0])
        arr[i].append(similar.Room_Type.values[0])
        arr[i].append(similar.Service.values[0])
        arr[i].append(similar.duration.values[0])
        i=i+1
        st.write(f"{similar.id.values[0]} {similar.Service.values[0]} {similar.Room_Type.values[0]} {similar.Tier.values[0]} {similar.Location.values[0]} {similar.duration.values[0]}")