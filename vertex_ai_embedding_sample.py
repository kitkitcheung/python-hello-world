import math
import pandas as pd
from google.cloud import bigquery
from typing import Any, Generator, List, Tuple, Optional
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm.auto import tqdm
import random
import tempfile
from pathlib import Path
import gc
import json


# Load the "Vertex AI Embeddings for Text" model
from vertexai.preview.language_models import TextEmbeddingModel

PROJECT_ID = "kitkit-vertex-ai" 
REGION = "us-central1"  
BUCKET_URI = f"gs://vector-storage-{PROJECT_ID}-{REGION}"

client = bigquery.Client(project=PROJECT_ID)
QUERY_TEMPLATE = """
        SELECT distinct q.id, q.title, q.body
        FROM (SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` where Score>0 ORDER BY View_Count desc) AS q 
        LIMIT {limit} OFFSET {offset};
        """

def query_bigquery_chunks(
    max_rows: int, rows_per_chunk: int, start_chunk: int = 0
) -> Generator[pd.DataFrame, Any, None]:
    for offset in range(start_chunk, max_rows, rows_per_chunk):
        query = QUERY_TEMPLATE.format(limit=rows_per_chunk, offset=offset)
        query_job = client.query(query)
        rows = query_job.result()
        df = rows.to_dataframe()
        df["title_with_body"] = df.title + "\n" + df.body
        yield df

# Get a dataframe of 1000 rows for demonstration purposes
df = next(query_bigquery_chunks(max_rows=1000, rows_per_chunk=1000))

# Examine the data
print(df.head())

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# Define an embedding method that uses the model
def encode_texts_to_embeddings(sentences: List[str]) -> List[Optional[List[float]]]:
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception as error:
        # print("An error occurred:", type(error).__name__) 
        # print("An error occurred:", error) 
        return [None for _ in range(len(sentences))]
    
# Generator function to yield batches of sentences
def generate_batches(
    sentences: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]


def encode_text_to_embedding_batched(
    sentences: List[str], api_calls_per_second: int = 1, batch_size: int = 5
) -> Tuple[List[bool], np.ndarray]:

    embeddings_list: List[List[float]] = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total=math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]

    try:
      embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
      )
    except ValueError:
      embeddings_list_successful  = np.empty()
    return is_successful, embeddings_list_successful

# Encode a subset of questions for validation
questions = df.title.tolist()[:500]
is_successful, question_embeddings = encode_text_to_embedding_batched(
    sentences=df.title.tolist()[:500]
)

# Filter for successfully embedded sentences
questions = np.array(questions)[is_successful]

DIMENSIONS = len(question_embeddings[0])

print(DIMENSIONS)

question_index = random.randint(0, 99)

print(f"Query question = {questions[question_index]}")

# Get similarity scores for each embedding by using dot-product.
scores = np.dot(question_embeddings[question_index], question_embeddings.T)

# Print top 20 matches
for index, (question, score) in enumerate(
    sorted(zip(questions, scores), key=lambda x: x[1], reverse=True)[:20]
):
    print(f"\t{index}: {question}: {score}")

# Create temporary file to write embeddings to
embeddings_file_path = Path(tempfile.mkdtemp())

print(f"Embeddings directory: {embeddings_file_path}")


BQ_NUM_ROWS = 50000
BQ_CHUNK_SIZE = 500
BQ_NUM_CHUNKS = math.ceil(BQ_NUM_ROWS / BQ_CHUNK_SIZE)

START_CHUNK = 0

# Create a rate limit of 300 requests per minute. Adjust this depending on your quota.
API_CALLS_PER_SECOND = 60 / 60
# According to the docs, each request can process 5 instances per request
ITEMS_PER_REQUEST = 5

# Loop through each generated dataframe, convert
for i, df in tqdm(
    enumerate(
        query_bigquery_chunks(
            max_rows=BQ_NUM_ROWS, rows_per_chunk=BQ_CHUNK_SIZE, start_chunk=START_CHUNK
        )
    ),
    total=BQ_NUM_CHUNKS - START_CHUNK,
    position=-1,
    desc="Chunk of rows from BigQuery",
):
    # Create a unique output file for each chunk
    chunk_path = embeddings_file_path.joinpath(
        f"{embeddings_file_path.stem}_{i+START_CHUNK}.json"
    )
    with open(chunk_path, "a") as f:
        id_chunk = df.id

        # Convert batch to embeddings
        is_successful, question_chunk_embeddings = encode_text_to_embedding_batched(
            sentences=df.title_with_body,
            api_calls_per_second=API_CALLS_PER_SECOND,
            batch_size=ITEMS_PER_REQUEST,
        )

        # Append to file
        embeddings_formatted = [
            json.dumps(
                {
                    "id": str(id),
                    "embedding": [str(value) for value in embedding],
                }
            )
            + "\n"
            for id, embedding in zip(id_chunk[is_successful], question_chunk_embeddings)
        ]
        f.writelines(embeddings_formatted)

        # Delete the DataFrame and any other large data structures
        del df
        gc.collect()

remote_folder = f"{BUCKET_URI}/{embeddings_file_path.stem}/"

print(remote_folder)