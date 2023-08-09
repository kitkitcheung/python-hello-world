from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from typing import List, Optional



PROJECT_ID = "kitkit-vertex-ai" 
REGION = "us-central1"  
BUCKET_URI = f"gs://vector-storage-{PROJECT_ID}-{REGION}"
DISPLAY_NAME = "stack_overflow"
DESCRIPTION = "question titles and bodies from stackoverflow"
BQ_NUM_ROWS = 50000
DIMENSIONS = 768

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

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

remote_folder="gs://vector-storage-kitkit-vertex-ai-us-central1/tmpgoi7oici/"

tree_ah_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=DISPLAY_NAME,
    contents_delta_uri=remote_folder,
    dimensions=DIMENSIONS,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=80,
    description=DESCRIPTION,
)

# Get the existing index
# tree_ah_index = aiplatform.MatchingEngineIndex(
#     index_name="8449518161039982592"
# )

# Using the resource name, you can retrieve an existing MatchingEngineIndex.
# INDEX_RESOURCE_NAME = tree_ah_index.resource_name
# tree_ah_index = aiplatform.MatchingEngineIndex(index_name=INDEX_RESOURCE_NAME)

my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=DISPLAY_NAME,
    description=DISPLAY_NAME,
    public_endpoint_enabled=True,
)

# Get Existing Index endpoint
# my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
#     index_endpoint_name='9005712715020238848'
# )

DEPLOYED_INDEX_ID = "deployed_index_id_unique"

# my_index_endpoint = my_index_endpoint.deploy_index(
#     index=tree_ah_index, deployed_index_id=DEPLOYED_INDEX_ID, min_replica_count=1, max_replica_count=1
# )

print(my_index_endpoint.deployed_indexes)

number_of_vectors = sum(
    aiplatform.MatchingEngineIndex(
        deployed_index.index
    )._gca_resource.index_stats.vectors_count
    for deployed_index in my_index_endpoint.deployed_indexes
)

print(f"Expected: {BQ_NUM_ROWS}, Actual: {number_of_vectors}")

test_embeddings = encode_texts_to_embeddings(sentences=["use Tensorflow"])

# Test query
NUM_NEIGHBOURS = 10

response = my_index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_INDEX_ID,
    queries=test_embeddings,
    num_neighbors=NUM_NEIGHBOURS,
)

print(response)

for match_index, neighbor in enumerate(response[0]):
    print(f"https://stackoverflow.com/questions/{neighbor.id}")

#Clean Up

# Force undeployment of indexes and delete endpoint
my_index_endpoint.delete(force=True)

# Delete indexes
tree_ah_index.delete()