#!/usr/bin/env python
import time
from collections import defaultdict

import numpy as np
import psycopg2
from opensearchpy import OpenSearch
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import AsIs, register_adapter
from sklearn.neighbors import NearestNeighbors

VECTOR_DIM = 2000
NUM_VECTORS = 1000
NUM_QUERIES = 500
K = 10  # Number of nearest neighbors to retrieve
ERASE_BEFORE_RUN = True

COMPARISON_PARAMS = {
    "pgvector_hnsw_vector_l2_ops": {
        "max_dimension": 2000,
        "table_name": "vectors",
        "type": "knn_vector",
        "method": {"engine": "vector", "space_type": "vector_l2_ops", "name": "hnsw"},
    },
    "pgvector_hnsw_halfvec_l2_ops": {
        "max_dimension": 4000,
        "table_name": "vectors",
        "type": "knn_vector",
        "method": {"engine": "halfvec", "space_type": "halfvec_l2_ops", "name": "hnsw"},
    },
    "pgvector_ivfflat_vector_l2_ops": {
        "max_dimension": 2000,
        "table_name": "vectors",
        "type": "knn_vector",
        "method": {"engine": "vector", "space_type": "vector_l2_ops", "name": "ivfflat"},
    },
    "pgvector_ivfflat_halfvec_l2_ops": {
        "max_dimension": 4000,
        "table_name": "vectors",
        "type": "knn_vector",
        "method": {"engine": "halfvec", "space_type": "halfvec_l2_ops", "name": "ivfflat"},
    },
    "os_lucene": {
        "max_dimension": 10000,
        "type": "knn_vector",
        "table_name": "vectors_lucene",
        "method": {"engine": "lucene", "space_type": "l2", "name": "hnsw"},
    },
    "os_nmslib": {
        "max_dimension": 10000,
        "table_name": "vectors_nmslib",
        "type": "knn_vector",
        "method": {"engine": "nmslib", "space_type": "l2", "name": "hnsw"},
    },
    "os_faiss": {
        "max_dimension": 10000,
        "table_name": "vectors_faiss",
        "type": "knn_vector",
        "method": {"engine": "faiss", "space_type": "l2", "name": "hnsw"},
    },
}
COMPARISONS_TO_RUN = [
    "pgvector_hnsw_vector_l2_ops",
    "pgvector_hnsw_halfvec_l2_ops",
    "pgvector_ivfflat_vector_l2_ops",
    "pgvector_ivfflat_halfvec_l2_ops",
    "os_lucene",
    "os_nmslib",
    "os_faiss",
]


def generate_vectors(_num_vectors, _vector_dim):
    return np.random.rand(_num_vectors, _vector_dim)


def addapt_numpy_array(numpy_array):
    return AsIs(str(numpy_array.tolist()))


register_adapter(np.ndarray, addapt_numpy_array)

pg_conn = psycopg2.connect(
    host="172.17.0.1",
    port=5432,
    database="vectordb",
    user="vectoruser",
    password="vectorpass",
)
pg_cur = pg_conn.cursor()
pg_cur.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
pg_cur.execute("SET SCHEMA 'public'")
pg_conn.commit()
register_vector(pg_conn)

os_client = OpenSearch(
    hosts=[{"host": "172.17.0.1", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

vectors = generate_vectors(NUM_VECTORS, VECTOR_DIM)
query_vectors = generate_vectors(NUM_QUERIES, VECTOR_DIM)

# Ground truth
nn = NearestNeighbors(n_neighbors=K, metric="euclidean")
nn.fit(vectors)
ground_truth = nn.kneighbors(query_vectors, return_distance=False)

# Leaderboard
leaderboard = defaultdict(lambda: {"store_time": 0, "query_time": 0, "precision": 0})

for comparison in COMPARISONS_TO_RUN:
    params = COMPARISON_PARAMS[comparison]

    print(f"\nRunning benchmark for: {comparison}")

    if VECTOR_DIM > params["max_dimension"]:
        print(
            f"Skipping {comparison} because vector size is greater than max dimension"
        )
        continue

    # Setup
    if comparison.startswith("pgvector"):
        if ERASE_BEFORE_RUN:
            pg_cur.execute(f"DROP TABLE IF EXISTS {params['table_name']}")
        pg_cur.execute(
            f"CREATE TABLE IF NOT EXISTS {params['table_name']} (id bigserial PRIMARY KEY, embedding {params['method']['engine']}({VECTOR_DIM}))"
        )
        pg_cur.execute(
            f"CREATE INDEX ON {params['table_name']} USING hnsw (embedding {params['method']['space_type']})"
        )
        pg_conn.commit()
    elif comparison.startswith("os"):
        index_body = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "embedding": {"type": params["type"], "dimension": VECTOR_DIM},
                }
            },
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
        }
        if ERASE_BEFORE_RUN and os_client.indices.exists(index=params["table_name"]):
            os_client.indices.delete(index=params["table_name"])
        os_client.indices.create(index=params["table_name"], body=index_body)
    else:
        print(f"WARN: Skipping {comparison} because it is not a supported comparison")
        break

    # Insert vectors
    start_time = time.time()
    if comparison.startswith("pgvector"):
        for i, vector in enumerate(vectors):
            pg_cur.execute(
                f"INSERT INTO {params['table_name']} (id, embedding) VALUES (%s, %s)",
                (i, vector),
            )
        pg_conn.commit()
    elif comparison.startswith("os"):
        for i, vector in enumerate(vectors):
            os_client.index(
                index=params["table_name"],
                body={"id": i, "embedding": vector.tolist()},
                id=str(i),
            )
    store_time = time.time() - start_time
    print(f"Time taken to store {NUM_VECTORS} vectors: {store_time:.2f} seconds")
    leaderboard[comparison]["store_time"] = store_time

    # Perform queries and measure precision
    start_time = time.time()
    correct_results = 0
    total_results = 0
    for i, query_vector in enumerate(query_vectors):
        if comparison.startswith("pgvector"):
            pg_cur.execute(
                f"SELECT id FROM {params['table_name']} ORDER BY embedding <-> %s::{params['method']['engine']} LIMIT {K}",
                (query_vector.tolist(),),
            )
            results = [row[0] for row in pg_cur.fetchall()]
        else:
            response = os_client.search(
                index=params["table_name"],
                body={
                    "size": K,
                    "query": {
                        "knn": {"embedding": {"vector": query_vector.tolist(), "k": K}}
                    },
                },
                request_timeout=600
            )
            results = [int(hit["_id"]) for hit in response["hits"]["hits"]]

        correct_results += len(set(results) & set(ground_truth[i]))
        total_results += K

    query_time = time.time() - start_time
    precision = correct_results / total_results
    print(f"Time taken for {NUM_QUERIES} queries: {query_time:.2f} seconds")
    print(f"Precision@{K}: {precision:.4f}")
    
    leaderboard[comparison]["query_time"] = query_time
    leaderboard[comparison]["precision"] = precision

# Close connections
pg_cur.close()
pg_conn.close()

# Print leaderboard
print("\nLeaderboard:")
print("{:<40} {:<15} {:<15} {:<15}".format("Comparison", "Store Time (s)", "Query Time (s)", "Precision"))
for comparison, results in sorted(leaderboard.items(), key=lambda x: x[1]['query_time']):
    print("{:<40} {:<15.2f} {:<15.2f} {:<15.4f}".format(
        comparison, 
        results['store_time'], 
        results['query_time'], 
        results['precision']
    ))
