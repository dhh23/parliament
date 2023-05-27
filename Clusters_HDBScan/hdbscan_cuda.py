##########################################################################################
# Digital Humanities Hackathon 2023 - Parliament
# Authors: Vadym Kuzyak, Pontus Hedlund
# This script is used to run the HDBSCAN clustering algorithm on the embeddings using cuML
#
# Requirements: GPU with CUDA support, cuML
# Usage: python hdbscan_cuda.py
# Output: A csv file with the cluster labels for each speech
#
# Note: For installation of cuML, see https://rapids.ai/start.html
##########################################################################################
import csv
from cuml.cluster import HDBSCAN
import numpy as np
import pickle
import time

print("DHH23 Parliament - HDBSCAN Clustering with cuML")
print("Authors: Vadym Kuzyak, Pontus Hedlund")
print("-------------------------------------------------")
print("Starting")
# Load embeddings from pickle file

data_filename = "ParlaMint_GB_commons_aggregated_embeddings.pkl"
with open(data_filename, "rb") as openfile:
    print(f"Opened file '{data_filename}'")
    # dictionary speech_id -> embeddings
    embeddings = pickle.load(openfile)

print("Loaded embeddings")

# ------------------- PARAMETERS -------------------
# MINIMUM CLUSTER SIZE FOR HDBSCAN
MIN_CLUSTER_SIZE = 5

clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)

print("Starting clustering")
# Time it
start = time.time()
cluster_labels = clusterer.fit_predict(np.array(list(embeddings.values())))
end = time.time()
# Print the time it took in seconds with a precision of 1 decimals
print(f"Clustering finished in: {round(end - start, 1)} seconds")

keys_list = list(embeddings.keys())
print(f"Number of Clusters generated: {len(set(cluster_labels))}")
filename = f"minclustersize_{MIN_CLUSTER_SIZE}.csv"
print(f"Writing results to '{filename}'")
f = open(filename, "w", newline="")
writer = csv.writer(f)
fieldnames = ["ID", "cluster_ID"]
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
for i in range(len(keys_list)):
    writer.writerow({"ID": keys_list[i], "cluster_ID": cluster_labels[i]})

f.close()
print("Done")
