import pickle
import hdbscan
from tqdm import tqdm
import numpy as np
import time
import csv

print("starting")
# Load embeddings from pickle file
with open("ParlaMint_GB_commons_aggregated_embeddings.pkl", "rb") as openfile:
    print("opened file")
    #dictionary speech_id -> embedings
    embeddings = pickle.load(openfile)

embeddings_array = np.array(list(embeddings.values()))

#speech_id, cluster_id

# First part: not aggregated
clusterer = hdbscan.HDBSCAN(min_cluster_size=2)

'''
Parameters
min_cluster_size: 100 (not aggregated), 5 (if we aggregate based on speaker)

remove non-NPs and cheer-persons, and short speeches before clustering (TODO!!!)
 
min_samples (number of neighbours which will be considered): 
--> test for around 1000 speeches

cluster_selection_epsilon (radius of neighbours which will be considered) -> leave at default 
'''
subset_size = 2000
subset_array = embeddings_array[:subset_size]

#subset_indices = np.random.choice(embeddings_array.shape[0], subset_size, replace=False)
#subset_embeddings = embeddings_array[subset_indices]

print("starting clustering")

cluster_labels = clusterer.fit_predict(subset_array)

#now write into csv
f = open('minclustersize_1.csv', 'w',newline='')
writer = csv.writer(f)
fieldnames = ['ID', 'cluster_ID']
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
    
for i in range(2000):
    writer.writerow({'ID': str(i+1), 'cluster_ID': cluster_labels[i]})

f.close()
