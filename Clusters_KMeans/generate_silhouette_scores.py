import pickle
import faiss
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Load the data
with open('ParlaMint_GB_commons_embeddings_truncated.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)
embeddings_matrix = np.array(list(embeddings_dict.values()))

# Generate a random subset of the data
np.random.seed(42)
SUBSET_SIZE = 50000
subset_indices = np.random.choice(
    np.arange(embeddings_matrix.shape[0]), SUBSET_SIZE, replace=False
)
embeddings_matrix = embeddings_matrix[subset_indices]

def generate_silhouette_scores(n_clusters):
    """
    Generate silhouette scores for k-means clustering with k=n_clusters
    """
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(embeddings_matrix) + (n_clusters + 1) * 10])

    # Perform k-means clustering for k=n_clusters
    print(f"Performing k-means clustering for k={n_clusters}")
    niter = 20
    verbose = False
    d = embeddings_matrix.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=niter, verbose=verbose)
    kmeans.train(embeddings_matrix)
    D, cluster_labels = kmeans.index.search(embeddings_matrix, 1)
    cluster_labels = cluster_labels.flatten()

    # Save the cluster labels to a file
    with open(f'cluster_labels_{n_clusters}.pkl', 'wb') as f:
        pickle.dump(cluster_labels, f)
    print("Saved cluster labels to a file")

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    print("Calculating silhouette score. This may take a while. (1-2 hours)")
    silhouette_avg = silhouette_score(embeddings_matrix, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    print("Calculating silhouette scores for each sample. This may take a while. (1-2 hours)")
    sample_silhouette_values = silhouette_samples(embeddings_matrix, cluster_labels)
    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(f'silhouette_scores_{n_clusters}.png')
    print("Saved silhouette scores plot to a PNG file")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_silhouette_scores.py <n_clusters>")
        sys.exit(1)
    n_clusters = int(sys.argv[1])
    generate_silhouette_scores(n_clusters)