from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import string


def perform_kmeans_clustering(data_frame, clustering_fields, cmap="viridis", plot_title="K-means Clustering with 3 clusters", save_as_image=False):
    clustering_data = data_frame[clustering_fields]
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(clustering_data)


    data_frame['Cluster'] = kmeans.labels_
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data_frame[clustering_fields[0]], 
        data_frame[clustering_fields[1]], 
        c=data_frame['Cluster'], 
        cmap=cmap, 
        norm=mcolors.Normalize(vmin=0, vmax=2), alpha=0.5
    )

    plt.title(plot_title)
    plt.xlabel(clustering_fields[0])
    plt.ylabel(clustering_fields[1])
    plt.colorbar(label='Cluster')

    if save_as_image:
        plt.savefig("distributions/" + ''.join(random.choices(string.ascii_letters + string.digits, k=10)))

    plt.show() 


def get_annual_return(initial_value, final_value):
    return ((final_value - initial_value) / initial_value) * 100

