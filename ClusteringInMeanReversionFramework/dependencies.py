from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import string
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def perform_kmeans_clustering(data_frame, clustering_fields, cmap="viridis",
                              plot_title="K-means Clustering with 3 clusters", save_as_image=False):
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


def perform_polynomial_clustering(data_frame, clustering_fields, plot_title, save_as_image=False):
    xs = data_frame[clustering_fields[0]].values.reshape(-1, 1)
    ys = data_frame[clustering_fields[1]].values
    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    xs_poly = poly_features.fit_transform(xs)
    model = LinearRegression()
    model.fit(xs_poly, ys)

    xs_pred = np.linspace(xs.min(), xs.max(), 100).reshape(-1, 1)
    xs_pred_poly = poly_features.transform(xs_pred)
    y_pred = model.predict(xs_pred_poly)

    plt.scatter(xs, ys, color='blue', label='Data points')
    plt.plot(xs_pred, y_pred, color='red', label=f'Polynomial regression (degree={degree})')
    plt.xlabel(clustering_fields[0])
    plt.ylabel(clustering_fields[1])
    plt.title(plot_title)
    plt.legend()

    if save_as_image:
        plt.savefig("regressions/" + ''.join(random.choices(string.ascii_letters + string.digits, k=10)))

    plt.show()


def get_annual_return(initial_value, final_value):
    return ((final_value - initial_value) / initial_value) * 100
