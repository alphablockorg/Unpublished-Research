from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import string
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

POLYNOMIAL_DEGREE = 3


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


def get_polynomial_features(data_frame, clustering_fields):
    xs = data_frame[clustering_fields[0]].values.reshape(-1, 1)
    ys = data_frame[clustering_fields[1]].values

    poly_features = PolynomialFeatures(degree=POLYNOMIAL_DEGREE)
    xs_poly = poly_features.fit_transform(xs)
    model = LinearRegression()
    model.fit(xs_poly, ys)

    xs_pred = np.linspace(xs.min(), xs.max(), 100).reshape(-1, 1)
    xs_pred_poly = poly_features.transform(xs_pred)
    y_pred = model.predict(xs_pred_poly)

    return xs, ys, xs_pred, y_pred


def perform_polynomial_on_clustering(data_frame, clustering_fields, plot_title, save_as_image=False):
    xs, ys, xs_pred, y_pred = get_polynomial_features(data_frame, clustering_fields)

    plt.scatter(xs, ys, color='blue', label='Data points')
    plt.plot(xs_pred, y_pred, color='red', label=f'Polynomial regression (degree={POLYNOMIAL_DEGREE})')
    plt.xlabel(clustering_fields[0])
    plt.ylabel(clustering_fields[1])
    plt.title(plot_title)
    plt.legend()

    if save_as_image:
        plt.savefig("regressions/" + ''.join(random.choices(string.ascii_letters + string.digits, k=10)))

    plt.show()


def combine_polynomial_plots(data_frame, clustering_fields, plot_color, all_xs_pred=None, all_y_pred=None):
    xs, ys, xs_pred, y_pred = get_polynomial_features(data_frame, clustering_fields)

    if all_xs_pred is None:
        all_xs_pred = []
    if all_y_pred is None:
        all_y_pred = []

    all_xs_pred.append(xs_pred)
    all_y_pred.append(y_pred)

    return all_xs_pred, all_y_pred


def get_annual_return(initial_value, final_value):
    return ((final_value - initial_value) / initial_value) * 100
