from sklearn.cluster import KMeans


def create_model(n_clusters=3):
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        random_state=42
    )

    return model