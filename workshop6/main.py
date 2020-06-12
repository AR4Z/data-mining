from sklearn.datasets import load_iris
from sklearn import preprocessing
import random

data = load_iris()
X = list(preprocessing.normalize(data.data))

for x in range(len(X)):
    X[x] = list(X[x])


class KMeans():
    def __init__(self, X, n_clusters=3):
        self.X = X
        self.n_clusters = n_clusters
        self.clusters = random.sample(self.X, self.n_clusters)

        for x in self.X:
            cluster = self.get_cluster(x)
            # last position -> cluster number
            x.append(cluster)
        self.iterate()

    def iterate(self):
        changes = True
        iteraciones = 0
        while changes:
            iteraciones += 1
            print(iteraciones)
            cont_changes = 0
            for x in self.X:
                cluster = self.get_cluster(x)
                # last position -> cluster number
                if cluster != x[-1]:
                    cont_changes += 1

                x[-1] = cluster
            self.update_clusters()
            changes = cont_changes > 0

        self.show_clusters()

    def get_cluster(self, x):
        distances = [self.distance(x, cluster) for cluster in self.clusters]
        cluster = distances.index(min(distances))
        return cluster

    def update_clusters(self):
        new_clusters = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        cont = [0, 0, 0]

        for x in self.X:
            cluster = x[-1]
            cont[cluster] += 1
            for i in range(self.n_clusters):
                new_clusters[cluster][i] += x[0]

        for new_cluster in range(len(new_clusters)):
            for feature in range(len(new_clusters[new_cluster])):
                new_clusters[new_cluster][feature] /= cont[new_cluster]

        self.clusters = new_clusters

    def distance(self, attrs1, attrs2):
        return ((attrs1[0] - attrs2[0])**4 + (attrs1[1] - attrs2[1])**4 +
                (attrs1[2] - attrs2[2])**4 + (attrs1[3] - attrs2[3])**4) ** (0.25)

    def show_clusters(self):
        for num_cluster in range(self.n_clusters):
            print(f'Elementos cluster {num_cluster}')
            for x in self.X:
                if x[-1] == num_cluster:
                    print(x)


kmeans = KMeans(X, 3)
