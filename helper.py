import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats

class knn:
    def __init__(self, x_train, y_train, k):
        self.k = k
        self.x = x_train
        self.y = np.array(y_train)

    def predict(self, x_test):
        # find the euclidean distance between each point in x_test and self.x
        distances = pairwise_distances(x_test, self.x, metric='euclidean')
        # choose k min distances for each test point
        indices = distances.argsort(axis=1)[:, :self.k]
        # lookup up class in y_train
        candidates = (self.y[np.ravel(indices)]).reshape(indices.shape)
        # mode of chosen classes is prediction
        self.predictions = stats.mode(candidates,axis=1)[0]
        return self.predictions.flatten()



class kmeans:
    def __init__(self, x_train, y_train, k, iter=100, tolerance=0.1):
        self.k = k
        self.x = np.array(x_train)
        self.y = np.array(y_train)
        self.centers = self.x[np.random.choice(range(self.x.shape[0]), size=self.k, replace=False)]
        self.classes = [None] * self.k  # the class corresponding to each cluster
        self.iterations = iter
        for i in range(self.iterations):
            #print("iteration ", i)
            distances = pairwise_distances(self.centers, self.x)  # rows = center, columns = x_train
            assigned_clusters = np.argmin(distances, axis=0)
            old_centers = self.centers
            # recalculate centroidss
            for cluster in range(self.k):
                indicesInCluster = np.where(
                    assigned_clusters == cluster)[0]  # indices of points in xtrain that belong to the specified cluster k
                pointsInCluster = self.x[indicesInCluster]  # the points corresponding to said indices
                classesInCluster = self.y[indicesInCluster]  # the classes corresponding to said points
                self.classes[cluster] = stats.mode(classesInCluster)[0]
                self.centers[cluster] = np.mean(pointsInCluster,
                                                axis=0)  # the average of all the points that belong to this center
            # detect centeroids movement
            # if abs(np.sum((self.centers - old_centers) / self.centers)) < 0.01:
            #     break

    def predict(self, x_test):
        distances = pairwise_distances(self.centers, x_test)
        assigned_clusters = np.argmin(distances, axis=0)
        predictions = np.array(self.classes)[assigned_clusters].flatten()
        return predictions
