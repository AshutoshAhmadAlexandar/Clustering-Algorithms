from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
import random
from matplotlib import pyplot as plt
import tkinter.filedialog

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

input_file = tkinter.filedialog.askopenfilename()
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = data[:, 1]
data = data[:, 2:]
k = input("Enter the k value: ")
k=int(k)
rand_num = [int(x) for x in input("Enter the initial centroid: ").split()]
iter = int(input("Enter the Iteration value: "))

data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = (data[:,1])
data=data[:,2:]
#rand_num = random.sample(range(0, data.shape[0]), k)
centroids=data[rand_num,:]
C_old = np.zeros(centroids.shape)
clusters = np.zeros(data.shape[0])
error=dist(centroids,C_old,None)
count=0
while error != 0 and count<iter :
    # Assigning each value to its closest cluster
    count=count+1
    for i in range(data.shape[0]):
        distances = dist(data[i,:], centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(centroids)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [data[j,:] for j in range(data.shape[0]) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    error = dist(centroids, C_old, None)
clusters=clusters+1
print("Number of iterations done:"+ str(count))
######
def get_jaccard_similarity(data_matrix, labels_list, data_feature_matrix):
    nowrows = len(data_matrix)
    predicted_matrix = np.zeros((nowrows, nowrows))
    ground_truth_matrix = np.zeros((nowrows, nowrows))

    # populate the same cluster matrices
    for i in range(predicted_matrix.shape[0]):
        predicted_matrix[i][i] = 1
        ground_truth_matrix[i][i] = 1
        for j in range(i + 1, predicted_matrix.shape[1]):
            if labels_list[i] == labels_list[j]:
                predicted_matrix[i][j] = 1
                predicted_matrix[j][i] = 1
            if data_feature_matrix[i] == data_feature_matrix[j]:
                ground_truth_matrix[i][j] = 1
                ground_truth_matrix[j][i] = 1

    # calculate the jaccard similarity
    same1Count=0
    diff=0
    for i in range(len(predicted_matrix)):
        for j in range(len(predicted_matrix)):
            if ground_truth_matrix[i][j] ==1:
                if ground_truth_matrix[i][j] == predicted_matrix[i][j]:
                    same1Count +=1
                else :
                    diff +=1
            else:
                if ground_truth_matrix[i][j] != predicted_matrix[i][j]:
                    diff +=1

    Jaccard = (same1Count)/(same1Count+diff)
    return Jaccard
###########################
def algoPCA(data):
    # calculating the reduced sample space
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(data)
    reducedData = pca.transform(data)
    return reducedData
########################################
def plotGraph(reduData,typeOfPlot, labels):
    setlabel=set(labels)
    f, ax = plt.subplots(figsize=(10, 5))
    for name in setlabel:
        x = reduData[labels[:] == name, 0]
        y = reduData[labels[:] == name, 1]
        ax.scatter(x, y, marker='o', label=name)
    plt.title(typeOfPlot)
    plt.legend( ncol=1, fontsize=12)
###################################
plotGraph(algoPCA(data),"K-means plot for "+input_file, clusters)
plt.show()

######
jaccard_similarity = get_jaccard_similarity(data, clusters, data_feature_matrix)
print("Jaccard similarity: " + str(jaccard_similarity))