import numpy as np
import random
import sys
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import tkinter.filedialog
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
clusters = []

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
'''
#################################################
    calculate the distance matrix
#################################################
'''
def distanceMatrix(features):
    distMat =np.zeros(shape=(features.shape[0],features.shape[0]))
    for i in range(0,features.shape[0]):
        for j in range(i,features.shape[0]):
            distMat[i][j] = distMat[j][i]= dist(features[i], features[j], None)
    for i in range(0, features.shape[0]):
        distMat[i][i] = sys.maxsize
    return distMat

'''
    Hierarchical Clustering Algorithm
'''
def hierarchicalClustering(data, k):
    #calculate the distance matrix
    distMatrix = distanceMatrix(data)

    #clusters= []
    for i in range (0, data.shape[0]):
        clusters.append([i+1])
    #calculate the cluster with min distance
    while k < distMatrix.shape[0]:
        Min = sys.maxsize;c_row = 0; c_col = 0
        for j in range(0, distMatrix.shape[0]):
            temp = min(distMatrix[j])
            if (Min > temp):
                Min = temp; c_col = np.argmin(distMatrix[j]); c_row = j

        for i in range(0, distMatrix.shape[0]):
            if i != c_row and i != c_col:
                temp1 = distMatrix[c_row][i]
                temp2 = distMatrix[c_col][i]
                if (temp1 > temp2):
                    distMatrix[c_row][i] = distMatrix[i][c_row] = temp2
                else:
                    distMatrix[c_row][i] = distMatrix[i][c_row] = temp1
        distMatrix = np.delete(distMatrix, c_col, axis=0)
        distMatrix = np.delete(distMatrix, c_col, axis=1)
        clusters[c_row] = clusters[c_row] + clusters[c_col]
        del clusters[c_col]

input_file = tkinter.filedialog.askopenfilename()
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = (data[:,1])
data=data[:,2:]
k = input("Enter the k value: ")
k=int(k)
hierarchicalClustering(data,k)
labels = np.zeros(data.shape[0])
for i in range(0, len(clusters)):
    for j in clusters[i]:
        labels[j-1]=i+1
def algoPCA(data):
    # calculating the reduced sample space
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(data)
    reducedData = pca.transform(data)
    return reducedData
########################################
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
########################################
def plotGraph(reduData,typeOfPlot, labels):
    setlabel=set(labels)
    f, ax = plt.subplots(figsize=(10, 5))
    for name in setlabel:
        x = reduData[labels[:] == name, 0]  # all places where label is met and then plot (1st eigen*x) as x axis
        y = reduData[labels[:] == name, 1]  # all places where label is met and then plot (2nd eigen*x) as y axis
        ax.scatter(x, y, marker='o', label=name)
    plt.title(typeOfPlot+" plot for " + input_file)
    plt.legend( ncol=1, fontsize=12)
###################################
plotGraph(algoPCA(data),"Hierarchical", labels)
plt.show()
#####################################
jaccard_similarity = get_jaccard_similarity(data, labels, data_feature_matrix)
print("Jaccard similarity: " + str(jaccard_similarity))