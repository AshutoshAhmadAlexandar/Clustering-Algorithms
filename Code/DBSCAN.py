import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import tkinter.filedialog
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
clusters = []

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def expandcluster(p, neighborPts, c, eps, minPoints, visited, data):
    visited[p] = c
    for j in neighborPts:
        if visited[j] == 0:
            visited[j] = c
            neighborP = regionQuery(data, j, eps)
            if len(neighborP) >= minPoints:
                neighborPts += neighborP
        if visited[j] == -1:
            visited[j] = c

def regionQuery(data, p, eps):
    neighbors = []
    for i in range(0, data.shape[0]):
        if dist(data[p], data[i], ax=None) <= eps:
            neighbors.append(i)
    return neighbors

def dbscan(data, eps, minPoints):
    c = 0
    visited = np.zeros(data.shape[0])
    for i in range(0, data.shape[0]):
        if visited[i] != 0:
            continue
        neighborPts = regionQuery(data, i , eps)
        if len(neighborPts) < minPoints:
            visited[i] = -1
        else:
            c += 1
            expandcluster(i, neighborPts, c, eps, minPoints, visited, data)
    return visited
input_file = tkinter.filedialog.askopenfilename()
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = (data[:,1])
data=data[:,2:]
e = float(input("Enter the e value: "))
minpts=k = int(input("Enter the number of points: "))


labels = dbscan(data, e, minpts)

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
    setvisited=set(labels)
    f, ax = plt.subplots(figsize=(10, 5))
    for name in setvisited:
        x=reduData[labels[:] == name, 0]
    for name in setvisited:
        x = reduData[labels[:] == name, 0]
        y = reduData[labels[:] == name, 1]
        ax.scatter(x, y, marker='o', label=name)
    plt.title(typeOfPlot)
    plt.legend( ncol=1, fontsize=12)
###################################
plotGraph(algoPCA(data),"DBSCAN plot for "+input_file, labels)
plt.show()
#####################################
jaccard_similarity = get_jaccard_similarity(data, labels, data_feature_matrix)
print("Jaccard similarity: " + str(jaccard_similarity))