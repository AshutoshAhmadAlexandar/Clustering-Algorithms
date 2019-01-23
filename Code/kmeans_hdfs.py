import numpy as np
import random
import os
import subprocess
import tempfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to run the mapper and reducer 

def hadoop_processing(hadoop_cmd):
    os.system("hdfs dfs -rm -r /user/" )
    os.system("hdfs dfs -mkdir /user/" )
    os.system("hdfs dfs -mkdir /user/input/" )
    os.system("hdfs dfs -put input.txt /user/input/")
    os.system("hdfs dfs -put cluster.txt /user/")
    os.system("hdfs dfs -put mapper_kmeans.py /user/")
    os.system("hdfs dfs -put reducer_kmeans.py /user/")
    os.system(hadoop_cmd)

def get_cluster():
    clusterDict = dict()
    with tempfile.TemporaryFile() as tempf:
        proc = subprocess.Popen(["hadoop", "fs", "-cat", "/user/output/part*"], stdout=tempf)
        proc.wait()
        tempf.seek(0)
        lines = tempf.readlines()
        
        for line in lines:
            decodedline = line.decode('ascii').strip()
            clusterIndex, points, newCluster = decodedline.split("\t")
            clusterDict[int(clusterIndex)] = [[int(point) for point in points.split(",")],np.fromstring(newCluster, dtype='float', sep=',')]
    return clusterDict

# function for kMeans
def hadoop_kmeans(cluster, data, hadoop_cmd):

    cluster_old = np.zeros(cluster.shape)
    count = 0

    while not np.array_equal(cluster, cluster_old):
        count += 1
        np.savetxt("cluster.txt", cluster, delimiter=' ')
        np.savetxt("input.txt", data, delimiter=' ')
        hadoop_processing(hadoop_cmd)

        cluster_dict = get_cluster()

        np.copyto(cluster_old, cluster)

        for clusterIndex in cluster_dict:
            cluster[clusterIndex, :] = cluster_dict[clusterIndex][1]

    print("Total run: " + str(count))
    
    label_list = np.ones(data.shape[0])
    label_list = label_list * -1
    for point_index in range (data.shape[0]):
        for cluster_index in cluster_dict:
            if point_index in cluster_dict[cluster_index][0]:
                label_list[point_index] = cluster_index
    return label_list

# reading input
input_file = tkinter.filedialog.askopenfilename()
data = np.loadtxt(input_file, dtype='float')
data_feature_matrix = data[:, 1]
data = data[:, 2:]
k = input("Enter the k value: ")
rand_num = [int(x) for x in input("Enter the initial centroid: ").split()]
iteration = int(input("Enter the Iteration value: "))

#k = len(set(data_feature_matrix))
#rand_num = random.sample(range(0, data.shape[0]), k)

cluster = data[rand_num,:]
hadoop_cmd = "hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.9.1.jar -Dmapreduce.job.maps="+str(k)+" -Dmapreduce.job.reduces="+str(k)+ " -input /user/input/ -file mapper_kmeans.py -mapper mapper_kmeans.py -file reducer_kmeans.py -reducer reducer_kmeans.py -output /user/output/" 

# k-means hadoop
labels = hadoop_kmeans(cluster, data, hadoop_cmd)

# PCA
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

    # calculate the jaccard similarity
    same1Count=0
    #same0Count=0
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
    setvisited=list(set(labels))
    f, ax = plt.subplots(figsize=(10, 5))
    colors=[plt.cm.jet(float(i)/max(setvisited)) for i in setvisited]

    for name in setvisited:
        x = reduData[labels[:] == name, 0]  # all places where visited is met and then plot (1st eigen*x) as x axis
        y = reduData[labels[:] == name, 1]  # all places where visited is met and then plot (2nd eigen*x) as y axis
        ax.scatter(x, y,marker='o',c=colors[int(name)], label=name)
    plt.title(typeOfPlot+" plot for : " + "kMeans")
    plt.legend( ncol=1, fontsize=12)

###################################
plotGraph(algoPCA(data),"Clustering", labels)
plt.show()

#####################################
jaccard_similarity = get_jaccard_similarity(data, labels, data_feature_matrix)
print("Jaccard similarity: " + str(jaccard_similarity))
