CSE-601

Read Me - K_means hadoop:
=================================================================

-Setup the hadoop framework of single node on a machine.
-Run following commands to setup the hdfs.
	- hdfs namenode -format
	- Run start-dfs.sh
	- check that instance of datanode , namenode and secondary node are up and running using command "jps"
-Run the file using python as follows:
	python3 kmeans_hdfs.py 
-You will be prompted to select the file:
	select filename
	e.g. choose file:cho.txt 
-You will be prompted to Enter the cluster info:
	e.g. Enter the k value: 4
		 Enter the initial centroids: 3 45 5 7
	     Enter the Iteration value: 34
-One plots will be displayed for K-means.
-Repeat the steps for other dataset files.