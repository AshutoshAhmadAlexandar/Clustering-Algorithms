# Clustering-Algorithms

Complete the following tasks:

1. Implement three clustering algorithms to find clusters of genes that exhibit similar expression profiles: K-means, Hierarchical Agglomerative clustering with Single Link (Min), and one from (density-based, mixture model, spectral). Compare these three methods and discuss their pros and cons.
For each of the above tasks, you are required to validate your clustering results using the following methods:

• Choose an external index (Rand Index or Jaccard Coefficient) and compare the clustering results from different clustering algorithms. The ground truth clusters are provided in the datasets.

• Visualize data sets and clustering results by Principal Component Analysis (PCA). You can use the PCA you implemented in Project 1 or use any existing implementation or package.

2. Set up a single-node Hadoop cluster on your own machine and implement MapReduce K- means. Compare with non-parallel K-means on the given datasets. Try to improve the running time. To set up single-node Hadoop, we provide an instruction file Hadoop_setup.pdf that can be found on Piazza.
