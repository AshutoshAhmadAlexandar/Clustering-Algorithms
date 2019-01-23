#!/usr/bin/env python
"""reducer.py"""
import sys
import numpy as np

curCluster = None
newDataset = ""
oldCluster = None
count = 0
total = None

# input comes from STDIN
for line in sys.stdin:
    # remove whitespace
    line = line.strip()

    oldCluster, datasetIndex, datasetStr = line.split('\t')
    dataset = np.fromstring(datasetStr, dtype='float', sep=',')

    if curCluster == oldCluster:
        newDataset += "," + datasetIndex
        count += 1
        total += dataset
    else:
        if curCluster:
            newCluster = total / count
            newClusterStr = ','.join(str(feature) for feature in newCluster)
            print ('%s\t%s\t%s' % (curCluster, newDataset, newClusterStr))
        newDataset = datasetIndex
        curCluster = oldCluster
        count = 1
        total = np.copy(dataset)

if curCluster == oldCluster and curCluster != None:
    newCluster = total / count
    newClusterStr = ','.join(str(feature) for feature in newCluster)
    print ('%s\t%s\t%s' % (curCluster, newDataset, newClusterStr))