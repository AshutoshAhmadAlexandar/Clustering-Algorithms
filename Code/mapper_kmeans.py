#!/usr/bin/env python
"""mapper.py"""

import sys
import numpy as np
import os

inputData = np.loadtxt(sys.stdin, dtype='float')

clusters = np.loadtxt("./cluster.txt", dtype='float')
feature_matrix = inputData[0:inputData.shape[0], :]

for i in range(feature_matrix.shape[0]):
    point = feature_matrix[i]
    point_string = ','.join(str(feature) for feature in point)
    clusterIndex = np.argmin(np.linalg.norm(clusters - point, axis=1))
    print('%s\t%s\t%s' % (clusterIndex, i, point_string))
