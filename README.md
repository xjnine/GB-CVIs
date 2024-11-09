# An Efficient Internal Clustering Validity Index Optimization Method Based on Coarse-Grained Approach
Internal clustering validity indices (CVIs) serve as tools for evaluating and optimizing unsupervised clustering algorithms. However, most current internal CVIs are computed based on single-granularity data points, involving repetitive calculations of relationships between all pairs of points (such as Euclidean
distances), resulting in low efficiency in CVI assessment. Moreover, for cluster structures with vague inter-cluster data point distribution, the computation of internal CVIs is easily influenced by boundary data points and noisy data points within clusters. These issues affect the accuracy and evaluation efficiency of CVIs. A potential solution is to select representative points to participate in the computation of internal CVIs. Inspired by this,
we considered implementing a coarse-grained representation of unsupervised data using a Granular-Ball (GB) structure based on a coarse-grained method. This structure can delineate clearer clustering structures without altering the original distribution of the dataset, thereby enhancing both the efficiency and accuracy of the original CVI algorithms. We conducted experimental analyses on multiple CVIs using both synthetic and real datasets. 
# Files
datasets: This file contains the synthetic datasets and large-scale datasets used in the experiment.

test-: The files starting with test are the optimization indicators used in the article. The three indicators ANCV, DCVI and LCCV are separate folders.

HyperBallClustering_acceleration_v4: This file is the algorithm for obtaining the granular-ball collection.
# run
Files named beginning with test can be run directly. Be sure to modify the path of the dataset.
