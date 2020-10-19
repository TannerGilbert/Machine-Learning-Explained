# KMeans Explained

Clustering is a machine learning technique that involves grouping similar data points together into so called clusters. Clustering is an unsupervised learning method commonly used in data science and other fields.

KMeans is probably the most well-known of all the clustering algorithm. Its goal is to separate the data into K distinct non-overlapping subgroups (clusters) of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.

<p align="center"><img src="tex/5d2031093fe35c15cf01b562bab7d54f.svg?invert_in_darkmode" align=middle width=154.94883855pt height=44.89738935pt/></p>

## KMeans theory

KMeans works as follows:
1. First, pick the number of clusters (For more info, check the ["Choosing K" section](#choosing-k)).
2. Initialize the center points of the cluster (centroids) by shuffling the dataset and then selecting K data points for the centroids.
3. Assign data points to the cluster with the nearest centroid.
4. Recompute centroid position by taking the mean of all data points assigned to the cluster. 
5. Repeat steps 3 and 4 for a set number of iterations or until the centroids aren't moving much between iterations anymore.

![k_means](doc/k_means.gif)

## Choosing K

Choosing the right K value by hand can get quite tricky, especially if you're working with 3+ dimensional data. If you select a too small or big number for K, the result can be quite underwhelming.

![choose_k_value](doc/choose_k_value.jpeg)

In this section, I'll show you two methods commonly used to choose the right K value:
* The Elbow Method
* Silhouette Analysis

### Elbow Method

The Elbow Method shows us what a good number for K is based on the sum of squared distances (SSE) between data points and their assigned clusters' centroid. We pick k at the spot where the SSE starts to flatten out, which looks like an elbow. Below you can see an example created using [Yellowbrick](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html).

![elbow_method_using_yellowbrick](doc/elbow_method_using_yellowbrick.png)

### Silhouette Analysis

The Silhouette Analysis can be used to study the separation distance between the resulting clusters. It displays a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation) and can thus be used to assess the number of clusters k. 

The Silhouette Analysis is computed as follows:
* Compute the average distance between all data points in one cluster <img src="tex/db0e77b2ab4f495dea1f5c5c08588288.svg?invert_in_darkmode" align=middle width=16.39974929999999pt height=22.465723500000017pt/>
<p align="center"><img src="tex/065cfac694daeb1fff1264475e035c67.svg?invert_in_darkmode" align=middle width=214.9944654pt height=43.7234787pt/></p>
* For all data points <img src="tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> in cluster <img src="tex/db0e77b2ab4f495dea1f5c5c08588288.svg?invert_in_darkmode" align=middle width=16.39974929999999pt height=22.465723500000017pt/> compute the average distance to all points in another cluster <img src="tex/1a567506286617473a9c0d9b2172f951.svg?invert_in_darkmode" align=middle width=19.014878849999988pt height=22.465723500000017pt/> (where <img src="tex/f0c3f612efc905c5a416138c62517a36.svg?invert_in_darkmode" align=middle width=58.15417244999999pt height=22.831056599999986pt/>) 
<p align="center"><img src="tex/b776953fbf2b14971aa17331a8640386.svg?invert_in_darkmode" align=middle width=195.02974589999997pt height=43.5956565pt/></p>

>The <img src="tex/b3520dc7da5f9731724eb6e1768a45a7.svg?invert_in_darkmode" align=middle width=29.96320304999999pt height=21.68300969999999pt/> is used, because we want to know the average distance to the closed cluster <img src="tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> is not a member of.

With <img src="tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/> and <img src="tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054796099999991pt height=22.831056599999986pt/> we can now calculate the silhouette coefficient:

<p align="center"><img src="tex/16ceb724dafaab6c19cf71bc5c460244.svg?invert_in_darkmode" align=middle width=251.1557565pt height=38.83491479999999pt/></p>

The coefficient can take values in the interval <img src="tex/43ca5ad9e1f094a31392f860ef481e5c.svg?invert_in_darkmode" align=middle width=45.66218414999998pt height=24.65753399999998pt/>. Zero means the sample is very close to the neighboring clusters. One means the sample is far away from the neighboring cluster, and negative one means the sample is probably assigned to the wrong cluster.

Below you can see an [example of silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) using [Scikit Learn](https://scikit-learn.org/stable/index.html):

![silhouette_analysis_3_clusters](doc/silhouette_analysis_3_clusters.jpeg)

![silhouette_analysis_4_clusters](doc/silhouette_analysis_4_clusters.jpeg)

![silhouette_analysis_5_clusters](doc/silhouette_analysis_5_clusters.jpeg)

## Advantages

KMeans is an easy-to-implement algorithm that is also quite fast with an average complexity of <img src="tex/5c9a23f70c5920444f4613242c1e95fb.svg?invert_in_darkmode" align=middle width=87.66234179999998pt height=24.65753399999998pt/>, where n is the number of samples, and T is the number of iteration.

## Drawbacks

As mentioned above, KMeans makes use of the **sum-of-squares criterion**, which works well if the clusters have a spherical-like shape. It doesn't work well on many other types of data like complicated shapes, though. In this section, we'll go over a few cases where KMeans performs poorly.

First, KMeans doesn't put data points that are far away from each other into the same cluster, even when they obviously should be because they underly some obvious structure like points on a line, for example.

![two_lines-1](doc/two_lines.png)

In the image above, you can see that KMeans creates the clusters in between the two lines and therefore splits each line into one of two clusters rather than classifying each line as a cluster. On the right side, you can see the DBSCAN (Density-based spatial clustering of applications with noise) algorithm, which is able to separate the two lines without any issues.

Also, as mentioned at the start of the section KMeans performs poorly for complicated geometric shapes such as the moons and circles shown below.

![noisy_moons_with_true_output](doc/noisy_moons_with_true_output.png)

![noisy_circles_with_true_output](doc/noisy_circles_with_true_output.png)

Other clustering algorithms like Spectral Clustering, Agglomerative Clustering, or DBSCAN don't have any problems with such data. For a more in-depth analysis of how different clustering algorithms perform on different interesting 2d datasets, I recommend checking out ['Comparing different clustering algorithms on toy datasets'](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html) from Scikit-Learn.  

## Code
* [KMeans from Scratch in Python](code/kmeans.py)

## Credit / Other resources
* https://scikit-learn.org/stable/modules/clustering.html#k-means
* https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
* https://www.youtube.com/watch?v=4b5d3muPQmA