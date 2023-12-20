import time
import os
import math
import numpy as np
import pandas as pd
from math import gcd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from matplotlib import cm
from LCM_clustering.function.test_update_hamaad import doEverything
from LCM_clustering.function.test_update_hamaad import doGroup

def compute_distance(point, centroids):
    distances = np.linalg.norm(centroids - point, axis=1)
    return distances

def calculate_centroids(X, assigned):
    centroids = []
    unique_clusters = np.unique(assigned)
    for cluster_id in unique_clusters:
        indices = np.where(assigned == cluster_id)[0]
        cluster_points = X[indices]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return centroids

def assign_to_clusters(data, centroids):
    assigned = []
    for point in data:
        distances = [euclidean(point, centroid) for centroid in centroids]
        assigned.append(np.argmin(distances))
    return np.array(assigned)

def read_data(file_path, columns):
    df = pd.read_csv(file_path, usecols=columns)#, nrows=25000)
    return df.values  

def find_lcm_diff(data):
    n = len(data)
    diff = np.zeros((n, n), dtype=np.int64)
    assigned = [-1] * n 
    clusters = []
    lcm_threshold =1
    distance_scale_factor = 10
    
    for i in tqdm(range(n), desc="euclidean distance", leave=False):
        diff[i, i+1:] = diff[i+1:, i] = np.round(np.linalg.norm(data[i] - data[i+1:], axis=1) * distance_scale_factor)
     
    for i in tqdm(range(n),desc ="Clustering",leave=False):
        if assigned[i] >= 0:
            continue

        found_cluster = False
        
        for j in range(len(clusters)):
            lcm_cluster = np.lcm.reduce(diff[clusters[j], clusters[j]])
            lcm_new_point = np.lcm.reduce(diff[i, clusters[j]])
            if lcm_cluster <= lcm_threshold and lcm_cluster != 0 and lcm_new_point % lcm_cluster == 0 and lcm_new_point % lcm_cluster == 0:#all(np.lcm.reduce([lcm_new_point] + [diff[i, sample] for sample in clusters[j]]) == lcm_new_point for sample in clusters[j]) :
                clusters[j].append(i)
                assigned[i] = j
                found_cluster = True
                break
          
        if not found_cluster:
            new_cluster = [i]
            assigned[i] = len(clusters)
            for j in range(i+1, n):
                lcm_diff = np.lcm.reduce(diff[i, j])
                if lcm_diff <= lcm_threshold:
                    new_cluster.append(j)
                    assigned[j] = len(clusters)
            clusters.append(new_cluster) 
           
    clusters = [cluster for cluster in clusters if cluster]

    print("length of clusters: ", len(clusters))
    
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}: {cluster}")

    clusters = doEverything(clusters)

    print("Number of clusters after modifications:: ", len(clusters))

    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}: {cluster}")
   
    cluster_info = {}
    for i in range(len(clusters)):
        cluster_data = clusters[i]
        cluster_diff = diff[np.ix_(cluster_data, cluster_data)]
        cluster_lcm_diff = np.lcm.reduce(cluster_diff.flatten())
        cluster_info[i] = {"data": cluster_data, "diff": cluster_diff, "lcm_diff": cluster_lcm_diff}

    return assigned, cluster_info

data_path = "Dataset/path" 
selected_columns = ['write the column names']
X = read_data(data_path, selected_columns)

assigned, cluster_info = find_lcm_diff(X)

clusters = pd.Series(assigned, name="cluster")

new_assigned =[]
centroids = calculate_centroids(X, assigned)
print("Centroid:", len(centroids))

num_points_in_clusters = [ len(np.where(np.array(assigned) == c)[0]) for c in np.unique(clusters) ]
for i, point in enumerate(X):

    cluster_of_point = assigned[i]
    num_points_in_cluster_of_point = num_points_in_clusters[cluster_of_point]
   
    if num_points_in_cluster_of_point == 1:
        distances = compute_distance(point, centroids)
        print("Point:", point)
        distances[distances == 0] = 'inf'
        distances[[num_points_in_clusters[c] == 1 for c in np.unique(clusters) ]] = 'inf'
        print("Distances:", distances)
        assigned_cluster = np.argmin(distances)
        print("Assigned Cluster:", assigned_cluster)
        print("Original cluster:",assigned[i])
        print("---------------")
        new_assigned.append(assigned_cluster)
    else:
        new_assigned.append(assigned[i])

new_clusters = pd.Series(new_assigned, name="cluster")

df = X.copy() 

df = pd.concat([pd.DataFrame(X, columns=["column names"] ), new_clusters], axis=1)
df_with_clusters = pd.concat([df, new_clusters], axis=1)

df.to_csv('zero_phase_redmi_fd_900samples_clusters_withlabels.csv', index=False)

if False:
    print("Cluster Information:")
    for cluster_id, info in cluster_info.items():
        print("Cluster:", cluster_id)
        print("Data:", info["data"])
        print("Diff:", info["diff"])
        print("LCM Diff:", info["lcm_diff"])
        print("---------------")

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')  
ax2 = fig.add_subplot(122, projection='3d') 

n_clusters = len(cluster_info)

centroids = calculate_centroids(X, assigned)
new_num_clusters = len(np.unique(new_assigned))
old_num_clusters = len(np.unique(assigned))

cmap = plt.get_cmap('tab20')
colors = [cmap(i % cmap.N) for i in range(new_num_clusters)]
colors_old = [cmap(i % cmap.N) for i in range(old_num_clusters)]

for i, cluster_id in enumerate(np.unique(assigned)):
    indices = np.where(np.array(assigned) == cluster_id)[0]
    ax1.scatter3D(X[indices, 0], X[indices, 1], X[indices, 2], s=40, color=colors_old[i], alpha=0.5)

ax1.set_title('Initial Clusters')
ax1.set_xlabel('Bx')
ax1.set_ylabel('By')
ax1.set_zlabel('Bz')

for i, cluster_id in enumerate(np.unique(new_assigned)):
    indices = np.where(np.array(new_assigned) == cluster_id)[0]
    ax2.scatter3D(X[indices, 0], X[indices, 1], X[indices, 2], s=40, color=colors[i], alpha=0.5)


ax2.set_title('New Clusters')
ax2.set_xlabel('Bx')
ax2.set_ylabel('By')
ax2.set_zlabel('Bz')

silhouette_avg = silhouette_score(X, assigned)
calinski_harabasz_avg = calinski_harabasz_score(X, assigned)
davies_bouldin_avg = davies_bouldin_score(X, assigned)


print("For n_clusters =", len(cluster_info))
print("Silhouette score of initial clusters  :", silhouette_avg)
print("Calinski-Harabasz index initial clusters:", calinski_harabasz_avg) 
print("Davies-Bouldin index initial clusters:", davies_bouldin_avg)


silhouette_avg_new = silhouette_score(X, new_assigned)
calinski_harabasz_avg_new = calinski_harabasz_score(X, new_assigned)
davies_bouldin_avg_new = davies_bouldin_score(X, new_assigned)


print("NEW - No of n_clusters  =", new_num_clusters)
print("NEW - Silhouette score is :", silhouette_avg_new)
print("NEW - Calinski-Harabasz index is:", calinski_harabasz_avg_new)
print("NEW - Davies-Bouldin index is:", davies_bouldin_avg_new)

print("Old clusters:", assigned)
print("New clusters:", new_assigned)

plt.show()
