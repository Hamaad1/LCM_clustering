
from collections import defaultdict
from scipy.spatial.distance import euclidean 
import numpy as np

#remove the minDistance related code from the function to return to the main code

def checkNeighbours(clusters, rangeNeighbours=4, maxNeighbours=3):
    checkClusters=[]
    # checkClusters will contain only lists of True or False, it deepends if the number at that index has neighbours in that cluster or not.
    for cluster in clusters:
        checkCluster=[]
        # checkCluster is the list of the single cluster that will be added to checkClusters
        for element in cluster:
            count=0
            # for every number in the single cluster i have to count the number of neighbours, i use "count" for that.
            for otherElements in cluster:
                if abs(element-otherElements)<rangeNeighbours and not element==otherElements:
                    count+=1
                    # if we are in the range established to define this number as the 'neighbour' of another and "element" 
                    # and "otherElements" are not the same number (you could also see if the subtraction results 0)

                    if count==maxNeighbours:
                        # if I reach the target number i have to consider that number with neighbours,
                        # so i have to append True and break the cycle.
                        checkCluster.append(True)
    #                    break
            else:
                # if i didn't break (count doesn't reaches the target number) i have to consider that number without neighbours,
                # so i have to append False.
                checkCluster.append(False)
        # at the end of that iteration i should have a list, that represents my cluster, full of "True" or "False", 
        # so i have to append that list to the clusters list
        checkClusters.append(checkCluster)
    return checkClusters

def makeUniqueInd(clustersUnique, allIndexes, listRep):
    clustersN=checkNeighbours(clustersUnique)

    newClusters=defaultdict(list)
    # i need a new dictionary full of lists, i'll use the number of the cluster as key, and the cluster itself as value, it will be a list, so i use defaultdict()

    for element in allIndexes:
        # i need to iterate every element, allIndexes contains all the numbers in clusters, without clusters or repetitions.
        
        rep=0               # numbers of repetitions for that number (the index inside every cluster)
        numeroC=None        # Cluster number
        numero=None         # element
        stored=False        # is that number stored yet?
        for nCluster, c in enumerate(listRep):
            #for every cluster in listRep. nCluster will contain the number of that cluster, c will contain the dictionary of listRep with the index repetitions for that cluster
            if element in c.keys():
                # fif the element is in that cluster
                hasN=clustersN[nCluster][clustersUnique[nCluster].index(element)]
                # hasN will contain "True" if that element has neighbours

                if hasN:
                    rep, numeroC, numero=c[element], nCluster, element
                    stored=True
                elif c[element]>rep and not stored:
                    rep, numeroC, numero=c[element], nCluster, element
                # print("elemento: ", element, " in", nCluster, "rep", c[element])
        newClusters[numeroC].append(numero)
    # this creates the definitive dictionary of clusters

    newClusters=sorted(newClusters.items())
    # sorting that dictionary by the keys

    listDef=[element for _, element in newClusters]
    # creating a list where every element will be a list from "newClusters"

    return listDef

def doEverything(clusters):
    clustersUnique=[list(set(c)) for c in clusters]
    # The instruction above is equivalent to:
    # for c in clusters:
    #     cUnique=list(set(c))
    #     clustersUnique.append(cUnique)

    # Sets are like lists but they cannot have two items with the same value.

    allIndexes={ind for c in clustersUnique for ind in c}
    # Similar to clustersUnique, but using only sets

    listRep=[]
    for nCluster, c in enumerate(clusters):
        dict={}
        for element in clustersUnique[nCluster]:
            dict[element]=c.count(element)
        listRep.append(dict)

    # Every element of listRep is a Dictionary, in there the key is the element of every cluster and the element is the repetition number.
    # i.e.: if c1 has 3 times 0 and 4 times 2, so the first dictionary inside listRep contains {"0":3, "2":4}
    # if c2 has 6 times 1, 2 times 4 and 5 times 3 the second dictionary inside listRep contains {"1":6, "4":2, "3":5}
    # so, if the cluster is composed by c1 and c2, if you execute print(listRep) you should see [{"0":3, "2":4}, {"1":6, "4":2, "3":5}]

    listDef=makeUniqueInd(clustersUnique, allIndexes, listRep)
    return listDef

#this code function is addid my me tointegerate the logic of the article
def merge_clusters(data, clusters):
    merged_clusters = []
    
    for cluster_data in clusters:
        # Step 2: Identify V1, V2, V3
        sum_distances = [sum(euclidean(data[i], data[j]) for j in cluster_data) for i in cluster_data]
        v1_index = cluster_data[np.argmin(sum_distances)]
        distances_to_v1 = [euclidean(data[v1_index], data[i]) for i in cluster_data]
        v2_index = cluster_data[np.argmin(distances_to_v1)]
        distances_to_v1_v2 = [euclidean(data[v1_index], data[i]) for i in cluster_data if i != v2_index]
        v3_index = cluster_data[np.argmin(distances_to_v1_v2)]

        # Step 3: Calculate the radius
        pairwise_distances = [euclidean(data[i], data[j]) for i in cluster_data for j in cluster_data]
        sample_mean = np.mean(pairwise_distances)
        adjusted_std_dev = np.std(pairwise_distances, ddof=1)
        radius = sample_mean + (1.96 * adjusted_std_dev) / np.sqrt(len(cluster_data))

        # Step 4: Merge nearby points within the radius
        merged_cluster = [v1_index, v2_index, v3_index]  # Start with V1, V2, V3
        
        # Iterate over remaining points in the cluster
        for point_index in cluster_data:
            if point_index not in merged_cluster:
                point = data[point_index]
                v2_dist = euclidean(data[v2_index], point)
                v3_dist = euclidean(data[v3_index], point)
                
                if v2_dist <= radius and v3_dist <= radius:
                    merged_cluster.append(point_index)

        merged_clusters.append(merged_cluster)

    return merged_clusters

def doGroup(clusters, rangeN=20):
    groupCluster=[]
    min,max=0,0
    # i could also use min and max function for lists.
    for c in clusters:
        if len(c)==1 and (abs(c[0]-min)<rangeN or abs(c[0]-max)<rangeN or (c[0]>min and c[0]<max)):
            if c[0]>max:
                max=c[0]
            elif c[0]<min:
                min=c[0]
            groupCluster.extend(c)
            clusters.remove(c)
    clusters.append(groupCluster)
    return clusters
# trying to minimize clusters with single indexes, gruoping them by checking if they have len one, 
# check if they're neighbours of min and max element of the list (range 20 by default) or if that number 
# is between the max and min element of the list.
# when i find that condition True i check if i have to update min and max, than i put that element in a 
# new cluster and i update it from the old cluster.

if __name__=="__main__":
    C1=[0,0,0,1,2,3,2,2,3,5,36,4]
    C2=[1,1,1,1,6,7,4,9,3,23,43,32]
    C3=[100,32,32,32,32]
    C4=[0,0,0,0,0,0,0,0,0,0,1]
    C5=[75]
    clusters=[C1, C2, C3, C4,C5]
    # i used that for testing with a little set of clusters, then i used points_combined.csv.

    print("Old Clusters: ", clusters)
    listDef=doEverything(clusters)
    #listDef=doGroup(clusters)
    print("New clusters: ", listDef)
