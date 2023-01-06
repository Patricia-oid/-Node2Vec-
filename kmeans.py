# -*- coding: utf-8 -*-
import math
import random
#an implementation of K-means++

class Kmeans:
    #the number of k needs to be tested
    def __init__(self,n_clusters,x):
        self.dataset=[]
        self.x=x
        self.nums=x.shape[0]
        self.K=n_clusters
        self.dimension=x.shape[1]

    # Euclidean distance
    def distance(self,l1, l2):
        res = 0
        for i in range(self.dimension):
            res += (l1[i] - l2[i]) * (l1[i] - l2[i])
        return math.sqrt(res)

    def load_data(self):
        for i in range(self.nums):
            self.dataset.append(self.x[i])

    def implement_cluster(self):
        cluster_centroids=[]
        cluster_ids=[]
        state=[-1 for i in range(self.nums)]

        init_point=random.randint(0,self.nums-1)

        cluster_centroids.append(self.dataset[init_point])
        cluster_ids.append([init_point])
        state[init_point]=0

        candidates=[9999999 for i in range(self.nums)]
        while len(cluster_ids)<self.K:
            candidates= [min(float(candidates[i]),self.distance(self.dataset[i],cluster_centroids[-1])) for i in range(self.nums)]
            next_one=candidates.index(max(candidates))
            cluster_centroids.append(self.dataset[next_one])
            cluster_ids.append([next_one])
            state[next_one]=len(cluster_ids)-1


        change=True
        counter=0
        while change:
            change = False
            counter+=1
            for id in range(self.nums):
                coordinate=self.dataset[id]
                distances=[self.distance(coordinate,centroid_cor) for centroid_cor  in cluster_centroids]
                new_cluster=distances.index(min(distances))
                if state[id]!=new_cluster:
                    change=True
                    if state[id]!=-1:
                        cluster_ids[state[id]].remove(id)
                    state[id]=new_cluster
                    cluster_ids[new_cluster].append(id)
            for c in range(self.K):
                new_cor=[0 for i in range(100)]
                numbers=len(cluster_ids[c])
                for PID in cluster_ids[c]:
                    PID_cor=self.dataset[PID]
                    new_cor=[new_cor[i]+PID_cor[i] for i in range(self.dimension)]
                new_cor=[new_cor[i]/numbers for i in range(self.dimension)]
                cluster_centroids[c]=new_cor
        return state