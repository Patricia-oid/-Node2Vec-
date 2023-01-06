# -*- coding: utf-8 -*-
import networkx as nx
import pandas as pd 
import csv
from kmeans import Kmeans
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']  # to display chinese labels 
plt.rcParams['axes.unicode_minus']=False  # to display symbol minus 

class RelationEmbedding():
    def __init__(self):
        self.p=1  #node2vec回家参数
        self.q=3  #node2vec外出参数
        self.G = nx.DiGraph()
        self.display_graph=nx.DiGraph()
        self.colors=[]
    
    def load_data(self,csv_path,is_weighted):
        data = pd.read_csv(csv_path,encoding='utf-8')
        if(is_weighted):
            for edge in zip(data['head'],data['tail'],data['weight']):
                self.G.add_edge(edge[0],edge[1],weight=edge[2])
                self.display_graph.add_edge(edge[0],edge[1])
        else:
            for edge in zip(data['head'],data['tail']):
                self.G.add_edge(edge[0],edge[1],weight=1)
                self.display_graph.add_edge(edge[0],edge[1])
        #print(self.G)
        plt.figure(figsize=(30,28))
        pos = nx.spring_layout(self.display_graph,seed=5)
        options = {
            "font_size": 6,
            "node_size": 150,
            "node_color": "red",
            "edgecolors": "black",
            "linewidths": 1, 
            "width": 1
        }
        nx.draw_networkx(self.display_graph, pos, **options)
        plt.show()
    
    def embedding_and_clustering(self,num_walks,walk_length,window_size,dimension,epochs,workers):
        embedder=Node2Vec(self.G,self.p,self.q,walk_length,num_walks,dimension)
        embedder.preprocess_transition_probs()
        self.embedding_result=embedder.train(window_size,epochs,workers)
        print(self.embedding_result.shape)
        print("embedding success!")

        best_n=0
        score=0
        for n_clusters in range(5,20):
            cluster=Kmeans(n_clusters,self.embedding_result)
            cluster.load_data()
            cluster_labels=cluster.implement_cluster()
            cur_score=silhouette_score(self.embedding_result,cluster_labels)
            if cur_score>score:
                score=cur_score
                best_n=n_clusters
        
        print(best_n)
        cluster=Kmeans(best_n,self.embedding_result)
        cluster.load_data()
        cluster_labels=cluster.implement_cluster()
        for node in self.G.nodes:
            for i in embedder.index:
                if str(node)==i[0]:
                    self.colors.append(cluster_labels[i[1]])

    def lower_dimension(self):
        tsne = TSNE(n_components=2, n_iter=5000)
        embed_2d = tsne.fit_transform(self.embedding_result)
        plt.figure(figsize=(14,14))
        plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
        plt.show()

    def final_visualize(self):
        plt.figure(figsize=(30,28))
        pos = nx.spring_layout(self.display_graph,seed=5)
        options = {
            "font_size": 6,
            "node_size": 150,
            "edgecolors": "black",
            "linewidths": 1, 
            "width": 1,
            "node_color":self.colors,
            "with_labels":True
        }
        nx.draw(self.display_graph, pos, **options)
        plt.show()

if __name__=='__main__':
    csv_path="data/dreams_of_red_mansion.csv"
    r=RelationEmbedding()
    r.load_data(csv_path,False)
    r.embedding_and_clustering(600,20,3,64,10,4)
    #r.lower_dimension()
    r.final_visualize()

