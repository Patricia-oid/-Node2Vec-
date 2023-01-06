import csv
import numpy.random as random
import time
import numpy as np
from gensim.models import Word2Vec

class Node2Vec(object):
    def __init__(self,graph,p,q,walk_length,num_walks,dimension):
        self.graph=graph
        self.p=p
        self.q=q
        self.walk_length=walk_length
        self.num_walks=num_walks
        self.dimension=dimension
        self.alias_node={}
        self.alias_edge={}
        self.index=[]
    
    def alias_setup(self, probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = [] 
        larger = [] 
        for i, prob in enumerate(probs):
            q[i] = K * prob  #
            if q[i] < 1.0:
                smaller.append(i)
            else:
                larger.append(i)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large  
            q[large] = q[large] - (1.0 - q[small])  #

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def get_alias_edge(self,src,des):
        probs = []
        for nbr in sorted(self.graph.neighbors(des)):
            if nbr==src:
                probs.append(self.graph[des][nbr]['weight']/self.p)
            elif self.graph.has_edge(nbr, src):
                probs.append(self.graph[des][nbr]['weight'])
            else:
                probs.append(self.graph[des][nbr]['weight']/self.q)
        normalized_probs = [float(prob)/sum(probs) for prob in probs]
        return self.alias_setup(normalized_probs)
    
    def alias_draw(self, J, q):
        K = len(J)

        kk = int(np.floor(random.rand() * K))  # random

        if random.rand() < q[kk]:  # compare
            return kk
        else:
            return J[kk]

    def preprocess_transition_probs(self):
        graph = self.graph

        alias_nodes = {}
        for node in graph.nodes():
            unnormalized_probs = [graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(unn_prob) / norm_const for unn_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}

        #is_directed
        for edge in graph.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    #generate random walk sequence for each selected node
    def walk(self, walk_length, start_node):
        graph = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk_res = [start_node]

        while len(walk_res) < walk_length:
            cur = walk_res[-1]
            cur_nbrs = sorted(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk_res) == 1:
                    walk_res.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                #if there are previous nodes in the sequence
                else:
                    prv = walk_res[-2]
                    walk_res.append(cur_nbrs[self.alias_draw(alias_edges[(prv, cur)][0], alias_edges[(prv, cur)][1])])
            else:
                break
        return walk_res

    #generate random walk sequences
    def simulate_walks(self):
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())
        for i in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(walk_length=self.walk_length, start_node=node))
        return walks

    #transfer from Word2Vec to Node2Vec
    def train(self,window_size,epochs,workers):
        walks=self.simulate_walks()
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, vector_size=self.dimension, window=window_size, min_count=1, sg=1, workers=workers,batch_words=4,epochs=epochs)
        for node in list(self.graph.nodes):
            idx= model.wv.key_to_index[str(node)]
            self.index.append([str(node),idx])
        return model.wv.vectors
