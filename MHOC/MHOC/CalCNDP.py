import networkx as nx
import numpy as np
import math
import copy 

class CNDP:
    def __init__(self, path):
        self.initial_G = nx.read_edgelist(path)
        self.nodes = sorted(self.initial_G.nodes)
        self.test_set = []
        self.nonexistent_set = sorted(nx.non_edges(self.initial_G))
        self.β = 2.5
        
        # 计算测试集个数
        edges = sorted(self.initial_G.edges)
        initial_edges_number = len(edges)
        self.test_number = math.floor(0.1 * initial_edges_number)
        
        # 划分测试集（从训练集中删除10%的边作为测试集，剩余的作为训练集）
        self.train_G = copy.deepcopy(self.initial_G)
        idx = np.random.permutation(np.arange(initial_edges_number))
        shuffled_edges = np.array(edges)[idx]
        shuffled_edges_list = shuffled_edges.tolist()
        delete_dege = []
        for index in range(self.test_number):
            self.train_G.remove_edge(shuffled_edges_list[index][0], shuffled_edges_list[index][1])
            delete_dege.append(shuffled_edges_list[index])
            self.test_set = sorted(delete_dege)
        
        #计算平均聚类系数
        node_clustering = []
        total = 0
        for node in self.nodes:
            node_clustering.append(nx.clustering(self.train_G, node))
        for i in range(0, len(node_clustering)):
            total += node_clustering[i]
        self.average_clustering = total / len(node_clustering)

        self.βC = self.β * self.average_clustering        
            
      # 为训练集中不存在边的每对节点计算相似度分数  
    def predict(self):
        self.similar_score = {}
        non_edges = sorted(nx.non_edges(self.train_G))
        for edge in non_edges:
            common_neighbors = sorted(nx.common_neighbors(self.train_G, edge[0], edge[1]))
            if len(common_neighbors) != 0:
                score = 0
                for node in common_neighbors:
                    node_neighbors = list(self.train_G.neighbors(node))
                    res = list(set(common_neighbors) & set(node_neighbors))
                    C = len(res)
                    score += C / self.train_G.degree(node) ** self.βC
                self.similar_score[edge] = score
                    
            else:
                self.similar_score[edge] = 0
        
        return self.similar_score
                

        # 计算AUC，PR 
    def compute_score(self):
        similar_score_sort = sorted(self.similar_score.items(), key = lambda x: x[1], reverse = True)
        predicted_set = []
        for i in range(self.test_number):
            predicted_set.append(similar_score_sort[i][0])    
        correct_predict = [v for v in predicted_set if v in self.test_set]
        PR = len(correct_predict) / len(predicted_set)
        
        
        n = len(self.nonexistent_set) * len(self.test_set)
        n1 = 0
        n2 = 0
        for active_edge in self.test_set:
            for negative_edge in self.nonexistent_set:
                if self.similar_score[tuple(active_edge)] > self.similar_score[negative_edge]:
                    n1 += 1
                elif self.similar_score[tuple(active_edge)] == self.similar_score[negative_edge]:
                    n2 += 1
                    
        AUC = (n1 + 0.5 * n2) / n
        
        return PR, AUC
        
        
        
        
        
        
       


        

        
        