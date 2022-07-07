import numpy as np
import networkx as nx
import random
from tqdm import tqdm

from gensim.models import Word2Vec


class deepwalk_model:
    @staticmethod
    def read_edge(path):
        edges = np.loadtxt(path, dtype=np.int16)
        edges = [(edges[i, 0], edges[i, 1]) for i in range(edges.shape[0])]
        return edges

    def __init__(self, node_num: int, path: str, undirected=False) -> None:
        if undirected:
            self.G = nx.Graph()  # 创建无向图
        else:
            self.G = nx.DiGraph()  # 创建有向图
        self.G.add_nodes_from(list(range(node_num)))  # 加入节点
        edges = self.read_edge(path)
        self.G.add_edges_from(edges)  # 加入边
        self.adjacency = np.array(nx.adjacency_matrix(self.G).todense())  # 生成邻接矩阵(多维列表)
        self.G_neighbor = {}
        for i in range(self.adjacency.shape[0]):  # i代表当前节点编号(shape[0]代表一维长度→节点个数)
            self.G_neighbor[i] = []
            for j in range(self.adjacency.shape[0]):
                if self.adjacency[i, j] > 0:  # 获取邻居节点信息
                    self.G_neighbor[i].append(j)

    def random_walk(self, path_len, alpha=0, rand_iter=random.Random(9931), start=None):
        G = self.G_neighbor
        if start:
            rand_path = [start]
        else:
            rand_path = [rand_iter.choice(list(G.keys()))]  # 从列表中随机选择一个位置位置
        while len(rand_path) < path_len:
            current_pos = rand_path[-1]
            if len(G[current_pos]) > 0:
                if rand_iter.random() >= alpha:
                    rand_path.append(rand_iter.choice(G[current_pos]))  # 从当前位置寻找一个相邻节点
                else:
                    rand_path.append(rand_path[0])  # 重新开始随机游走(处理方式也可以随机选择一个点重新开始)
            else:
                rand_path.append(rand_iter.choice(list(G.keys())))  # 处理死区问题,防止陷入单点聚集
                break
        return [str(node) for node in rand_path]

    def build_total_corpus(self, num_paths, path_length, alpha=0, rand_iter=random.Random(9931)):
        print("Start randomwalk.")
        total_walks = []
        G = self.G_neighbor
        nodes = list(G.keys())
        for cnt in tqdm(range(num_paths)):  # num_path为训练随机游走队列组数,每次对于所有节点进行随机游走测试
            rand_iter.shuffle(nodes)
            for node in nodes:
                total_walks.append(self.random_walk(path_length, rand_iter=rand_iter, alpha=alpha, start=node))
        return total_walks

    @staticmethod
    def train_deepwalk(total_walks, window_size=3, output=""):
        print("Training deepwalk.")
        model = Word2Vec(total_walks, window=window_size, min_count=0, sg=1, hs=1, workers=8)
        model.wv.save_word2vec_format(output)


if __name__ == '__main__':
    node_num = 100
    edge_test = 'text.txt'
    walker = deepwalk_model(node_num=node_num, path=edge_test)
    deepwalk_corpus = walker.build_total_corpus(2, 10)
    walker.train_deepwalk(deepwalk_corpus, window_size=3, output='test.embeddings')
