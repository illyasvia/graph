import numpy as np
import numpy.random as npr
from gensim.models import Word2Vec
import networkx as nx
import matplotlib.pyplot as plt


class node2vec:
    def __init__(self, p, q, r, d, k, G):
        self.G = G
        self.r = r
        self.d = d
        self.k = k
        self.init_transition_prob(p, q)

    def init_transition_prob(self, p, q):
        g = self.G
        nodes_info, edges_info = {}, {}
        for node in g.nodes:  # nodes代表获取所有节点集合
            nbs = sorted(g.neighbors(node))  # neighbors代表获取对应接节点邻居节点集合
            weights = [g[node][n]['weight'] for n in nbs]  # 获取所有权重结合
            norm = sum(weights)
            normalized_weights = [float(n) / norm for n in weights]  # 归一化所有路径权重
            nodes_info[node] = self.alias_setup(normalized_weights)  # Alias Method进行随机取样

        for edge in g.edges:  # edge对应是类似(x,y)组合
            if g.is_directed():
                edges_info[edge] = self.get_alias_edge(edge[0], edge[1], p, q)
            else:
                edges_info[edge] = self.get_alias_edge(edge[0], edge[1], p, q)
                edges_info[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], p, q)

        self.nodes_info = nodes_info
        self.edges_info = edges_info

    def get_alias_edge(self, x, y, p, q):
        g = self.G
        unnormalized_probs = []
        for k in sorted(g.neighbors(y)):
            if k == x:  # 返回上一次节点
                unnormalized_probs.append(g[y][k]['weight'] / p)
            elif g.has_edge(k, x):  # 进入相邻节点
                unnormalized_probs.append(g[y][k]['weight'])
            else:  # 进入非相邻节点
                unnormalized_probs.append(g[y][k]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]  # 归一化
        return self.alias_setup(normalized_probs)  # 生成随机游走的采样序列

    @staticmethod
    def alias_setup(weights):
        length = len(weights)
        upper = np.zeros(length)
        down = np.zeros(length, dtype=np.int64)
        smaller = []  # 记录所有低于平均可能性的概率分布
        larger = []  # 记录所有高于平均可能性的概率分布

        for i, weight in enumerate(weights):
            upper[i] = length * weight  # 使prob值在1上下区间浮动
            if upper[i] < 1.0:
                smaller.append(i)
            else:
                larger.append(i)

        while len(smaller) > 0 and len(larger) > 0:  # 通过评测使得每一分组可能性之和均为1
            x = smaller.pop()
            y = larger.pop()
            down[x] = y  # 底层每次选择小可能性概率分布
            upper[y] = upper[y] - (1.0 - upper[x])  # 顶层每次选择与底层之和为1的概率分布
            if down[y] < 1.0:  # 将修改后的down[y]重新放入分配数组
                smaller.append(y)
            else:
                larger.append(y)
        return down, upper

    @staticmethod
    def alias_draw(down, upper):
        length = len(down)
        random = int(np.floor(npr.rand() * length))  # 随机选取采样列表其中一列
        if npr.rand() < upper[random]:  # 随机选择采样列表其中一种可能性
            return random
        else:
            return down[random]

    def node2vecWalk(self, start, len):
        g = self.G
        walk = [start]
        nodes_info, edges_info = self.nodes_info, self.edges_info
        while len(walk) < len:
            curr = walk[-1]  # 当前位置
            v_curr = sorted(g.neighbors(curr))
            if len(v_curr) > 0:
                if len(walk) == 1:  # 起始状态随机选取一个节点
                    walk.append(v_curr[self.alias_draw(nodes_info[curr][0], nodes_info[curr][1])])
                else:  # 存在pre记录之后采用node2vec方式进行随机游走
                    prev = walk[-2]
                    ne = v_curr[self.alias_draw(edges_info[(prev, curr)][0], edges_info[(prev, curr)][1])]
                    walk.append(ne)
            else:
                break
        return walk

    def learning_features(self, len): # 采用word2vec对应随机游走产生数据进行训练
        walks = []
        g = self.G
        nodes = list(g.nodes())
        for t in range(self.r):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vecWalk(node, len)
                walks.append(walk)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(sentences=walks, vector_size=self.d, window=self.k, min_count=0, sg=1, workers=3)
        f = model.wv
        res = [f[x] for x in nodes]
        return res


if __name__ == '__main__':
    p, q = 1, 0.5
    d, r, l, k = 128, 10, 80, 10
    G = nx.les_miserables_graph()
    node2vec = node2vec(p, q, r, d, k, G)
    model = node2vec.learning_features(l)
