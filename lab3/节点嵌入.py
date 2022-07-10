# 目标:分离有边连接点乘趋向于1,无边连接点乘趋向于0
import random
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.optim import SGD

G = nx.karate_club_graph()


def graph_to_edge_list():
    edge_list = []
    for edge in G.edges():
        edge_list.append(edge)
    return edge_list


def edge_list_to_tensor(edge_list):
    edge_index = torch.LongTensor(edge_list).t()
    return edge_index


def sample_negative_edges(num_neg_samples):
    neg_edge_list = []
    non_edges = list(enumerate(nx.non_edges(G)))  # 枚举所有不存在的边(形式类似于[(0, (0, 32)), (1, (0, 33))])
    list_indices = random.sample(range(0, len(non_edges)), num_neg_samples)  # 第一个参数为随机生成数范围,第二个参数为随机生成数数量
    for i in list_indices:
        neg_edge_list.append(non_edges[i][1])  # 采样不存在的边作为负边
    return neg_edge_list


def create_node_emb(num_node=34, embedding_dim=16):  # num_node为节点数量，embedding_dim对应嵌入向量大小
    emb = nn.Embedding(num_node, embedding_dim)
    emb.weight.data = torch.rand(num_node, embedding_dim)  # 生成均匀分布张量映射
    return emb


def visualize_emb(emb):
    X = emb.weight.data.numpy()  # 获取映射权重
    pca = PCA(n_components=2)  # 通过PCA进行降维（参数为降维目标）
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))  # 创建指定画布
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes(data=True):  # data=True用于获取对应标签([(0, {'club': 'Mr. Hi'}), (1, {'club': 'Mr. Hi'})]形式)
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")  # 生成对应散点图
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.show()


def accuracy(pred, label):  # 统计准确性
    accu = round(((pred > 0.5) == label).sum().item() / (pred.shape[0]), 4)
    return accu


def train(emb, train_label, train_edge):
    epochs = 5000  # 学习次数
    learning_rate = 0.1  # SGD学习率
    optimizer = SGD(emb.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    for i in range(epochs):
        optimizer.zero_grad()  # 清除各点梯度
        train_node_emb = emb(train_edge)  # 将矩阵映射为一维张量
        dot_product_result = train_node_emb[0].mul(train_node_emb[1])  # 将所有节点位置信息(x,y)数值
        dot_product_result = torch.sum(dot_product_result, 1)
        sigmoid_result = sigmoid(dot_product_result)  # 归一化结果
        loss_result = loss_fn(sigmoid_result, train_label)  # 获取归一化结果和训练级损失函数
        loss_result.backward()
        optimizer.step()
        if i % 10 == 0:
            print(loss_result)
            print(accuracy(sigmoid_result, train_label))


if __name__ == '__main__':
    torch.manual_seed(1)
    emb = create_node_emb()
    visualize_emb(emb)
    edge_list = graph_to_edge_list()
    edge_index = edge_list_to_tensor(edge_list)
    neg_edge_list = sample_negative_edges(len(edge_list))
    neg_edge_index = edge_list_to_tensor(neg_edge_list)
    label = torch.ones(edge_index.shape[1], )
    neg_label = torch.zeros(neg_edge_index.shape[1], )
    train_label = torch.cat([label, neg_label], dim=0)  # 用于获取训练集标签(cat为拼接函数)
    train_edge = torch.cat([edge_index, neg_edge_index], dim=1)  # 用于执行训练使用的边
    train(emb, train_label, train_edge)
    visualize_emb(emb)
