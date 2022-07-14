import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class NeighborAggregator(nn.Module):  # 聚合邻居节点
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method='mean'):
        # input_dim输入层，output_dim输出层，aggr_method聚合方式，use_bias偏置矩阵
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        assert neighbor_feature in ['mean', 'sum', 'max']
        if self.aggr_method == 'mean':
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == 'sum':
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == 'max':
            aggr_neighbor = neighbor_feature.max(dim=1)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)  # 邻居节点特征经过线性变换得到隐藏层
        if self.use_bias:  # 增加矩阵偏置
            neighbor_hidden += self.bias
        return neighbor_hidden


class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10, base_model=None,
                 gcn=False, cuda=False, feature_transform=False):
        # feature=特征矩阵,feature_dim=特征矩阵维度,embed_dim=输出维度,adj_lists=输入临界矩阵,aggregator聚合函数,num_sample=采样数目
        super(Encoder, self).__init__()
        self.feature_transform = feature_transform
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model is not None:
            self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)  # 聚合采样结果
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)  # 加入自身属性信息
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))  # 使用激活函数
        return combined


class SupervisedGraphSage(nn.Module):  # 有监督学习
    def __init__(self, num_classes, encoder):
        super(SupervisedGraphSage, self).__init__()
        self.encoder = encoder
        self.loss_function = nn.CrossEntropyLoss()  # 损失函数
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, encoder.embed_dim))
        init.xavier_uniform(self.weight)  # 均匀分布初始化

    def forward(self, nodes):
        embeds = self.encoder(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.loss_function(scores, labels.squeeze())


if __name__ == '__main__':
    pass
