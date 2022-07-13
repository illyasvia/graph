import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import copy


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        # input_dim=输入层特征数,hidden_dim=隐藏层特征数,output_dim=输出层特征数
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()  # 存储多个不同的卷积
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(input_dim, hidden_dim))  # 建立隐藏层卷积
            input_dim = hidden_dim
        self.convs.append(GCNConv(hidden_dim, output_dim))  # 建立输出层卷积
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers - 1)])  # 网络参数进行归一化处理
        self.softmax = torch.nn.LogSoftmax()  # 选择归一化函数
        self.dropout = dropout  # 产生dropout几率
        self.return_embeds = return_embeds

    def reset_parameters(self):  # 该函数用于重置网络层参数
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = None
        for layer in range(len(self.convs) - 1):  # 对于隐藏层前向传播
            x = self.convs[layer](x, adj_t)
            x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        out = self.convs[-1](x, adj_t)
        if not self.return_embeds:  # 输出层
            out = self.softmax(out)
        return out


def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    train_output = out[train_idx]
    train_label = data.y[train_idx, 0]
    loss = loss_fn(train_output, train_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc


if __name__ == '__main__':
    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    split_idx = dataset.get_idx_split()  # 将数据集分成了train，valid，test三部分
    train_idx = split_idx['train'].to(device)

    args = {  # 设置超参数
        'device': device,
        'num_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.5,
        'lr': 0.01,
        'epochs': 100,
    }
    model = GCN(data.num_features, args['hidden_dim'],
                dataset.num_classes, args['num_layers'],
                args['dropout']).to(device)
    evaluator = Evaluator(name='ogbn-arxiv')
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = F.nll_loss
    best_model = None
    best_valid_acc = 0
    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, data, train_idx, optimizer, loss_fn)
        result = evaluate(model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')

        best_result = evaluate(best_model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = best_result
        print(f'Best model: '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
