import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    # 输入的特征数以及输出特征数
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # 增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print(edge_index)
        # Step 2: Linearly transform node feature matrix.
        # 线性层实现（可以有多层GCN）
        # W权重矩阵的维度是[out_channels,in_channels]
        # 得到的x维度是[节点数,输出特征维度]
        x = self.lin(x)
        print(x)
        # Step 3: Compute normalization.
        # 计算归一化
        # row，col表示edge_index第一行和第二行

        row, col = edge_index
        # 保存度矩阵（包含自环）主对角线的数据，如tensor([4., 1., 1., 1.])
        deg = degree(col, x.size(0), dtype=x.dtype)
        print(deg)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # The result is saved in the tensor norm of shape [num_edges, ]
        # norm的维度是[1行num_edges(包含自环边)列]
        # 如tensor([0.5000, 0.5000, 0.5000, 0.2500, 1.0000, 1.0000, 1.0000])
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        print(deg_inv_sqrt[row])
        print(deg_inv_sqrt[col])
        print(norm)
        # Step 4-5: Start propagating messages.
        # 开启消息传递
        return self.propagate(edge_index, x=x, norm=norm)

    # x_j是修改后的特征矩阵（）
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        # 转化维度 [1行num_edges(包含自环边)列]转化为[num_edges(包含自环边)行1列]。
        print(norm.view(-1, 1) * x_j)
        # 最后的结果是num_edges(包含自环边)行，out_channels（输出特征维度）列
        return norm.view(-1, 1) * x_j
