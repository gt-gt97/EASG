import dgl
import torch as th

from dgl.nn.pytorch import nn, HeteroGraphConv, GraphConv
import torch.nn.functional as F

class FirstGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = HeteroGraphConv({
            'M_M': GraphConv(in_feats, out_feats),
            'M_E': GraphConv(in_feats, out_feats),
            'M_S': GraphConv(in_feats, out_feats),
            'E_S': GraphConv(in_feats, out_feats),
            'E_M': GraphConv(in_feats, out_feats),
            'S_M': GraphConv(in_feats, out_feats),
            'S_E': GraphConv(in_feats, out_feats),
            'S_S': GraphConv(in_feats, out_feats),
            'E_E': GraphConv(in_feats, out_feats)},
            aggregate='sum')
        # self.conv2 = HeteroGraphConv({
        #     'M_M': GraphConv(hid_feats, out_feats),
        #     'M_E': GraphConv(hid_feats, out_feats),
        #     'M_S': GraphConv(hid_feats, out_feats),
        #     'E_S': GraphConv(hid_feats, out_feats),
        #     'E_M': GraphConv(hid_feats, out_feats),
        #     'S_M': GraphConv(hid_feats, out_feats),
        #     'S_E': GraphConv(hid_feats, out_feats),
        #     'S_S': GraphConv(hid_feats, out_feats),
        #     'E_E': GraphConv(hid_feats, out_feats)},
        #     aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典，{"第一类节点"：[x*n],"第二类节点"：[y*n]},x,y位节点类型的数量，n是特征维度。
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv2(graph, h)
        return h

class SecondGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = HeteroGraphConv({
            'M_M': GraphConv(in_feats, out_feats),
            'M_E': GraphConv(in_feats, out_feats),
            'M_S': GraphConv(in_feats, out_feats),
            'E_S': GraphConv(in_feats, out_feats),
            'E_M': GraphConv(in_feats, out_feats),
            'S_M': GraphConv(in_feats, out_feats),
            'S_E': GraphConv(in_feats, out_feats),
            'S_S': GraphConv(in_feats, out_feats),
            'E_E': GraphConv(in_feats, out_feats)},
            aggregate='sum')
        # self.conv2 = HeteroGraphConv({
        #     'M_M': GraphConv(hid_feats, out_feats),
        #     'M_E': GraphConv(hid_feats, out_feats),
        #     'M_S': GraphConv(hid_feats, out_feats),
        #     'E_S': GraphConv(hid_feats, out_feats),
        #     'E_M': GraphConv(hid_feats, out_feats),
        #     'S_M': GraphConv(hid_feats, out_feats),
        #     'S_E': GraphConv(hid_feats, out_feats),
        #     'S_S': GraphConv(hid_feats, out_feats),
        #     'E_E': GraphConv(hid_feats, out_feats)},
        #     aggregate='sum')

    def forward(self, graph, inputs, edge_weight=None):
        mod_kwargs = {"M_M": {'edge_weight': edge_weight[('mention', 'M_M', 'mention')]},
                      "M_E": {'edge_weight': edge_weight[('mention', 'M_E', 'entity')]},
                      'M_S': {'edge_weight': edge_weight[('mention', 'M_S', 'sentence')]},
                      'E_S': {'edge_weight': edge_weight[('entity', 'E_S', 'sentence')]},
                      'E_M': {'edge_weight': edge_weight[('entity', 'E_M', 'mention')]},
                      'S_M': {'edge_weight': edge_weight[('sentence', 'S_M', 'mention')]},
                      'S_E': {'edge_weight': edge_weight[('sentence', 'S_E', 'entity')]},
                      'S_S': {'edge_weight': edge_weight[('sentence', 'S_S', 'sentence')]},
                      'E_E': {'edge_weight': edge_weight[('entity', 'E_E', 'entity')]}}
        # self.conv1.eval()
        # 输入是节点的特征字典，{"第一类节点"：[x*n],"第二类节点"：[y*n]},x,y位节点类型的数量，n是特征维度。
        h = self.conv1(graph, inputs, mod_kwargs=mod_kwargs)
        h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv2(graph, h, mod_kwargs=mod_kwargs)
        return h
