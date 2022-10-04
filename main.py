import json
import os

import torch
from torch_geometric.data import Data

from src.code.model.GCNConv import GCNConv

if __name__ == '__main__':
    # edge_index = torch.tensor([[1, 2, 3],
    #                            [0, 0, 0]],
    #                            dtype=torch.long)
    # x = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)
    # data = Data(x=x, edge_index=edge_index)
    # gcn_conv = GCNConv(1, 2)
    # # 得到的输出是 每个节点的增强表征。
    # ret = gcn_conv(x, edge_index)
    # print(ret)
    # print(ret)
    with open(os.path.join('dataset/', '100_lower-doc/small_train.json'), 'r') as fh:
        row_data = json.load(fh)
    # 读取每一条json数据
    for index, d in enumerate(row_data):
        # if index == 0:
        #     print(d["vertexSet"])
        #     print(d['labels'])
        # else:
        #     break
        if index == 0:
            print(d['pos_ins'])
            print(d['neg_ins'])
