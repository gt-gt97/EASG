import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HeteroConv, GATConv, SAGEConv
from torch_geometric.datasets import Planetoid

if __name__ == '__main__':
    with open('../../../dataset/100_lower-doc/small_train.json', 'r') as fh:
        documents = json.load(fh)
    with open('../../../dataset/train_annotated.json', 'r') as train:
        train_file = json.load(train)
    documents0 = documents[0]
    train_file0 = train_file[0]
    print(train_file0)
    # 模拟全文的词向量
    documentsToVec = torch.randn(documents0['doc_len'], 4)
    # print(documentsToVec)
    dic = {}
    res = []
    mention_all = []
    entity_all = []
    sent_len = len(documents0['doc_sent_bound'])
    for index, entity in enumerate(train_file0['vertexSet']):
        temp = []
        entity_temp = [0] * documents0['doc_len']
        for mention_index, mention in enumerate(entity):
            value = 1 / (mention['pos'][1] - mention['pos'][0])
            pos = documents0['doc_sent_bound'][mention['sent_id']][0]
            list1 = [0] * (pos+mention['pos'][0]) + (mention['pos'][1] - mention['pos'][0]) * [value] + (
                    documents0['doc_len'] - (mention['pos'][1] - mention['pos'][0]) - (pos+mention['pos'][0])) * [0]
            temp.append(list1)
            mention_all.append(list1)
            k = 1 / len(entity)
            for i in range(0, len(list1)):
                entity_temp[i] = k * list1[i] + entity_temp[i]
        entity_all.append(entity_temp)
        res.append(temp)
    # 异构图节点的过渡表示
    dic['mention_entity_mask'] = res
    dic['mention_all_mask'] = mention_all
    dic['entity_all_mask'] = entity_all
    res = []
    for sentences in documents0['doc_sent_bound']:
        temp = []
        value = 1 / (sentences[1] - sentences[0])
        temp.append([0] * sentences[0] + (sentences[1] - sentences[0]) * [value] + (
                documents0['doc_len'] - sentences[1]) * [0])
        res.append(temp)
    dic['sentences_mask'] = res
    # 获取异构图的边M-M(包含自反边)===等同于无向图
    # M_M = []
    # for around in documents0['doc_sent_bound']:
    #     bound = [0] * (around[1] - around[0])
    #     for i, val_i in enumerate(dic['mention_all_mask']):
    #         for j, val_j in enumerate(dic['mention_all_mask']):
    #             if i != j:
    #                 temp_i = val_i[around[0]:around[1]]
    #                 temp_j = val_j[around[0]:around[1]]
    #                 if bound != temp_j and bound != temp_j:
    #                     M_M.append([i, j])
    M_E = []
    index = 0
    for i, entity in enumerate(dic['mention_entity_mask']):
        for mention in entity:
            M_E.append([index, i])
            index = index+1
    M_S = []
    E_S = []
    index = 0
    for i, entity in enumerate(train_file0['vertexSet']):
        for mention in entity:
            M_S.append([index, mention['sent_id']])
            index = index+1
            if [i,  mention['sent_id']] not in E_S:
                E_S.append([i,  mention['sent_id']])
    S_S = []
    for i in range(0, sent_len):
        if i != (sent_len-1):
            S_S.append([i, i+1])
            S_S.append([i+1, i])
        else:
            S_S.append([sent_len-1, 0])
            S_S.append([0, sent_len-1])
    M_M = []
    for i in range(0, sent_len):
        for j, m_s_start in enumerate(M_S):
            if m_s_start[1] == i:
                for k in range(j+1, len(M_S)):
                    if M_S[k][1] == i:
                        M_M.append([j, k])
                        M_M.append([k, j])

    print(torch.tensor(dic['mention_all_mask']).size())
    ddd = [[1, 2, 3], [4, 6]]
    print(torch.tensor(ddd))


    # 声明神经网络
    # hetero_conv = HeteroConv({
    #     ('paper', 'cites', 'paper'): GCNConv(-1, 64),
    #     ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
    #     ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
    # }, aggr='sum')
    #
    # #
    # # 实例化
    # data = HeteroData()
    #
    # # 设置4种结点类型，[行是paper类型的节点数量，列是paper类型的节点表征维度]，也就是每行都是一个节点表示
    # data['paper'].x = torch.Tensor([[]],
    #                                [[]],
    #                                )  # [num_papers, num_features_paper]
    # data['author'].x = ...  # [num_authors, num_features_author]
    # data['institution'].x = ...  # [num_institutions, num_features_institution]
    # data['field_of_study'].x = ...  # [num_field, num_features_field]
    #
    # # 设置对应的边类型，以13行为例，例如[[0,1],[0,2]，...]代表paper类型的0号结点和paper类型的1号结点有一条cites类型的边;paper类型的0号结点和paper类型的2号结点有一条cites类型的边
    # data['paper', 'cites', 'paper'].edge_index = ...  # [2, num_edges_cites]
    # data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges_writes]
    # data['author', 'affiliated_with', 'institution'].edge_index = ...  # [2, num_edges_affiliated]
    # data['author', 'has_topic', 'institution'].edge_index = ...  # [2, num_edges_topic]
    #
    # # 设置对应边的表示（这个是可选的，暂时不用）
    # data['paper', 'cites', 'paper'].edge_attr = ...  # [num_edges_cites, num_features_cites]
    # data['author', 'writes', 'paper'].edge_attr = ...  # [num_edges_writes, num_features_writes]
    # data['author', 'affiliated_with', 'institution'].edge_attr = ...  # [num_edges_affiliated, num_features_affiliated]
    # data['paper', 'has_topic', 'field_of_study'].edge_attr = ...  # [num_edges_topic, num_features_topic]
    # #
    # out_dict = hetero_conv(x_dict, edge_index_dict)
