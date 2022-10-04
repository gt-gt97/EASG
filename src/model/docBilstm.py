"""
GCN model for relation extraction.
"""
import copy
import heapq
import math
import profile

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.autograd import Variable
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from BiLSTM import EncoderLSTM
from localEncoder import LocalEncoder
from classifier_new import EntityClassifier
import sys
from ESAG import FirstGCN, SecondGCN
from Attention import DotProductAttention



def get_evidence(all_sample_head_tail, sentences, k):
    return torch.topk(torch.matmul(all_sample_head_tail, torch.transpose(sentences, 0, 1)).sum(dim=1), dim=1, k=k)

class DocClassifier(nn.Module):
    # self表示全局的意思，便于设置全局变量。
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt

        # Doc Embedding
        # 参数分别为词典的大小尺寸、嵌入向量的维度、填充0
        # nn.Embedding用于存储单词嵌入, 并使用索引检索它们。模块的输入是索引列表, 输出是相应的词嵌入。
        # 给一个编号(索引)，嵌入层就能返回这个编号对应的嵌入向量
        # 作为训练的一层，随模型训练得到适合的词向量。一开始什么都没有，初始化填充0
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=0)
        # print("self.emb")
        # print(self.emb.weight)
        # 词向量矩阵进行初始化（固定矩阵前topk行，后面的行设置为0）。
        self.init_embeddings(emb_matrix)

        # （三元表达式）参数分别为实体类型、命名实体维度
        # 实体类型嵌入层。定义词表长度是7，维度是20，使用的时候使用下标即可。
        self.ner_embed = nn.Embedding(opt['ner_num'], opt['ner_dim']) if opt['ner_dim'] > 0 else None
        # print("嵌入层")
        # 测试
        # print(self.ner_embed(torch.tensor(1)))
        # 共指嵌入层。定义词表是max_len（512），维度是20，使用的时候使用下标即可。
        self.coref_embed = nn.Embedding(opt['max_len'], opt['coref_dim']) if opt['coref_dim'] > 0 else None
        # 定义dropout层，丢弃率是input_dropout
        # 随机将输入张量设置为0。对于每次前向调用，被置0的通道都是随机的。
        self.in_drop = nn.Dropout(opt['input_dropout'])

        # Global Encoder
        # 包含词嵌入，实体类型嵌入，共指嵌入的向量维度。（维度相加）
        in_dim = opt['emb_dim'] + opt['ner_dim'] + opt['coref_dim']
        # input_size, num_units, nlayers, concat, bidir, dropout, return_last, require_h=False
        # 设置自定义LSTM实例。a // 2 表示a除2之后向下取整。
        self.global_encoder = EncoderLSTM(in_dim, opt['hidden_dim'] // 2, 1, True, True, opt['rnn_dropout'], False,
                                          True)
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        # hidden_channels, out_channels, num_layers
        self.firstGNN = FirstGCN(256, 512, 256)
        self.attention = DotProductAttention(dropout=0.5)
        self.secondGNN = SecondGCN(256, 512, 256)
        self.GAT = GATConv(256, 256, 2)
        # 设置自定义局部编码器实例。
        # Local Encoder[GCN-抽取证据句-GCN]
        self.local_encoder = LocalEncoder(in_dim, opt['hidden_dim'], opt['input_dropout'], opt['rnn_dropout'],
                                          self.emb, self.ner_embed, self.coref_embed, opt['max_len'], opt['pos_dim'])

        # 设置自定义分类器实例。
        self.entity_classifier = EntityClassifier(opt['hidden_dim'], opt['num_class'], opt['mlp_dropout'])

    def init_embeddings(self, emb_matrix):
        # 决定word embedding中的哪些部分更新，但实际上只有UNK（低频词或未在词表中的词）会更新
        def keep_partial_grad(grad, topk):
            """
            Keep only the topk rows of grads.
            """

            assert topk < grad.size(0)
            # topk后面的梯度都设置为0，那么就不更新了。
            grad.data[topk:].zero_()
            # print("grad")
            # print(grad)
            return grad

        if emb_matrix is None:
            # 对词向量矩阵进行填充（均匀分布抽样进行），填充-1到1的数。填充到矩阵中的非第一行。
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            # 矩阵(标量)转化为向量（张量）
            emb_matrix = torch.from_numpy(emb_matrix)
            # 把glove词向量矩阵设置nn.Embedding层的矩阵（初始化）。
            self.emb.weight.data.copy_(emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            # 不做微调了(固定embedding)
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        # topn小于词表数量
        elif self.opt['topn'] < self.opt['vocab_size']:
            #
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            # 每次计算张量的梯度时，都会调用该钩子函数。x是计算出来的梯度。（第一次因为没计算梯度，所以不执行）
            self.emb.weight.register_hook(lambda x: keep_partial_grad(x, self.opt['topn']))
        else:
            # 微调所有的词嵌入
            print("Finetune all embeddings.")

    def forward(self, sample):
        """
          words   : (bz, doc_len) int64
          ner     : (bz, doc_len) int64
          coref   : (bz, doc_len) int64
          length  : (bz) int64
          head_mask: (bz, doc_len) float32
          tail_mask: (bz, doc_len) float32

          sent_doc_mp: (bz, sent_num, doc_len) float32
          sent_num  : (bz, ) int64
          support_set: (bz, sent_num)  #float32
        """
        for k, v in sample.items():
            sample[k] = v.cuda()
        words, ner, coref, length, head_mask, tail_mask = \
            sample['words_id'], sample['ners_id'], sample['coref_id'], sample['doc_len'], \
            sample['head_mask'], sample['tail_mask']

        mention_all_mask, entity_all_mask, sentences_mask, M_M, M_E, M_S, E_S, S_S = sample['mention_all_mask'], \
                                                                                     sample['entity_all_mask'], sample[
                                                                                         "sentences_mask"], sample[
                                                                                         'M_M'], sample["M_E"], sample[
                                                                                         "M_S"], sample["E_S"], sample[
                                                                                         "S_S"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Doc Embedding
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['ner_dim'] > 0:
            embs += [self.ner_embed(ner)]
        if self.opt['coref_dim'] > 0:
            # 指向同一实体的指称对应词的coref_id相同，根据下标取到的指称层对应向量表示也相同。
            embs += [self.coref_embed(coref)]
        # 第3维(嵌入维度)进行拼接（三维：bz * doc_len * 嵌入维度）===> bz * doc_len * 140
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # 全局编码器
        doc_outputs, last_h = self.global_encoder(embs, length)

        # 进行了2层drop
        doc_outputs = self.rnn_drop(doc_outputs)  # (bz, doc_len, hs)
        doc_outputs = self.rnn_drop(doc_outputs)  # (bz, doc_len, hs)

        # ---------------异构图开始（这里并没有将每层的表征连接起来）
        # 补一下E_E
        E_E = []
        for E_id in range(len(sample['entity_all_mask'])):
            E_E.append([E_id, E_id])
        E_E = torch.tensor(E_E).to(device)
        graph_data = {
            # 其中M_M和S_S表示的是无向边（包含a到b和b到a的2种单向边）
            ('mention', 'M_M', 'mention'): (M_M.t().contiguous()[0], M_M.t().contiguous()[1]),
            ('mention', 'M_E', 'entity'): (M_E.t().contiguous()[0], M_E.t().contiguous()[1]),
            ('mention', 'M_S', 'sentence'): (M_S.t().contiguous()[0], M_S.t().contiguous()[1]),
            ('entity', 'E_S', 'sentence'): (E_S.t().contiguous()[0], E_S.t().contiguous()[1]),
            ('entity', 'E_M', 'mention'): (M_E.t().contiguous()[1], M_E.t().contiguous()[0]),
            ('sentence', 'S_M', 'mention'): (M_S.t().contiguous()[1], M_S.t().contiguous()[0]),
            ('sentence', 'S_E', 'entity'): (E_S.t().contiguous()[1], E_S.t().contiguous()[0]),
            ('sentence', 'S_S', 'sentence'): (S_S.t().contiguous()[0], S_S.t().contiguous()[1]),
            ('entity', 'E_E', 'entity'): (E_E.t().contiguous()[0], E_E.t().contiguous()[1])
        }
        gg = dgl.heterograph(graph_data)
        # g给图节点赋值。
        gg.nodes['mention'].data['feature'] = torch.matmul(sample['mention_all_mask'], doc_outputs.squeeze(0)).squeeze(1)
        gg.nodes['entity'].data['feature'] = torch.matmul(sample['entity_all_mask'], doc_outputs.squeeze(0)).squeeze(1)
        gg.nodes['sentence'].data['feature'] = torch.matmul(sample['sentences_mask'], doc_outputs.squeeze(0)).squeeze(1)
        # 增加自反边
        g = dgl.add_self_loop(gg, etype='M_M')
        g = dgl.add_self_loop(g, etype='S_S')
        # outOfGAT = self.GAT(g, {'mention': g.nodes['mention'].data['feature'],
        #                         'entity': g.nodes['entity'].data['feature'],
        #                         'sentence': g.nodes['sentence'].data['feature']})

        outOfFirstGNN = self.firstGNN(g, {'mention': g.nodes['mention'].data['feature'],
                                          'entity': g.nodes['entity'].data['feature'],
                                          'sentence': g.nodes['sentence'].data['feature']})
        # ---------------异构图结束（这里并没有将每层的表征连接起来）

        all_sample_head = []
        all_sample_tail = []
        all_sample_pre_evidence = []
        # 到时候计算权重需要
        all_top_similarity = []
        # ------- 得到所有样本的证据句（这里只用相似度计算即可） 首先需要得到样本（实体对）的表征，然后再计算相似度，得到每一个样本与之相关的证据句。
        all_sample_sentences = []
        all_sample_edge_weight = []
        all_sample_head_tail = []
        '''
            得到证据句
        '''
        # for head_tail in sample['all_sample']:
        #     head = outOfFirstGNN['entity'][[head_tail[0]]]
        #     tail = outOfFirstGNN['entity'][[head_tail[1]]]
        #     all_sample_head.append(head)
        #     all_sample_tail.append(tail)
            # all_sample_head_tail.append(torch.cat([head.unsqueeze(0), tail.unsqueeze(0)], dim=0))
        all_sentence = outOfFirstGNN['sentence']
        all_sample_head = torch.index_select(outOfFirstGNN['entity'], 0, sample['all_sample'].split(1, dim=1)[0].squeeze(1))
        all_sample_tail = torch.index_select(outOfFirstGNN['entity'], 0, sample['all_sample'].split(1, dim=1)[1].squeeze(1))

        # 按相似度计算相似度最高的句子。
        # evidence_vec, evidence_id = get_evidence(torch.stack(all_sample_head_tail), all_sentence, 3)
        # torch.index_select(all_sentence, 0, g.edges(etype='E_S')[1])
        evidence_id = sample['support_set']
        '''
            计算每条边的权重
        '''

        # print(1)
        edge_weight = {}
        all_sample_len = len(sample['label'])
        for edge_type in g.etypes:
            if edge_type in ['M_E', 'E_M']:
                edge_weight[edge_type] = torch.zeros(len(g.edges(etype=edge_type)[0]), all_sample_len, 1).to(device)
            if edge_type in ['M_M', 'E_E']:
                # 自反边设置为1，其他设置为0
                edge_weight[edge_type] = torch.ones(len(g.edges(etype=edge_type)[0]), all_sample_len, 1).to(device)
                edge_weight[edge_type][:len(gg.edges(etype=edge_type)[0]),:,:] = 0
            # 'E_S', 'S_E'
            # if 'E_S' not in edge_weight:
            #     edge_weight['E_S'] = torch.ones(len(g.edges(etype='E_S')[0]), all_sample_len, 1).to(device)
            #     for i in range(len(g.edges(etype='E_S')[0])):
            #         for j, evi_id in enumerate(evidence_id):
            #             # 下面3行耗时占据运行一个文档的90%时长。
            #             num_e = g.edges(etype='E_S')[0][i]
            #             num_s = g.edges(etype='E_S')[1][i]
            #             if num_s in evi_id and num_e in sample['all_sample'][j]:
            #                 edge_weight['E_S'][i][j][0] = 1 + torch.abs(torch.cosine_similarity(
            #                     g.nodes['entity'].data['feature'][num_e].unsqueeze(0),
            #                     g.nodes['sentence'].data['feature'][num_s].unsqueeze(0)))
            #     edge_weight['S_E'] = edge_weight['E_S']
        # 计算E_S/S_E的权重
        len_E_S = len(g.edges(etype='E_S')[0])
        # （边长度，样本数，2）
        e1 = g.edges(etype='E_S')[0].unsqueeze(1).unsqueeze(2).repeat(1, all_sample_len, 2)
        # （边长度，样本数，3）, （边长度，样本数，5）
        s1 =g.edges(etype='E_S')[1].unsqueeze(1).unsqueeze(2).repeat(1, all_sample_len, 3)
        # （边长度，样本数，2）
        e2 = sample['all_sample'].unsqueeze(0).repeat(len_E_S, 1, 1)
        # （边长度，样本数，3）  # （边长度，样本数，5）
        s2 = evidence_id.unsqueeze(0).repeat(len_E_S, 1, 1)
        eq_e = torch.eq(e1, e2)
        eq_e = torch.sum(eq_e, dim=2).unsqueeze(2)
        eq_s = torch.eq(s1, s2)
        eq_s = torch.sum(eq_s, dim=2).unsqueeze(2)
        flag = eq_e + eq_s
        # 换顺序会出问题
        flag[flag < 2] = 0
        flag[flag == 2] = 1

        # 得到每条边的entity表征
        # (边长度，样本数，表征维度)
        E = torch.index_select(outOfFirstGNN['entity'], 0, g.edges(etype='E_S')[0]).unsqueeze(1).repeat(1, all_sample_len, 1)
        S = torch.index_select(outOfFirstGNN['sentence'], 0, g.edges(etype='E_S')[1]).unsqueeze(1).repeat(1, all_sample_len, 1)
        sim = torch.abs(torch.cosine_similarity(E, S, dim=2)).unsqueeze(2)+1
        res = sim * flag
        res[res == 0] = 0.5
        edge_weight['E_S'] = res
        edge_weight['S_E'] = res

        # 计算M_S/S_M的权重
        len_M_S = len(g.edges(etype='M_S')[0])
        # （边长度，样本数，2）
        e1 = g.edges(etype='M_S')[0].unsqueeze(1).unsqueeze(2).repeat(1, all_sample_len, 2)
        # （边长度，样本数，3）
        s1 =g.edges(etype='M_S')[1].unsqueeze(1).unsqueeze(2).repeat(1, all_sample_len, 3)
        # （边长度，样本数，2）
        e2 = sample['all_sample'].unsqueeze(0).repeat(len_M_S, 1, 1)
        # （边长度，样本数，3）
        s2 = evidence_id.unsqueeze(0).repeat(len_M_S, 1, 1)
        eq_e = torch.eq(e1, e2)
        eq_e = torch.sum(eq_e, dim=2).unsqueeze(2)
        eq_s = torch.eq(s1, s2)
        eq_s = torch.sum(eq_s, dim=2).unsqueeze(2)
        flag = eq_e + eq_s
        # 换顺序会出问题
        flag[flag < 2] = 0
        flag[flag == 2] = 1

        # 得到每条边的entity表征
        # (边长度，样本数，表征维度)
        M = torch.index_select(g.nodes['mention'].data['feature'], 0, g.edges(etype='M_S')[0]).unsqueeze(1).repeat(1, all_sample_len, 1)
        S = torch.index_select(g.nodes['sentence'].data['feature'], 0, g.edges(etype='M_S')[1]).unsqueeze(1).repeat(1, all_sample_len, 1)
        sim = torch.abs(torch.cosine_similarity(M, S, dim=2)).unsqueeze(2)+1
        res = sim * flag
        res[res == 0] = 0.5
        edge_weight['M_S'] = res
        edge_weight['S_M'] = res

        # 计算S_S的权重
        len_M_S = len(g.edges(etype='S_S')[0])
        # （边长度，样本数，2）
        e1 = g.edges(etype='S_S')[0].unsqueeze(1).unsqueeze(2).repeat(1, all_sample_len, 2)
        # （边长度，样本数，3）
        s1 =g.edges(etype='S_S')[1].unsqueeze(1).unsqueeze(2).repeat(1, all_sample_len, 3)
        # （边长度，样本数，2）
        e2 = sample['all_sample'].unsqueeze(0).repeat(len_M_S, 1, 1)
        # （边长度，样本数，3）
        s2 = evidence_id.unsqueeze(0).repeat(len_M_S, 1, 1)
        eq_e = torch.eq(e1, e2)
        eq_e = torch.sum(eq_e, dim=2).unsqueeze(2)
        eq_s = torch.eq(s1, s2)
        eq_s = torch.sum(eq_s, dim=2).unsqueeze(2)
        flag = eq_e + eq_s
        # 换顺序会出问题
        flag[flag < 2] = 0
        flag[flag == 2] = 1

        # 得到每条边的entity表征
        # (边长度，样本数，表征维度)
        S1 = torch.index_select(g.nodes['sentence'].data['feature'], 0, g.edges(etype='S_S')[0]).unsqueeze(1).repeat(1, all_sample_len, 1)
        S2 = torch.index_select(g.nodes['sentence'].data['feature'], 0, g.edges(etype='S_S')[1]).unsqueeze(1).repeat(1, all_sample_len, 1)
        sim = torch.abs(torch.cosine_similarity(S1, S2, dim=2)).unsqueeze(2)+1
        res = sim * flag
        res[res == 0] = 0.5
        edge_weight['S_S'] = res

        # 设置S_S自反边为1
        edge_weight['S_S'][:len(gg.edges(etype='S_S')[0]), :, :] = 1

        '''
            全图特征叠加GCN
        '''
        g.edata['edge_weight'] = edge_weight
        mention_feature = g.nodes['mention'].data['feature'].unsqueeze(1).repeat(1, all_sample_len, 1)
        entity_feature = g.nodes['entity'].data['feature'].unsqueeze(1).repeat(1, all_sample_len, 1)
        sentence_feature = g.nodes['sentence'].data['feature'].unsqueeze(1).repeat(1, all_sample_len, 1)
        secondGNN = self.secondGNN(g, {'mention': mention_feature,
                                       'entity': entity_feature,
                                       'sentence': sentence_feature},
                                   g.edata['edge_weight'])

        # ------- 为每一个样本进行分类（分类器）。
        # 1.得到全局表征，第一次GCN的结果，第二次GCN的结果
        global_head = torch.matmul(head_mask, doc_outputs.squeeze(0))
        global_tail = torch.matmul(tail_mask, doc_outputs.squeeze(0))
        local_head1 = all_sample_head
        local_tail1 = all_sample_tail
        local_head2 = []
        local_tail2 = []
        # ins_path_num * 256
        local_head, local_tail = self.local_encoder(sample, doc_outputs, embs, head_mask, tail_mask)

        for i, head_tail in enumerate(sample['all_sample']):
            head = head_tail[0]
            tail = head_tail[1]
            local_head2.append(secondGNN['entity'][head][i])
            local_tail2.append(secondGNN['entity'][tail][i])
        all_sample_head = torch.index_select(outOfFirstGNN['entity'], 0, sample['all_sample'].split(1, dim=1)[0].squeeze(1))

        pred = self.entity_classifier(global_head, global_tail,
                                      local_head, local_tail,
                                      local_head1, local_tail1,
                                      torch.stack(local_head2), torch.stack(local_tail2),
                                      sample['path2ins']
                                      )
        # Document Representation
        # Local Encoder
        # local_head, local_tail = self.local_encoder(sample, doc_outputs, embs, head_mask, tail_mask)

        # Classifier
        # pred = self.entity_classifier(global_head, global_tail, local_head, local_tail, sample['path2ins'])

        return pred

