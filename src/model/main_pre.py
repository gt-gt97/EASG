import argparse
from torch_geometric.data import HeteroData

import json
import os

import numpy as np
import torch


# 加载设置的参数
def arg_parser():
    parser = argparse.ArgumentParser(description='')

    # # Random Seed(随机种子，用于初始化)，根据初始值生成伪随机数，便于之后的代码复现，以为如果每次随机数不一样，最后的结果有差异，不能复现。
    # # 设置为了31
    # parser.add_argument('--dataset_seed', type=int, help="Dataset random seed")
    # # 设置为了32
    # parser.add_argument('--model_seed', type=int, help="Model random seed")
    #
    # # Args for utils
    # # 模型训练或者验证
    # parser.add_argument('--mode', type=str, help="Train or eval")
    # # 不清楚(设置的doc_mode=2)
    # parser.add_argument('--doc_mode', type=int, help="Train or eval", default=1)
    # # 使用GPU（显卡）的ID
    # parser.add_argument('--gpu_no', type=str, help="GPU number")
    # # 数据集字典
    # parser.add_argument('--data_dir', type=str, help="Dataset directory")
    #
    # # 设置生成日志路径
    # parser.add_argument('--log_path', type=str, help="Path of log file")
    # # 设置模型路径
    # parser.add_argument('--model_path', type=str, help="Path to store model")
    # # 设置参数batch(梯度下降算法的超参数，一个周期内一次批量训练的样本数)
    # parser.add_argument('--batch_size', type=int, help="Batch size", default=1)
    # # 设置参数max_epoch(最大训练的次数)
    # parser.add_argument('--max_epoch', type=int, help="Maximum epoch to train")
    # # 评估（验证）开始前的步骤[不清楚],设置的0
    # parser.add_argument('--warmup_step', type=int, help="Steps before evaluation begins")
    # # 每个步骤的评估，如果是-1就是每一次epoch后再评估。设置的1
    # parser.add_argument('--eval_step', type=int, help="Evaluation per step. If -1, eval per epoch")
    # # 是否保存模型
    # parser.add_argument('--save_model', type=int, help="1 to save model, 0 not to save")
    #
    # # Args for dataset,模型的随机种子，最大实体对数量？？？
    # parser.add_argument('--max_entity_pair_num', type=int, help="Model random seed", default=1800)
    #
    # # Embds（词嵌入）
    # # 微调n个词的词向量。
    # parser.add_argument('--topn', type=int, help="finetune top n word vec")
    # # 隐含层层数
    # parser.add_argument('--hidden_size', type=int, help="Size of hidden layer")
    # # 预训练矩阵路径（glove）
    # parser.add_argument('--pre_trained_embed', type=str, help="Pre-trained embedding matrix path")
    # # 实体种类（7种）
    # parser.add_argument('--ner_num', type=int, help="Ner Num", default=7)
    # # 实体词向量的维度
    # parser.add_argument('--ner_dim', type=int, help="Ner embed size")
    # # 共指嵌入的维度
    # parser.add_argument('--coref_dim', type=int, help="Coref embedd size")
    # # 位置嵌入的维度
    # parser.add_argument('--pos_dim', type=int, help="relative position embed size", default=0)
    #
    # # Optimizer (Only useful for Bert)，对于bert的优化
    # # 使用自适应学习率。
    # parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
    # # L2正则化（权重衰减）
    # parser.add_argument('--weight_decay', type=float, default=0, help='.')
    # # Gradient Clipping(梯度裁剪)去解决梯度爆炸问题。
    # parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    #
    # # Dropout
    # # Dropout可以作为训练深度神经网络的一种trick供选择。在每个训练批次中，通过忽略一半【可以设置】的特征检测器（让一半的隐层节点值为0），可以明显地减少过拟合现象。
    # # 设置输入层的Dropout
    # parser.add_argument('--input_dropout', type=float, help="")
    # # 设置RNN的Dropout
    # parser.add_argument('--rnn_dropout', type=float, help="")
    # # 设置多层感知机的Dropout
    # parser.add_argument('--mlp_dropout', type=float, help="")
    #
    # # Classifier
    # # 是否使用全局表征（文档嵌入）使用None
    # parser.add_argument('--doc_embed_required', type=int, help="Whether use global entity rep")
    # # 是否使用局部表征（path_bilstm）
    # parser.add_argument('--local_required', type=int, help="Whether use local entity rep")
    # # 设置最大序列长度（默认512）
    # parser.add_argument('--max_sent_len', type=int, help="The maximum sequence length", default=512)
    # # 设置分类器标签种类（也就是关系的种类）
    # parser.add_argument('--class_num', type=int, help="Number of classification labels", default=97)
    #
    args = parser.parse_args()
    return args

# # 得到指称节点表征(词嵌入+类型嵌入)
# def get_mention_vector():
#     # print(document)
#     # 加载训练集
#     with open("../dataset/DocRED_baseline_metadata/glove_100_lower_word2id.json", 'r') as fh:
#         word2id = json.load(fh)
#     all_mention_word_emb = []
#     all_mention_type_emb = []
#
#     # print(word2id[document["sents"][0][0].lower()])
#     for entity in document["vertexSet"]:
#         for mention in entity:
#             sent_id = mention["sent_id"]
#             # 记录每个指称所包含的词
#             mention_nodes = []
#             for index in range(mention["pos"][0], mention["pos"][1]):
#                 if document["sents"][sent_id][index].lower() in word2id:
#                     mention_nodes.append(pre_trained_embed[word2id[document["sents"][sent_id][index].lower()]])
#                 else:
#                     # 词表中没有这个词
#                     print("词表中没有指称中所包含的字：{}".format(document["sents"][sent_id][index].lower()))
#             # 指称节点的所有词的词向量取均值，并把这个指称节点加入到all_mention_word_emb中。
#             all_mention_word_emb.append(np.array(mention_nodes).mean(axis=0))
#         # 记录每个指称的类型
#         all_mention_type_emb.append("<"+entity[0]["type"]+">")
#
#     print(all_mention_type_emb)

if __name__ == '__main__':
    # 加载模型参数
    args = arg_parser()

    # 加载预训练词嵌入模型glove的矩阵结果
    pre_trained_embed = np.load("../dataset/DocRED_baseline_metadata/glove_100_lower_vec.npy")
    print(len(pre_trained_embed))

    # 加载训练集
    # with open("../dataset/train_annotated.json", 'r') as fh:
    #     documents = json.load(fh)
    #
    # # 遍历训练集
    # for key, document in enumerate(documents):
    #     # 测试阶段 只训练前5个文档
    #     if key >= 5:
    #         break
    #     else:
            # 每个文档需要构建一个异构图
            # data = HeteroData()
            # 获取实体节点表示
            # get_mention_vector()
            # 获取句子节点表示

            # 设置异构图节点，维度是[num_entity, num_features_paper]
            # data["mention"].x = torch.tensor()  # [num_papers, num_features_paper]
            # data["entity"].x = torch.tensor()  # [num_papers, num_features_paper]
            # data["sentences"].x = ...  # [num_authors, num_features_author]
            #
            # # 设置异构图的边，维度是[2, num_edges_cites]
            # data['entity', 'cites', 'entity'].edge_index = ...
            # data['entity', 'cites', 'entity'].edge_index = ...
            # data['entity', 'writes', 'sentences'].edge_index = ...
            # data['entity', 'writes', 'sentences'].edge_index = ...


            # for label in document["labels"]:
            #     print(label)
