import numpy as np
import json
import random
import math
from collections import defaultdict
import copy
import torch
import os


class Dataset(object):
    # Dataset(os.path.join(args.data_dir, 'train'), 97, True, max_entity_pair_num=args.max_entity_pair_num, args=args)
    def __init__(self, data_file_path, label_num, shuffle, max_sent_len=512,
                 max_entity_pair_num=180000, mode="train", args=None):
        """
        :param data_type: if train, random dataset
        :param data_file_path: data dict
        :param batch_size:
        :param label_num:
        :param max_sent_len: 512 default
        :param negative_throw_prob: As negative instance are much more than positive instance, we don't use all negative
                instance during training.
        :param shuffle_level: when shuffle dataset, whether to keep all instance in the same document adjacent
        """
        self.label_num = label_num
        self.max_sent_len = max_sent_len
        self.shuffle = shuffle
        self.args = args
        # 继续对数据集进行加工
        self.documents = self.build_dataset(data_file_path)

        self.max_entity_pair_num = max_entity_pair_num
        # 指针，表示指向的当前文档下标
        self.ptr = 0
        self.size = len(self.documents)
        self.mode = mode

    def build_dataset(self, data_file_path):
        # 1）在pos_ins或者neg_ins中将关系编号label替换为97列one-hot表示的矩阵（数组）。
        # 2）将head_mask和tail_mask转化为长度为文档长度，如果不是相关的实体指称，值就是0，否则就是对应得分,便于后续求指称（实体）表征
        # 如： [ 0,0,0.5,0.5....]，（1）
        def pre_ins(ins):
            for key in ['head_mask', 'tail_mask']:
                new_dt = {}
                for k, v in ins[key].items():
                    # 保存了同一实体所有指称的下标编号以及对应的得分（1个实体得分为1，对应的指称均分1，指称中的每个字再均分。）
                    new_dt[int(k)] = v
                ins[key] = new_dt
                arr = np.zeros([doc['doc_len']], dtype=np.float32)
                arr_k = list(new_dt.keys())
                arr_v = [new_dt[k] for k in arr_k]
                arr[arr_k] = arr_v
                ins[key] = arr.tolist()

            ins_label = [0] * 97
            for label in ins['label']:
                # 把对应维度的数组变为1。类似one-hot
                ins_label[label] = 1
            if len(ins['label']) == 0:
                # 96种关系，97列，把第一列为1的作为负样本的关系表示。
                # 关系的第一个特征是1表示不确定关系，负样本中就是label没有值。[1,0,0,0...0]
                ins_label[0] = 1
            ins['label'] = np.array(ins_label, dtype=np.float32)
            return ins

        with open(data_file_path + '.json', 'r') as fh:
            # 加载数据集
            documents = json.load(fh)
        # 一篇文档一篇文档进行操作。
        for doc in documents:
            # 加了doc_label，
            doc['doc_label'] = np.zeros([97], dtype=np.float32)
            # 把sent_doc_mp设置为数组
            doc['sent_doc_mp'] = np.array(doc['sent_doc_mp'], dtype=np.float32)
            # 在正样本上，同时得到下标和值（对应的文档中的标明的关系）
            for idx, ins in enumerate(doc['pos_ins']):
                # 第n篇文档的第idx个关系
                doc['pos_ins'][idx] = pre_ins(ins)
                # 记录一篇文档中的所有正样本关系。其中doc['doc_label'][0]=[0,0,0...一共97个0]
                doc['doc_label'] += doc['pos_ins'][idx]['label']
            for idx, ins in enumerate(doc['neg_ins']):
                # 负样本中label为空。
                doc['neg_ins'][idx] = pre_ins(ins)

        if self.shuffle:
            # 打乱文档顺序
            random.shuffle(documents)

        return documents

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return self
    # 当对Dataset进行遍历的时候，每次循环就是一个文档，就先执行__next__方法，再执行循环体。
    def __next__(self):
        global all_ins

        def padding(lst, padding_len):
            data_len = len(lst)
            if data_len > padding_len:
                # 超过了截断
                return np.array(lst[:padding_len])
            elif data_len < padding_len:
                # 没超过，填充0
                return np.array(lst + [0] * (padding_len - data_len))
            else:
                return np.array(lst)
        # 如果遍历文档结束。
        if self.ptr == self.size:
            # 重置指针
            self.ptr = 0
            if self.shuffle:
                random.shuffle(self.documents)
            raise StopIteration
        # 得到当前遍历的文档数据。
        doc = self.documents[self.ptr]
        self.ptr += 1
        # 为sample设置一些值
        sample = {'support_set_change': [], 'support_set': [], 'sent_doc_mp': np.array(0), 'doc_label': np.array(0)}
        for key in ['doc_len', 'words_id', 'ners_id', 'coref_id', 'sent_num']:
            # 转化为数组，就有维度了。
            sample[key] = np.array([doc[key]])  # (1, ?)
        for key in ['head_mask', 'tail_mask', 'label', 'local_len', 'local_head_mask', 'local_tail_mask',
                    'local_words_id', 'local_ners_id', 'local_coref_id', 'local_first_head', 'local_first_tail',
                    'all_sample']:
            sample[key] = []
        for key in ['mention_all_mask', "entity_all_mask",
                    "sentences_mask", "M_M", "M_E", "M_S", "E_S", "S_S"]:
            sample[key] = doc[key]
        ins_num = 0
        path_num = []
        same_set_keys = ['head_entity', 'tail_entity', 'head_ner', 'tail_ner', 'pair_ner']
        same_set = {}
        for key in same_set_keys:
            # 使用defaultdict(list)，当没有键的时候不报错，返回默认值，文档对象中没有的，使用这种方式添加键。
            same_set[key] = defaultdict(list)

        # 正负样本实体对的和超过了180000（可以自己设置），打乱负样本顺序。
        # 正负样本实体对的和超过了180000（可以自己设置），打乱负样本顺序。
        if len(doc['pos_ins']) + len(doc['neg_ins']) > self.max_entity_pair_num:
            random.shuffle(doc['neg_ins'])
        all_ins = doc['pos_ins'] + doc['neg_ins'][:self.max_entity_pair_num - len(doc['pos_ins'])]
        random.shuffle(all_ins)
        #   遍历正负样本，且保证正负样本之和<=实体对的最大值，且保证了多出的负样本会舍去，因此上一个语句才会打乱负样本的顺序。
        #   本项目中的bz指的是一篇文档中的样本数量（正负样本）
        for ins_idx, ins in enumerate(all_ins):
            for key in ['head_mask', 'tail_mask', 'label']:
                # 保留正负样本数据的head_mask,tail_mask,label
                # 此时的head_mask,tail_mask,label在初始化init方法的时候就已经转化为向量了。
                sample[key].append(ins[key])
            # 计算样本涉及的证据句的长度。负样本涉及的证据句在第一步的时候就已经提取了。
            sample['local_len'] += ins['local_len']
            sample['all_sample'].append([ins['head'], ins['tail']])
            sample['support_set'].append(ins['support_set_change'])
            # sample['support_set'].append(ins['support_set'])
            for key in ['local_head_mask', 'local_tail_mask']:
                arr = np.zeros([len(ins[key]['k']), doc['max_path_len']], dtype=np.float32)
                arr_k_0 = [idx for idx, mask in enumerate(ins[key]['k']) for m in mask]
                arr_k_1 = [m for idx, mask in enumerate(ins[key]['k']) for m in mask]
                arr_v = [v for vals in ins[key]['v'] for v in vals]
                arr[arr_k_0, arr_k_1] = arr_v
                sample[key] += arr.tolist()

            for key in ['local_first_head', 'local_first_tail']:
                # s, e表示起始位置
                sample[key] += [list(range(-s, 0)) + [0] * (e - s) + list(range(e, doc['max_path_len'])) for s, e in
                                ins[key]]

            for key in ['words_id', 'ners_id', 'coref_id']:
                # 循环调用函数，并赋值给列表。
                arr = [padding([t for sent_id in sent_ids for t in doc['sent_{}'.format(key)][sent_id]],
                               doc['max_path_len']) for sent_ids in ins['support_set']]
                # 进行局部信息赋值。
                sample['local_{}'.format(key)] += arr

            ins_num += 1
            path_num.append(len(ins['local_len']))

        for key in ['head_mask', 'tail_mask', 'local_head_mask', 'local_tail_mask']:
            # head_mask和tail_mask维度是 样本数 * 文档长度（不确定先后顺序）
            sample[key] = np.array(sample[key], dtype=np.float32)  # .astype(np.float32)
            # print("sample[{}]:{}".format(key, sample[key].shape))
        for key in ['local_words_id', 'local_ners_id', 'local_coref_id']:
            sample[key] = np.array(sample[key], dtype=np.int64)

        sample['local_len'] = np.array(sample['local_len'][:sum(path_num)], np.int64)
        sample['label'] = np.array(sample['label'][:ins_num])
        path2ins = np.zeros([sum(path_num), ins_num], np.float32)
        offset = 0
        for pidx, pn in enumerate(path_num):
            arr_k_0 = [offset + t for t in range(pn)]
            arr_k_1 = [pidx] * pn
            path2ins[arr_k_0, arr_k_1] = 1.
            offset += pn
        sample['path2ins'] = path2ins
        sample['doc_label'] = np.array([doc['doc_label']], dtype=np.float32)

        for fl in ['mention_all_mask', "entity_all_mask", "sentences_mask"]:
            sample[fl] = torch.tensor(sample[fl], dtype=torch.float)
        for i in ["M_M", "M_E", "M_S", "E_S", "S_S", 'all_sample', 'support_set', 'doc_label']:
            sample[i] = torch.tensor(sample[i], dtype=torch.long)

        for key in sample:
            if key not in ['mention_all_mask', "entity_all_mask", "sentences_mask", "M_M", "M_E", "M_S", "E_S", "S_S",
                           'doc_label', 'all_sample', 'support_set']:
                sample[key] = torch.tensor(sample[key])
        # 这个返回值就是for x in y遍历时候的x
        # for key in sample:
        #     sample[key] = torch.tensor(sample[key])
        return sample

