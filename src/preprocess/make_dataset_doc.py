from collections import defaultdict
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

def make_pre_dataset(data_path, data_type, ner2id, word2id, label2id,
                     keep_sent_order=False, lower=False):
    with open(data_path, 'r') as fh:
        dataset = json.load(fh)
    
    documents = []
    # 使用枚举的方法，可以循环遍历index和value
    for doc_id, data in tqdm(enumerate(dataset)):
        sents = data["sents"]
        
        data_ner = [["None" for _ in sent] for sent in sents]
        data_coref = [[0 for _ in sent] for sent in sents]
        for ns_id, ns in enumerate(data['vertexSet']):
            for node in ns:
                sent_id = node['sent_id']
                for pos in range(node['pos'][0], min(node['pos'][1], len(data_ner[sent_id]))):
                    data_ner[sent_id][pos] = node['type']
                    data_coref[sent_id][pos] = ns_id
        
        doc_data = {'doc_id': doc_id, 'doc_title': data['title']}
        doc_data['words_id'] = []
        doc_data['sent_words_id'] = []
        for sent, ner in zip(sents, data_ner):
            word_id = []
            for w, e in zip(sent, ner):
                if lower:
                    w = w.lower()
                if w in word2id:
                    word_id.append(word2id[w])
                else:
                    word_id.append(word2id['UNK'])
            doc_data['words_id'] += word_id
            doc_data['sent_words_id'].append(word_id)
        
        doc_sent_offset = []
        doc_len = len(doc_data['words_id'])
        sent_doc_mp = np.zeros([len(sents), doc_len], dtype=np.float32)
        offset = 0
        for sent_id, sent in enumerate(sents):
            doc_sent_offset.append(offset)
            sent_doc_mp[sent_id][offset:offset + len(sent)] = 1.
            offset += len(sent)
        doc_sent_offset.append(offset)
        doc_data['sent_num'] = len(doc_sent_offset) - 1
        doc_data['sent_doc_mp'] = sent_doc_mp.tolist()
        doc_data['ners_id'] = [ner2id[e] if e in ner2id else ner2id['BLANK'] for ner in data_ner for e in ner]
        doc_data['sent_ners_id'] = [[ner2id[e] if e in ner2id else ner2id['BLANK'] for e in ner] for ner in data_ner]
        doc_data['sent_coref_id'] = data_coref
        doc_data['coref_id'] = [t for tt in data_coref for t in tt]
        
        doc_data['doc_len'] = len(doc_data['words_id'])
        
        sent2doc = []
        offset = 0
        for sent_id, sent in enumerate(sents):
            sent2doc.append(list(range(offset, offset + len(sent))))
            offset += len(sent)
        doc_data['sent2doc'] = sent2doc
        
        doc_sent_bound = []
        for i in range(len(doc_sent_offset) - 1):
            doc_sent_bound.append((doc_sent_offset[i], doc_sent_offset[i + 1]))
        doc_data['doc_sent_bound'] = doc_sent_bound

        doc_data['pos_ins'] = []
        doc_data['neg_ins'] = []

        # 保存了头实体与多个尾实体之间的关系，如  ht2label[0] 的结果是{1: [23], 2: [22]}
        # 表示编号为0的头实体，和以编号为1的尾实体关系类型是23。
        # 表示编号为0的头实体，和以编号为2的尾实体关系类型是22。
        ht2label = defaultdict(dict)
        if data_type != 'test':
            for r in data['labels']:
                if r['t'] not in ht2label[r['h']]:
                    ht2label[r['h']][r['t']] = []
                ht2label[r['h']][r['t']].append(label2id[r['r']])
        # 记录不同的头尾实体对（包含头尾互换的情况），不一定有关系。
        h_t_pair = []
        for h in range(len(data['vertexSet'])):
            for t in range(len(data['vertexSet'])):
                if h == t:
                    continue
                h_t_pair.append((h, t))

        # 记录不同的头尾实体对（不包含头尾互换的情况），不一定有关系。
        # h_t_pair = []
        # for h in range(len(data['vertexSet'])):
        #     for t in range(h, len(data['vertexSet'])):
        #         if h == t:
        #             continue
        #         h_t_pair.append((h, t))

        for h, t in h_t_pair:
            # if t not in merge[h]:
            #     continue
            #  记录了头尾实体的路径。（每条路径就是句子id构成的），这个路径就是支撑句。
            # sent_ids_list = merge[h][t]

            ins_data = {'doc_id': doc_id, 'doc_title': data['title'], 'head': h, 'tail': t,
                        'head_ner': ner2id[data['vertexSet'][h][0]['type']], 'tail_ner': ner2id[data['vertexSet'][t][0]['type']],
                        'label': [], 'head_mask': [], 'tail_mask': []}

            if t in ht2label[h]:
                ins_data['label'] = ht2label[h][t]
            
            head_entity_num, tail_entity_num = 0, 0
            head_mask = {}
            tail_mask = {}
            for node in data['vertexSet'][h]:
                # 对应句子的起始位置，如第一个句子开始位置是0，第二个句子开始位置是40
                offset = doc_sent_offset[node['sent_id']]
                head_entity_num += 1
                for pos in range(node['pos'][0], node['pos'][1]):
                    head_mask[pos + offset] = 1 / (node['pos'][1] - node['pos'][0])
            for node in data['vertexSet'][t]:
                offset = doc_sent_offset[node['sent_id']]
                tail_entity_num += 1
                for pos in range(node['pos'][0], node['pos'][1]):
                    tail_mask[pos + offset] = 1 / (node['pos'][1] - node['pos'][0])
            for k, v in head_mask.items():
                head_mask[k] = v / head_entity_num
            for k, v in tail_mask.items():
                tail_mask[k] = v / tail_entity_num
            
            ins_data['head_mask'] = head_mask
            ins_data['tail_mask'] = tail_mask

            
            if len(ins_data['label']) > 0:
                doc_data['pos_ins'].append(ins_data)
            else:
                # 保存所有没有关系的实体对到neg_ins,尽管没有指定的关系，但是却可能存在路径信息。
                doc_data['neg_ins'].append(ins_data)

        #   添加异构图节点过度信息（过度信息*全文表征=对应节点的表征）和边的关系
        # print(dataset[doc_id])
        # 模拟全文的词向量
        # documentsToVec = torch.randn(doc_data['doc_len'], 4)
        # print(documentsToVec)
        res = []
        mention_all = []
        entity_all = []
        sent_len = len(doc_data['doc_sent_bound'])
        for index, entity in enumerate(dataset[doc_id]['vertexSet']):
            temp = []
            entity_temp = [0] * doc_data['doc_len']
            for mention_index, mention in enumerate(entity):
                value = 1 / (mention['pos'][1] - mention['pos'][0])
                pos = doc_data['doc_sent_bound'][mention['sent_id']][0]
                list1 = [0] * (pos + mention['pos'][0]) + (mention['pos'][1] - mention['pos'][0]) * [value] + (
                        doc_data['doc_len'] - (mention['pos'][1] - mention['pos'][0]) - (pos + mention['pos'][0])) * [
                            0]
                temp.append(list1)
                mention_all.append(list1)
                k = 1 / len(entity)
                for i in range(0, len(list1)):
                    entity_temp[i] = k * list1[i] + entity_temp[i]
            entity_all.append(entity_temp)
            res.append(temp)
        # 异构图节点的过渡表示
        doc_data['mention_entity_mask'] = res
        doc_data['mention_all_mask'] = mention_all
        doc_data['entity_all_mask'] = entity_all
        res = []
        for sentences in doc_data['doc_sent_bound']:
            temp = []
            value = 1 / (sentences[1] - sentences[0])
            temp.append([0] * sentences[0] + (sentences[1] - sentences[0]) * [value] + (
                    doc_data['doc_len'] - sentences[1]) * [0])
            res.append(temp)
        doc_data['sentences_mask'] = res
        # 获取异构图的边M-M(包含自反边)===等同于无向图
        # M_M = []
        # for around in doc_data['doc_sent_bound']:
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
        for i, entity in enumerate(doc_data['mention_entity_mask']):
            for mention in entity:
                M_E.append([index, i])
                index = index + 1
        M_S = []
        E_S = []
        index = 0
        for i, entity in enumerate(dataset[doc_id]['vertexSet']):
            for mention in entity:
                M_S.append([index, mention['sent_id']])
                index = index + 1
                if [i, mention['sent_id']] not in E_S:
                    E_S.append([i, mention['sent_id']])
        S_S = []
        for i in range(0, sent_len):
            if i != (sent_len - 1):
                S_S.append([i, i + 1])
                S_S.append([i+1, i])
            else:
                S_S.append([sent_len - 1, 0])
                S_S.append([0, sent_len - 1])
        M_M = []
        for i in range(0, sent_len):
            for j, m_s_start in enumerate(M_S):
                if m_s_start[1] == i:
                    for k in range(j + 1, len(M_S)):
                        if M_S[k][1] == i:
                            M_M.append([j, k])
                            M_M.append([k, j])
        doc_data['M_M'] = M_M
        doc_data['M_E'] = M_E
        doc_data['M_S'] = M_S
        doc_data['E_S'] = E_S
        doc_data['S_S'] = S_S
        documents.append(doc_data)
    
    return documents


def load_docred_dict(dt_file_path):
    word2id = json.load(open(os.path.join(dt_file_path, "glove_100_lower_word2id.json"), 'r'))
    ner2id = json.load(open(os.path.join(dt_file_path, "ner2id.json"), 'r'))
    label2id = json.load(open(os.path.join(dt_file_path, 'rel2id.json'), "r"))
    
    return word2id, ner2id, label2id


if __name__ == '__main__':
    
    pre_dir = '../../dataset/'
    post_dir = '../../dataset'
    word2id, ner2id, label2id = load_docred_dict(
        '../../dataset/DocRED_baseline_metadata')
        
    if os.path.exists("../../dataset/100_lower-doc") == False:
        os.makedirs("../../dataset/100_lower-doc")
    
    for ds in ['train', 'dev', 'test']:
        if ds == 'train':
            file_name = 'train_annotated'
        else:
            file_name = ds
        documents = make_pre_dataset(os.path.join(pre_dir, '{}.json'.format(file_name)),
                                     ds, ner2id, word2id, label2id, keep_sent_order=True,
                                     lower=True)
        with open(os.path.join(post_dir, '100_lower-doc/{}.json'.format(ds)), 'w') as fh:
            json.dump(documents, fh, indent=2)

    with open(os.path.join(post_dir, '100_lower-doc/small_train.json'.format(ds)), 'w') as fh:
        # 获取训练集中的5个
        json.dump(documents[:3], fh, indent=2)
