import heapq
from collections import defaultdict
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import Counter

def extract_path(data, keep_sent_order):
    sents = data["sents"]
    nodes = [[] for _ in range(len(data['sents']))]
    e2e_sent = defaultdict(dict)

    # create mention's list for each sentence
    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    for sent_id in range(len(sents)):
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop Path
    path_two = defaultdict(dict)
    entityNum = len(data['vertexSet'])
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue
                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue
                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        if keep_sent_order == True:
                            cand_sents.sort()
                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop Path
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue
                            if cand2[1] == n1:
                                continue
                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            if keep_sent_order:
                                cand_sents.sort()
                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive Path
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 2:
                        continue
                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    # Merge
    merge = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n2 in path_two[n1]:
                merge[n1][n2] = path_two[n1][n2]
            if n2 in path_three[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += path_three[n1][n2]
                else:
                    merge[n1][n2] = path_three[n1][n2]

            if n2 in consecutive[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += consecutive[n1][n2]
                else:
                    merge[n1][n2] = consecutive[n1][n2]

    # Default Path
    for h in range(len(data['vertexSet'])):
        for t in range(len(data['vertexSet'])):
            if h == t:
                continue
            if t in merge[h]:
                continue
            merge[h][t] = []
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    cand_sents = [n1['sent_id'], n2['sent_id']]
                    if keep_sent_order:
                        cand_sents.sort()
                    merge[h][t].append([cand_sents])

    # Remove redundency
    tp_set = set()
    for n1 in merge.keys():
        for n2 in merge[n1].keys():
            hash_set = set()
            new_list = []
            for t in merge[n1][n2]:
                if tuple(t[0]) not in hash_set:
                    hash_set.add(tuple(t[0]))
                    new_list.append(t[0])
            merge[n1][n2] = new_list

    return merge


def make_pre_dataset(data_path, data_type, ner2id, word2id, label2id,
                     keep_sent_order=False, lower=False):
    with open(data_path, 'r') as fh:
        dataset = json.load(fh)

    documents = []

    for doc_id, data in tqdm(enumerate(dataset)):
        sents = data["sents"]
        merge = extract_path(data, keep_sent_order)

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

        ht2label = defaultdict(dict)
        if data_type != 'test':
            for r in data['labels']:
                if r['t'] not in ht2label[r['h']]:
                    ht2label[r['h']][r['t']] = []
                ht2label[r['h']][r['t']].append(label2id[r['r']])

        h_t_pair = []
        for h in range(len(data['vertexSet'])):
            for t in range(len(data['vertexSet'])):
                if h == t:
                    continue
                h_t_pair.append((h, t))

        max_path_len = 0
        for h, t in h_t_pair:
            if t not in merge[h]:
                continue
            sent_ids_list = merge[h][t]

            ins_data = {'doc_id': doc_id, 'doc_title': data['title'], 'head': h, 'tail': t,
                        'head_ner': ner2id[data['vertexSet'][h][0]['type']],
                        'tail_ner': ner2id[data['vertexSet'][t][0]['type']],
                        'label': [], 'head_mask': [], 'tail_mask': [], 'local_head_mask': {'k': [], 'v': []},
                        'local_tail_mask': {'k': [], 'v': []}, 'support_set': sent_ids_list, 'local_len': [],
                        'local_first_head': [], 'local_first_tail': []}

            """
            """
            if t in ht2label[h]:
                ins_data['label'] = ht2label[h][t]

            head_entity_num, tail_entity_num = 0, 0
            head_mask = {}
            tail_mask = {}
            for node in data['vertexSet'][h]:
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

            for sent_ids in sent_ids_list:
                head_entity_num, tail_entity_num = 0, 0
                head_mask_k = []
                head_mask_v = []
                tail_mask_k = []
                tail_mask_v = []
                head_mask_pos_pair = []
                tail_mask_pos_pair = []

                offset = 0

                for sent_id in sent_ids:
                    for node in data['vertexSet'][h]:
                        if node['sent_id'] == sent_id:
                            head_entity_num += 1
                            head_mask_pos_pair.append((node['pos'][0] + offset, node['pos'][1] + offset))
                            for pos in range(node['pos'][0], node['pos'][1]):
                                head_mask_k.append(pos + offset)
                                head_mask_v.append(1 / (node['pos'][1] - node['pos'][0]))
                    for node in data['vertexSet'][t]:
                        if node['sent_id'] == sent_id:
                            tail_entity_num += 1
                            tail_mask_pos_pair.append((node['pos'][0] + offset, node['pos'][1] + offset))
                            for pos in range(node['pos'][0], node['pos'][1]):
                                tail_mask_k.append(pos + offset)
                                tail_mask_v.append(1 / (node['pos'][1] - node['pos'][0]))
                    offset += len(sents[sent_id])
                ins_data['local_len'].append(offset)
                max_path_len = max(max_path_len, offset)
                for v in head_mask_v:
                    v = v / head_entity_num
                for v in tail_mask_v:
                    v = v / tail_entity_num

                head_mask_pos_pair.sort(key=lambda x: x[0])
                tail_mask_pos_pair.sort(key=lambda x: x[0])

                ins_data['local_head_mask']['k'].append(head_mask_k)
                ins_data['local_head_mask']['v'].append(head_mask_v)
                ins_data['local_tail_mask']['k'].append(tail_mask_k)
                ins_data['local_tail_mask']['v'].append(tail_mask_v)
                ins_data['local_first_head'].append(head_mask_pos_pair[0])
                ins_data['local_first_tail'].append(tail_mask_pos_pair[0])

            if len(ins_data['label']) > 0:
                doc_data['pos_ins'].append(ins_data)
            else:
                doc_data['neg_ins'].append(ins_data)
        doc_data['max_path_len'] = max_path_len

        # 证据句进行改造
        for p in doc_data['pos_ins']:
            temp = p['support_set']
            temp_list=[0 for x in range(0,doc_data['sent_num'])]
            for i, evis_id in enumerate(temp):
                for evi_id in evis_id:
                    temp_list[evi_id] += 1/len(temp[i])
            if len(temp_list) >= 3:
                s = list(pd.Series(temp_list).sort_values(ascending=False).index[:3])
            else:
                s = list(pd.Series(temp_list).sort_values(ascending=False).index[:])
                while len(s) < 3:
                    s.append(s[len(s)-1])
            p['support_set_change'] = s

        for p in doc_data['neg_ins']:
            temp = p['support_set']
            temp_list=[0 for x in range(0,doc_data['sent_num'])]
            for i, evis_id in enumerate(temp):
                for evi_id in evis_id:
                    temp_list[evi_id] += 1/len(temp[i])
            if len(temp_list) >= 3:
                s = list(pd.Series(temp_list).sort_values(ascending=False).index[:3])
            else:
                s = list(pd.Series(temp_list).sort_values(ascending=False).index[:])
                while len(s) < 3:
                    s.append(s[len(s)-1])
            p['support_set_change'] = s

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
                        doc_data['doc_len'] - (mention['pos'][1] - mention['pos'][0]) - (
                            pos + mention['pos'][0])) * [0]
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
                S_S.append([i + 1, i])
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