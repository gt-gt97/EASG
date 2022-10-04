import dgl
import numpy as np
import torch
import torch as th
import heapq
# 构建异构图（）邻接表
from dgl.nn.pytorch import nn, HeteroGraphConv, GraphConv
import torch.nn.functional as F

# coding:utf-8
import random


if __name__ == '__main__':
    insert_sql = "INSERT INTO t_student(id, subjects, score) VALUES "
    # with open("1.txt", "a") as fp:
    #         fp.write(insert_sql+"")
    for i in range(0, 10000):
        with open("1.txt", "a") as fp:
            fp.write(insert_sql + "")
        for j in range(1, 101):
            subjects = ['English', 'mathematics', 'Chinese', 'chemical', 'physical']
            a = "('{0}', '{1}', '{2}'),".format(str(100 * i + j), random.choice(subjects), random.randint(0, 100))
            with open("1.txt", "a") as fp:
                fp.write(a+"")