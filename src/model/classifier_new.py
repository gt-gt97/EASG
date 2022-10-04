import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityClassifier(nn.Module):
    # EntityClassifier(opt['hidden_dim'], opt['num_class'], opt['mlp_dropout']),隐含层维度，关系种类，mlp的dropout
    def __init__(self, hs, num_class, mlp_drop):
        super().__init__()
        # 隐含层*2。设置的隐含层是256
        indim = 3 * hs
        
        self.classifier = MLP(indim, hs, num_class, mlp_drop)

    def forward(self, global_head, global_tail, local_head, local_tail, local_head1, local_tail1,local_head2, local_tail2, path2ins):
        # 对于基于path的模型来说，我们要把local 和 global的entity rep统一在一起，需要完成ins2path或者path2ins的映射
        ins2path = torch.transpose(path2ins, 0, 1)  # (ins_num, path_num)

        global_head = torch.matmul(path2ins, global_head)
        global_tail = torch.matmul(path2ins, global_tail)
        local_head1 = torch.matmul(path2ins, local_head1)
        local_tail1 = torch.matmul(path2ins, local_tail1)
        local_head2 = torch.matmul(path2ins, local_head2)
        local_tail2 = torch.matmul(path2ins, local_tail2)
        # Construct entity representation
        head_rep, tail_rep = [], []
        # head_rep.append(global_head)
        # tail_rep.append(global_tail)
        head_rep.append(local_head)
        tail_rep.append(local_tail)
        head_rep.append(local_head1)
        tail_rep.append(local_tail1)

        # head_rep.append(local_head1)
        # tail_rep.append(local_tail1)
        # head_rep.append(local_head1)
        # tail_rep.append(local_tail1)
        head_rep.append(local_head2)
        tail_rep.append(local_tail2)

        # head_rep.append(gf)
        # tail_rep.append(gl)

        head_rep = torch.cat(head_rep, dim=-1)
        tail_rep = torch.cat(tail_rep, dim=-1)

        pred = self.classifier(head_rep, tail_rep)
        
        pred = pred.squeeze(-1)
        pred = torch.sigmoid(pred)
        
        # 需要把path prob转换成ins prob
        pred = pred.unsqueeze(0)  # (1, path_num, num_class)
        ins2path = ins2path.unsqueeze(-1)  # (ins_num, path_num, num_class)
        pred = torch.max(pred * ins2path, dim=1)[0]

        return pred
        

class MLP(nn.Module):
    # MLP(3*hs, hs, num_class, mlp_drop)
    def __init__(self, indim, hs, outdim, mlp_drop):
        super().__init__()
        '''
        eh, et, |eh-et|, eh*et
        '''
        # 8*hs
        indim = 2 * indim
        # 设置线性层，输入的维度是 8*hs ，输出的维度是 2*hs，为什么不设置为3*hs,更多的信息进行预测。
        self.linear1 = nn.Linear(indim, 2 * hs)
        # self.linear1 = nn.Linear(indim, 2 * hs)
        # 设置线性层,输入的维度是 2*hs ，输出的维度是 96（种关系）
        self.linear2 = nn.Linear(2 * hs, outdim)
        self.drop = nn.Dropout(mlp_drop)
        self.prelu = torch.nn.PReLU()
    
    def forward(self, head_rep, tail_rep):
        """
        :param head_rep: (?, hs)
        :param tail_rep: (?, hs)
        :param doc_rep: (1, hs)
        :return: logits (?, outdim)
        """
        # Construct input of MLP
        # mlp_input = [head_rep, tail_rep, torch.abs(head_rep - tail_rep), head_rep * tail_rep]
        mlp_input = [head_rep, tail_rep]
        mlp_input = torch.cat(mlp_input, -1)  # (bz, ?)
        # h = self.drop(F.leaky_relu(self.linear1(mlp_input)))
        # h = self.drop(F.relu(self.linear1(mlp_input)))
        # 在学习权重时，不应该使用权重衰减
        # h = self.drop(F.prelu(self.linear1(mlp_input), torch.tensor([0.25]).cuda()))
        h = self.drop(self.prelu(self.linear1(mlp_input)))
        return self.linear2(h)
