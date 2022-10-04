import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityClassifier(nn.Module):
    # EntityClassifier(opt['hidden_dim'], opt['num_class'], opt['mlp_dropout']),隐含层维度，关系种类，mlp的dropout
    def __init__(self, hs, num_class, mlp_drop):
        super().__init__()
        # 隐含层*2。设置的隐含层是256
        indim = 2 * hs
        
        self.classifier = MLP(indim, hs, num_class, mlp_drop)

    def forward(self, global_head, global_tail, local_head, local_tail, path2ins):
        # 对于基于path的模型来说，我们要把local 和 global的entity rep统一在一起，需要完成ins2path或者path2ins的映射
        ins2path = torch.transpose(path2ins, 0, 1)  # (ins_num, path_num)

        global_head = torch.matmul(path2ins, global_head)
        global_tail = torch.matmul(path2ins, global_tail)

        # Construct entity representation
        head_rep, tail_rep = [], []
        head_rep.append(local_head)
        tail_rep.append(local_tail)
        head_rep.append(global_head)
        tail_rep.append(global_tail)

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
    # MLP(2*hs, hs, num_class, mlp_drop)
    def __init__(self, indim, hs, outdim, mlp_drop):
        super().__init__()
        '''
        eh, et, |eh-et|, eh*et
        '''
        # 8*hs
        indim = 4 * indim
        # 设置线性层，输入的维度是 8*hs ，输出的维度是 2*hs
        self.linear1 = nn.Linear(indim, 2 * hs)
        # 设置线性层,输入的维度是 2*hs ，输出的维度是 96（种关系）
        self.linear2 = nn.Linear(2 * hs, outdim)
        self.drop = nn.Dropout(mlp_drop)
    
    def forward(self, head_rep, tail_rep):
        """
        :param head_rep: (?, hs)
        :param tail_rep: (?, hs)
        :param doc_rep: (1, hs)
        :return: logits (?, outdim)
        """
        # Construct input of MLP
        mlp_input = [head_rep, tail_rep, torch.abs(head_rep - tail_rep), head_rep * tail_rep] 
        mlp_input = torch.cat(mlp_input, -1)  # (bz, ?)
        
        h = self.drop(F.relu(self.linear1(mlp_input)))
        return self.linear2(h)

