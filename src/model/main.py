import argparse
import math
import torch
import os
import logging
import json
import numpy as np
import random
from docBilstm import DocClassifier
from evaluator import Evaluator
import line_profiler

def arg_parser():
    parser = argparse.ArgumentParser(description='')

    # Random Seed(随机种子，用于初始化)，根据初始值生成伪随机数，便于之后的代码复现，以为如果每次随机数不一样，最后的结果有差异，不能复现。
    # 数据集的随机种子设置为了31
    parser.add_argument('--dataset_seed', type=int, help="Dataset random seed", default=31)
    # 模型的随机种子设置为了32
    parser.add_argument('--model_seed', type=int, help="Model random seed", default=32)

    # Args for utils
    # 模型训练或者验证
    # 'train', 'eval_dev', 'eval_test'
    parser.add_argument('--mode', type=str, help="Train or eval", default="train")
    # 不清楚(设置的doc_mode=2)
    # parser.add_argument('--doc_mode', type=int, help="Train or eval", default=1)

    # 使用GPU（显卡）的ID
    parser.add_argument('--gpu_no', type=str, help="GPU number", default='0')
    # 数据集字典
    parser.add_argument('--data_dir', type=str, help="Dataset directory", default='../../dataset/100_lower-doc')

    # 设置生成日志路径
    parser.add_argument('--log_path', type=str, help="Path of log file", default='../../log/100_lower-doc/doc_0-pos_20'
                                                                                 '-seed_31_32')
    # 设置模型路径
    parser.add_argument('--model_path', type=str, help="Path to store model", default='../../model/100_lower-doc/doc_0'
                                                                                      '-pos_20-seed_31_32')
    # 设置参数batch(梯度下降算法的超参数，一个周期内一次批量训练的样本数)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=1)
    # 设置参数max_epoch(最大训练的次数)
    parser.add_argument('--max_epoch', type=int, help="Maximum epoch to train", default=1000)
    # 评估（验证）开始前的步骤[不清楚],设置的0
    parser.add_argument('--warmup_step', type=int, help="Steps before evaluation begins", default=0)
    # 每个步骤的评估，如果是-1就是每一次epoch后再评估。设置的1
    parser.add_argument('--eval_step', type=int, help="Evaluation per step. If -1, eval per epoch", default=1)
    # 是否保存模型
    parser.add_argument('--save_model', type=int, help="1 to save model, 0 not to save", default=1)

    # Args for dataset,模型的随机种子，最大实体对数量？？？
    parser.add_argument('--max_entity_pair_num', type=int, help="Model random seed", default=1800)

    # Embds（词嵌入）
    # 微调n个词的词向量。
    parser.add_argument('--topn', type=int, help="finetune top n word vec", default=10)
    # 隐含层中的神经单元个数
    parser.add_argument('--hidden_size', type=int, help="Size of hidden layer", default=256)
    # 预训练矩阵路径（glove）
    parser.add_argument('--pre_trained_embed', type=str, help="Pre-trained embedding matrix path",
                        default='../../dataset/DocRED_baseline_metadata/glove_100_lower_vec.npy')
    # 实体种类（7种）
    parser.add_argument('--ner_num', type=int, help="Ner Num", default=7)
    # 实体词向量的维度
    parser.add_argument('--ner_dim', type=int, help="Ner embed size", default=20)
    # 共指嵌入的维度
    parser.add_argument('--coref_dim', type=int, help="Coref embedd size", default=20)
    # 位置嵌入的维度
    parser.add_argument('--pos_dim', type=int, help="relative position embed size", default=20)

    # Optimizer (Only useful for Bert)，对于bert的优化
    # 使用自适应学习率。
    parser.add_argument('--lr', type=float, default=1e-3, help='Applies to sgd and adagrad.')
    # L2正则化（权重衰减）,原作者是0.01
    parser.add_argument('--weight_decay', type=float, default=0.01, help='.')
    # Gradient Clipping(梯度裁剪)去解决梯度爆炸问题。
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    # Dropout
    # Dropout可以作为训练深度神经网络的一种trick供选择。在每个训练批次中，通过忽略一半【可以设置】的特征检测器（让一半的隐层节点值为0），可以明显地减少过拟合现象。
    # 设置输入层的Dropout
    parser.add_argument('--input_dropout', type=float, help="", default=0.5)
    # 设置RNN的Dropout
    parser.add_argument('--rnn_dropout', type=float, help="", default=0.2)
    # 设置多层感知机的Dropout
    parser.add_argument('--mlp_dropout', type=float, help="", default=0.2)

    # Classifier
    # 是否使用全局表征（文档嵌入）使用None
    parser.add_argument('--doc_embed_required', type=int, help="Whether use global entity rep", default=0)
    # 是否使用局部表征（path_bilstm）
    parser.add_argument('--local_required', type=int, help="Whether use local entity rep", default=0)
    # 设置最大序列长度（默认512）
    parser.add_argument('--max_sent_len', type=int, help="The maximum sequence length", default=512)
    # 设置分类器标签种类（也就是关系的种类）96种关系+1种无关系
    parser.add_argument('--class_num', type=int, help="Number of classification labels", default=97)

    args = parser.parse_args()

    return args


def train(model, evaluator, logger, dataset_train, dataset_valid, args):
    # 设置模型为训练模式
    model.train()
    # 默认对一个batch里面的数据做二元交叉熵并且求平均
    criterion = torch.nn.BCELoss().cuda()

    # 定义参数（如果要求梯度的话）
    parameters = [p for p in model.parameters() if p.requires_grad]
    # AdamW：一种优化算法； weight_decay：权重衰减；  lr：学习率
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay, lr=args.lr)
    # optimizer：网络的优化器
    # 'min' - 监控量停止下降的时候，学习率将减小
    # 'max' - 监控量停止上升的时候，学习率将减小为min
    # patience：容忍网路的性能不提升的次数，高于这个次数就降低学习率
    # 学习率每次降低倍数，new_lr = old_lr * factor
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)

    train_pred, train_gold, train_hit = 0, 0, 0
    best_step, best_valid_p, best_valid_r, best_valid_f1 = 0, 0, 0, 0

    train_loss = []
    # 按epoch进行训练。
    for ep in range(args.max_epoch):
        # 经过处理后的训练集（处理过程还不清晰）
        # 每轮循环会先调用Dataset中的__next__方法，然后执行循环体,sample是每个文档经过__next__方法修改后的结果
        for i, sample in enumerate(dataset_train):
            print("ep:{},doc：{}".format(ep, i))
            label = sample['label'].cuda()

            # profile = line_profiler.LineProfiler(DocClassifier.forward)  # 把函数传递到性能分析器
            # profile.enable()  # 开始分析

            # 样本输入模型，结果张量维度是样本数*关系种类（96+1）
            prob = model(sample)
            # profile.disable()  # 停止分析
            # profile.print_stats()
            # prob_next = prob[sample['posToNeg_head_tail']+sample['num_pos_sample']:]
            # prob = torch.cat([prob, prob_next])
            # label_next = label[sample['posToNeg_head_tail']+sample['num_pos_sample']:]
            # label = torch.cat([label, label_next])
            # 计算损失
            loss = criterion(prob, label)
            # 梯度重置为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度正则？
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            # 更新参数
            optimizer.step()

            # 结果进行四舍五入
            predict = prob.round()
            # 去掉第一列（去掉第一类关系，第一类关系表示没有明确表达有关系，只预测有“答案”的关系。）
            sub_gold = label[:, 1:]
            sub_pred = predict[:, 1:]
            hit = torch.sum(sub_gold * torch.eq(sub_pred, sub_gold).to(torch.float32)).item()
            pred = torch.sum(sub_pred).item()
            gold = torch.sum(sub_gold).item()
            # gold是标准答案的，pred是预测的，hit是命中的
            train_gold, train_pred, train_hit = train_gold + gold, train_pred + pred, train_hit + hit

            train_loss.append(loss.item())
            # if i > 3050:
            #     print(1)

        logger.info("============= Global Epoch: {} ============".format(ep))
        train_p = train_hit / max(train_pred, 1)
        train_r = train_hit / max(train_gold, 1)
        train_f1 = 2 * train_p * train_r / max((train_p + train_r), 1)
        logger.info("Train Loss: {:.8f}, P: {:.6f}, R: {:.6f}, F1: {:.6f}".format(
            np.mean(train_loss), train_p, train_r, train_f1))

        if ep >= args.warmup_step:
            model.eval()
            valid_loss, valid_p, valid_r, valid_f1, valid_threshold, prob = evaluator.get_evaluation(
                dataset_valid,
                model, criterion,
                args)
            fmt_str = "Valid Loss: {:.8f}, P: {:.6f}, R: {:.6f}, F1: {:.6f}, Threshold: {:.3f}"
            logger.info(fmt_str.format(valid_loss, valid_p, valid_r, valid_f1, valid_threshold))
            model.train()

            scheduler.step(valid_f1)

            train_gold, train_pred, train_hit = 0, 0, 0
            if best_valid_f1 < valid_f1:
                best_step, best_valid_p, best_valid_r, best_valid_f1 = ep, valid_p, valid_r, valid_f1
                logger.info("Cur Best")
                if args.save_model:
                    # with open(os.path.join(args.model_path, 'best-prob'), 'w') as fh:
                    # json.dump(prob, fh, indent=2)
                    torch.save(model.state_dict(), os.path.join(args.model_path, 'best-model'))
        else:
            # acc_hit, acc_tot = 0, 0
            train_gold, train_pred, train_hit = 0, 0, 0
    logger.info("Best step: {}".format(best_step))
    logger.info("Best Valid P: {:.5f}, R: {:.5f}, F1: {:.5f}".format(best_valid_p, best_valid_r, best_valid_f1))


def eval(model, evaluator, dataset_valid, args):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    valid_loss, valid_p, valid_r, valid_f1, valid_threshold, prob = evaluator.get_evaluation(dataset_valid,
                                                                                             model, criterion,
                                                                                             args)
    print(valid_f1, valid_threshold)
    with open(os.path.join(args.model_path, 'test-prob'), 'w') as fh:
        json.dump(prob, fh, indent=2)


def print_configuration_op(logger, args):
    logger.info("My Configuration:")
    logger.info(args)
    print("My Configuration: ")
    print(args)


def main():
    args = arg_parser()
    # 设置程序可见的显卡ID（0），这个步骤需要在所有cuda最前面设置值。
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    # Random Seed
    # 数据集的随机种子，random.random()可以生成一个随机数
    random.seed(args.dataset_seed)
    # 模型的随机种子，np.random.random(维度)可以生成一个随机的n维数组，np.random.random((1000, 20))生成1000行 20列的浮点数，浮点数都是从0.0-1.0中随机。
    np.random.seed(args.model_seed)
    # 为CPU设置种子用于生成随机数，torch.random(2, 3)可以生成一个2*3的随机数矩阵。
    torch.manual_seed(args.model_seed)
    # 如果使用GPU的话，为所有的GPU设置相同的随机种子。
    torch.cuda.manual_seed_all(args.model_seed)

    # Logger setting
    if args.mode == 'train':
        # ../log/100_lower-doc/doc_0-pos_20-seed_31_32
        # 除了最后一个取全部，然后用/进行拼接===》../log/100_lower-doc
        log_dir = '/'.join(args.log_path.split('/')[:-1])
        # 判断../log/100_lower-doc是否为目录
        if not os.path.isdir(log_dir):
            # 如果不是目录（不存在），创建目录。
            os.makedirs(log_dir)
        # 如果文件存在，doc_0-pos_20-seed_31_32,表示训练过了，那就在这个模型基础上进行训练（微调）
        if os.path.exists(args.log_path):
            # ../log/100_lower-doc/doc_0-pos_20-seed_31_32finetune
            args.log_path += 'finetune'

        logging.basicConfig(filename=args.log_path,
                            format='[%(asctime)s:%(message)s]',
                            level=logging.INFO,
                            filemode='w',
                            datefmt='%Y-%m-%d %I:%M:%S')
        logger = logging.getLogger()
        # 如果保存模型，并且模型路径不存在，那就新建目录
        if args.save_model and not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

        print_configuration_op(logger, args)
    else:
        # 如果不是训练就不保存日志
        logger = None

    # Build Model
    # 加载预训练词嵌入模型glove的矩阵结果   ../dataset/DocRED_baseline_metadata/glove_100_lower_vec.npy
    pre_trained_embed = np.load(args.pre_trained_embed)

    opt = dict(
        # 最大序列长度
        max_len=args.max_sent_len,
        #
        topn=args.topn,
        # 关系种类
        num_class=args.class_num,
        # 隐含层的维度（256）
        hidden_dim=args.hidden_size,
        # 词典的大小(矩阵的行)，48439个，向量对应的id在word2id.json文件中
        vocab_size=pre_trained_embed.shape[0],
        # glove词嵌入维度（矩阵的列）
        emb_dim=pre_trained_embed.shape[1],
        # 实体类型个数
        ner_num=args.ner_num,
        # 实体嵌入维度
        ner_dim=args.ner_dim,
        # 位置嵌入维度
        pos_dim=args.pos_dim,
        # 共指嵌入维度
        coref_dim=args.coref_dim,
        # 设置3个地方的dropout(丢弃率)---要想DropOut层有效果，需要显示调用train()方法。
        input_dropout=args.input_dropout,
        rnn_dropout=args.rnn_dropout,
        mlp_dropout=args.mlp_dropout,
        # 首次的GCN层数
        first_gcn_layers=2,
        # 带权GCN层数
        last_gcn_layers=1,
    )
    # 把字典（参数）和glove模型生成的词向量矩阵作为输入。
    model = DocClassifier(opt, pre_trained_embed)

    print(model)
    # 使用显卡的数量
    device_num = len(args.gpu_no.split(','))
    # 使用id为0,1... device_num-1的GPU
    device_ids = list(range(device_num))
    # 使用了多个显卡
    if device_num > 1:
        device_ids = args.gpu_no.split(',')
        # 模型并行（多个显卡）
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        # 使用了一个显卡（id=0）,对模型使用GPU加速
        model.cuda()
    # parameters = list(model.parameters())
    # import ipdb; ipdb.set_trace()
    # 模型路径存在（保存了模型），那么下次训练，就在前面模型基础上进行
    if os.path.exists(os.path.join(args.model_path, 'best-model')):
        # 加载已经存在的模型（只包含模型的参数）。
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'best-model')))
        print("================Success Load Model !!==================")
    # if args.mode == 'eval':
    # model.load_state_dict(torch.load(os.path.join(args.model_path, 'best-model')))
    # model.cuda()
    # 保存日志a
    evaluator = Evaluator(logger)

    # Build Dataset
    print("Building Dataset")

    from path_dataset_2 import Dataset
    if args.mode == 'train':
        '''
        dataset_train = Dataset(os.path.join(args.data_dir, 'small_train'), 97, True, max_entity_pair_num=args.max_entity_pair_num, args=args)
        dataset_valid = Dataset(os.path.join(args.data_dir, 'small_train'), 97, False, args=args)
        '''
        # ../dataset/100_lower-doc/train
        dataset_train = Dataset(os.path.join(args.data_dir, 'train'), 97, True,
                                max_entity_pair_num=args.max_entity_pair_num, mode="train", args=args)
        print("---------------")

        dataset_valid = Dataset(os.path.join(args.data_dir, 'dev'), 97, False, mode="dev", args=args)
        # '''
    elif args.mode == 'eval_dev':
        dataset_test = Dataset(os.path.join(args.data_dir, 'dev'), 97, False, mode="dev", args=args)
    elif args.mode == 'eval_test':
        dataset_test = Dataset(os.path.join(args.data_dir, 'test'), 97, False, args=args)

    print("Finish Building Dataset")
    # Train and Eval

    if args.mode == 'train':
        # train(model, evaluator, logger, dataset_valid, dataset_valid, args)
        # 开始训练
        train(model, evaluator, logger, dataset_train, dataset_valid, args)
    elif args.mode in ['eval_dev', 'eval_test']:
        eval(model, evaluator, dataset_test, args)
    # else:
    # assert False, "mode is {}, which is not train or eval".format(args.mode)

if __name__ == '__main__':
    main()
