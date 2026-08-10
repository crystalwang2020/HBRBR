# @Time   : 2024/7/12
# @Author : Ablation Study
# ============================================================
# 消融变体: Multi-Hop GNN (直接多层GNN) —— 将级联单跳GCN替换为直接多跳GCN
# 回应审稿人: "compare with direct multi-layer GNN"
# 目的: 验证"将多跳传播分解为连续单跳级联"是否优于直接在每步做多跳GNN
# 控制变量: TextCNN/MLP/对抗训练均不变，仅 GNN 传播模式改变
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_pjf.gcn.layers import GraphConvolution
from recbole_pjf.model.hbrbr import GCN, TextCNN, generate_adjacent_matrix
from recbole_pjf.model.models import AdversarialLearning


class MultiHopGCN(nn.Module):
    """2层GCN模块 —— 在单次级联步骤内完成2跳传播，与级联单跳形成对比"""

    def __init__(self, infeat, midfeat, outfeat):
        super(MultiHopGCN, self).__init__()
        self.gc1 = GraphConvolution(infeat, midfeat)
        self.gc2 = GraphConvolution(midfeat, outfeat)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = torch.tanh(self.gc2(x, adj))
        return x

    def aggregation(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class BGNN_MultiHop(object):
    """多跳GNN —— 每步用2层GCN代替单层GCN，保持交替U/V模式"""

    def __init__(self, config):
        super(BGNN_MultiHop, self).__init__()
        def _cfg(key, default):
            val = config.final_config_dict.get(key, default)
            return val if val is not None else default

        self.weight_decay = _cfg('bgnn_weight_decay', 0.1)
        self.dropout = _cfg('bgnn_dropout', 0.3)
        self.batch_num_u = _cfg('bgnn_batch_num_u', 8)
        self.batch_num_v = _cfg('bgnn_batch_num_v', 8)
        self.layer_depth = _cfg('bgnn_layer_depth', 2)
        self.epochs = _cfg('bgnn_epochs', 1)
        self.learning_type = _cfg('bgnn_learning_type', 'inference')
        self.dis_hidden_dim = _cfg('bgnn_dis_hidden_dim', 1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = _cfg('bgnn_batch_size', 100)

        self.v_attr_dimensions = _cfg('bgnn_v_attr_dimensions', 64)
        self.learning_rate = _cfg('bgnn_learning_rate', 0.001)
        self.u_num = _cfg('bgnn_u_num', 256)
        self.u_attr_dimensions = _cfg('bgnn_u_attr_dimensions', 64)
        self.v_num = _cfg('bgnn_v_num', 256)
        self.gcn_layers, self.adversarial_layers = self.__layer_initialize()

    def __layer_initialize(self):
        gcn_layers = []
        adversarial_layers = []
        # 每步用 MultiHopGCN (2层) 替代原始 GCN (1层)
        # V→U 方向: V_dim → U_dim → U_dim
        # U→V 方向: U_dim → V_dim → V_dim
        for i in range(self.layer_depth):
            if i % 2 == 0:
                one_gcn_layer = MultiHopGCN(
                    self.v_attr_dimensions,
                    self.u_attr_dimensions,
                    self.u_attr_dimensions
                ).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer,
                                        self.u_attr_dimensions,
                                        self.v_attr_dimensions,
                                        self.dis_hidden_dim,
                                        self.learning_rate,
                                        self.weight_decay,
                                        self.dropout,
                                        self.device,
                                        outfeat=1))
            else:
                one_gcn_layer = MultiHopGCN(
                    self.u_attr_dimensions,
                    self.v_attr_dimensions,
                    self.v_attr_dimensions
                ).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer,
                                        self.v_attr_dimensions,
                                        self.u_attr_dimensions,
                                        self.dis_hidden_dim,
                                        self.learning_rate,
                                        self.weight_decay,
                                        self.dropout,
                                        self.device,
                                        outfeat=1))
        return gcn_layers, adversarial_layers

    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def __layer_inference(self, gcn, adversarial, real_batch_num, real_num,
                          real_embedding, real_adj, fake_embedding, step, is_training):
        # MultiHopGCN 需要全量邻接矩阵 (内部两层 GCN 共享 adj，batch 会导致维度不匹配)
        full_adj = self.__sparse_mx_to_torch_sparse_tensor(real_adj).to(device=self.device)

        for i in range(self.epochs):
            for iter in range(real_batch_num):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == real_batch_num - 1:
                    end_index = real_num
                attr_batch = real_embedding[start_index:end_index]

                gcn_output_full = gcn(fake_embedding, full_adj)
                gcn_output = gcn_output_full[start_index:end_index]
                if is_training:
                    lossD, lossG = adversarial.forward_backward(attr_batch, gcn_output,
                                                                step=step, epoch=i, iter=iter)

        gcn_output_full = gcn(torch.as_tensor(fake_embedding, device=self.device), full_adj)
        return gcn_output_full.detach()

    def adversarial_learning(self, u_previous_embedding, v_previous_embedding,
                             u_adj, v_adj, num, is_training):
        if num != self.u_num:
            u_adj, v_adj = generate_adjacent_matrix(num)
        if self.learning_type == 'inference':
            for i in range(self.layer_depth):
                if i % 2 == 0:
                    u_previous_embedding = self.__layer_inference(
                        self.gcn_layers[i], self.adversarial_layers[i],
                        self.batch_num_u, self.u_num,
                        u_previous_embedding, u_adj,
                        v_previous_embedding, i, is_training)
                else:
                    v_previous_embedding = self.__layer_inference(
                        self.gcn_layers[i], self.adversarial_layers[i],
                        self.batch_num_v, self.v_num,
                        v_previous_embedding, v_adj,
                        u_previous_embedding, i, is_training)
        return u_previous_embedding, v_previous_embedding


class HBRBR_MultiHopGCN(GeneralRecommender):
    """消融变体: 每步多跳GCN (替换级联单跳)，对抗训练保留"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HBRBR_MultiHopGCN, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        self.embedding_size = config['embedding_size']
        self.geek_channels = config['max_sent_num']
        self.job_channels = config['max_sent_num']

        bgnn_u_num = config.final_config_dict.get('bgnn_u_num', 256)
        self.u_adj, self.v_adj = generate_adjacent_matrix(bgnn_u_num)

        self.emb = nn.Embedding(len(dataset.wd2id.keys()), self.embedding_size, padding_idx=0)

        self.geek_layer = TextCNN(
            channels=self.geek_channels,
            kernel_size=[(5, 1), (3, 1)],
            pool_size=(2, 1),
            dim=self.embedding_size,
            method='max'
        )
        self.job_layer = TextCNN(
            channels=self.job_channels,
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=self.embedding_size,
            method='mean'
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 1)
        )
        self.bgnn = BGNN_MultiHop(config)
        self.loss = BPRLoss()
        self.apply(xavier_normal_initialization)

    def forward(self, geek_sents, job_sents, is_training=True):
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        geek_vec = self.geek_layer(geek_vec)
        job_vec = self.job_layer(job_vec)

        geek_vec_bgnn, job_vec_bgnn = self.bgnn.adversarial_learning(
            geek_vec, job_vec, self.u_adj, self.v_adj, job_vec.shape[0], is_training)
        geek_vec = geek_vec + geek_vec_bgnn
        job_vec = job_vec + job_vec_bgnn

        x = geek_vec * job_vec
        x = self.mlp(x).squeeze(1)
        return x

    def calculate_loss(self, interaction):
        geek_sents = interaction[self.USER_SENTS]
        job_sents = interaction[self.ITEM_SENTS]
        neg_job_sents = interaction[self.neg_prefix + self.ITEM_SENTS]

        output_pos = self.forward(geek_sents, job_sents)
        output_neg = self.forward(geek_sents, neg_job_sents)

        return self.loss(output_pos, output_neg)

    def predict(self, interaction):
        geek_sents = interaction[self.USER_SENTS]
        job_sents = interaction[self.ITEM_SENTS]
        with torch.no_grad():
            return torch.sigmoid(self.forward(geek_sents, job_sents, False))
