# @Time   : 2024/7/12
# @Author : Ablation Study
# ============================================================
# 消融变体: JK-Net GCN (多尺度聚合) —— 聚合所有级联层输出替代仅使用最后一层
# 回应审稿人: "compare with multi-scale aggregation or other anti-over-smoothing strategies"
# 目的: 验证级联架构是否因天然的多尺度结构而优于仅用最后一层 (JK-Net baseline)
# 控制变量: TextCNN/MLP/对抗训练/级联架构均不变，仅将输出从末层改为多层max-pool聚合
# ============================================================

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_pjf.model.hbrbr import GCN, TextCNN, generate_adjacent_matrix
from recbole_pjf.model.models import AdversarialLearning


class BGNN_JKNet(object):
    """JK-Net GNN —— 收集级联各层输出，max-pool聚合 (Jumping Knowledge)"""

    def __init__(self, config):
        super(BGNN_JKNet, self).__init__()
        def _cfg(key, default):
            try:
                return config[key]
            except (KeyError, AttributeError):
                return default

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
        for i in range(self.layer_depth):
            if i % 2 == 0:
                one_gcn_layer = GCN(self.v_attr_dimensions, self.u_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer,
                                        self.u_attr_dimensions, self.v_attr_dimensions,
                                        self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
            else:
                one_gcn_layer = GCN(self.u_attr_dimensions, self.v_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer,
                                        self.v_attr_dimensions, self.u_attr_dimensions,
                                        self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
        return gcn_layers, adversarial_layers

    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def _jk_aggregate(self, layer_outputs):
        """JK-Net max-pool 聚合: 收集多个层的输出，逐元素取最大值"""
        stacked = torch.stack(layer_outputs, dim=0)  # (num_layers, N, dim)
        return torch.max(stacked, dim=0)[0]          # (N, dim)

    def __layer_inference(self, gcn, adversarial, real_batch_num, real_num,
                          real_embedding, real_adj, fake_embedding, step, is_training):
        for i in range(self.epochs):
            for iter in range(real_batch_num):
                start_index = self.batch_size * iter
                end_index = self.batch_size * (iter + 1)
                if iter == real_batch_num - 1:
                    end_index = real_num
                attr_batch = real_embedding[start_index:end_index]
                adj_batch_temp = real_adj[start_index:end_index]
                adj_batch = self.__sparse_mx_to_torch_sparse_tensor(adj_batch_temp).to(device=self.device)

                gcn_output = gcn(fake_embedding, adj_batch)
                if is_training:
                    lossD, lossG = adversarial.forward_backward(attr_batch, gcn_output,
                                                                step=step, epoch=i, iter=iter)
        new_real_embedding = torch.FloatTensor([]).to(self.device)
        for iter in range(real_batch_num):
            start_index = self.batch_size * iter
            end_index = self.batch_size * (iter + 1)
            if iter == real_batch_num - 1:
                end_index = real_num
            adj_batch_temp = real_adj[start_index:end_index]
            adj_batch = self.__sparse_mx_to_torch_sparse_tensor(adj_batch_temp).to(device=self.device)
            gcn_output = gcn(torch.as_tensor(fake_embedding, device=self.device), adj_batch)
            new_real_embedding = torch.cat((new_real_embedding, gcn_output.detach()), 0)
        return new_real_embedding

    def adversarial_learning(self, u_previous_embedding, v_previous_embedding,
                             u_adj, v_adj, num, is_training):
        if num != self.u_num:
            u_adj, v_adj = generate_adjacent_matrix(num)

        # 收集各层的中间输出 (用于 JK-Net 聚合)
        u_layer_outputs = [u_previous_embedding]  # 第0层输入也纳入聚合
        v_layer_outputs = [v_previous_embedding]

        if self.learning_type == 'inference':
            for i in range(self.layer_depth):
                if i % 2 == 0:
                    u_previous_embedding = self.__layer_inference(
                        self.gcn_layers[i], self.adversarial_layers[i],
                        self.batch_num_u, self.u_num,
                        u_previous_embedding, u_adj,
                        v_previous_embedding, i, is_training)
                    u_layer_outputs.append(u_previous_embedding)
                else:
                    v_previous_embedding = self.__layer_inference(
                        self.gcn_layers[i], self.adversarial_layers[i],
                        self.batch_num_v, self.v_num,
                        v_previous_embedding, v_adj,
                        u_previous_embedding, i, is_training)
                    v_layer_outputs.append(v_previous_embedding)

        # JK-Net 聚合: max-pool 所有层的输出
        u_out = self._jk_aggregate(u_layer_outputs)
        v_out = self._jk_aggregate(v_layer_outputs)
        return u_out, v_out


class HBRBR_JKNetGCN(GeneralRecommender):
    """消融变体: JK-Net 多尺度聚合替代末层输出，对抗训练/级联不变"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HBRBR_JKNetGCN, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        self.embedding_size = config['embedding_size']
        self.geek_channels = config['max_sent_num']
        self.job_channels = config['max_sent_num']

        try:
            bgnn_u_num = config['bgnn_u_num']
        except (KeyError, AttributeError):
            bgnn_u_num = 256
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
        self.bgnn = BGNN_JKNet(config)
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
