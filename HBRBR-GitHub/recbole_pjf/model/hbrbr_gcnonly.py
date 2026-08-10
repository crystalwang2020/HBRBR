# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn
# ============================================================
# 消融变体 2: GCN Only —— 保留 GCN 图传播，去掉对抗训练，改用 MSE 重建损失
# 目的: 剥离对抗机制，验证对抗训练 vs 简单重建的增益
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_pjf.model.hbrbr import GCN, TextCNN, generate_adjacent_matrix


class BGNN_GCNOnly(nn.Module):
    """仅 GCN 传播，无对抗模块 —— 用 MSE 损失训练 GCN 层"""

    def __init__(self, config=None):
        super(BGNN_GCNOnly, self).__init__()
        self.weight_decay = 0.1
        self.dropout = 0.3
        self.batch_num_u = 8
        self.batch_num_v = 8
        self.layer_depth = 2
        self.epochs = 1
        self.learning_type = 'inference'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 100

        self.v_attr_dimensions = 64
        self.learning_rate = 0.001
        self.u_num = 256
        self.u_attr_dimensions = 64
        self.v_num = 256

        self.gcn_layers = nn.ModuleList()
        self.gcn_optimizers = []  # 每个 GCN 层独立的 optimizer
        self.__layer_initialize()

    def __layer_initialize(self):
        for i in range(self.layer_depth):
            if i % 2 == 0:
                one_gcn_layer = GCN(self.v_attr_dimensions, self.u_attr_dimensions)
            else:
                one_gcn_layer = GCN(self.u_attr_dimensions, self.v_attr_dimensions)
            self.gcn_layers.append(one_gcn_layer)
            opt = optim.SGD(one_gcn_layer.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay)
            self.gcn_optimizers.append(opt)

    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def __layer_inference(self, gcn, optimizer_G, real_batch_num, real_num,
                          real_embedding, real_adj, fake_embedding, is_training):
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
                    loss_recon = F.mse_loss(gcn_output, attr_batch)
                    optimizer_G.zero_grad()
                    loss_recon.backward(retain_graph=True)
                    optimizer_G.step()

        # 训练完成后做一次完整前向传播，得到最终输出
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

    def gcn_propagation(self, u_previous_embedding, v_previous_embedding,
                         u_adj, v_adj, num, is_training):
        if num != 256:
            u_adj, v_adj = generate_adjacent_matrix(num)
        if self.learning_type == 'inference':
            for i in range(self.layer_depth):
                if i % 2 == 0:
                    u_previous_embedding = self.__layer_inference(
                        self.gcn_layers[i], self.gcn_optimizers[i],
                        self.batch_num_u, self.u_num,
                        u_previous_embedding, u_adj,
                        v_previous_embedding, is_training)
                else:
                    v_previous_embedding = self.__layer_inference(
                        self.gcn_layers[i], self.gcn_optimizers[i],
                        self.batch_num_v, self.v_num,
                        v_previous_embedding, v_adj,
                        u_previous_embedding, is_training)
        return u_previous_embedding, v_previous_embedding


class HBRBR_GCNOnly(GeneralRecommender):
    """消融变体2: GCN 传播 + MSE 重建损失，无对抗训练"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HBRBR_GCNOnly, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']

        self.embedding_size = config['embedding_size']
        self.geek_channels = config['max_sent_num']
        self.job_channels = config['max_sent_num']

        self.u_adj, self.v_adj = generate_adjacent_matrix(256)

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

        self.bgnn = BGNN_GCNOnly()
        self.bgnn.to(self.device)

        self.loss = BPRLoss()
        self.apply(xavier_normal_initialization)

    def forward(self, geek_sents, job_sents, is_training=True):
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        geek_vec = self.geek_layer(geek_vec)
        job_vec = self.job_layer(job_vec)

        geek_vec_bgnn, job_vec_bgnn = self.bgnn.gcn_propagation(
            geek_vec, job_vec, self.u_adj, self.v_adj,
            job_vec.shape[0], is_training)

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
