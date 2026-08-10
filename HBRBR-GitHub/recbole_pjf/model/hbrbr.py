# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

from recbole_pjf.gcn.layers import GraphConvolution
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_pjf.model.models import AdversarialLearning
from recbole_pjf.model.prediction_heads import get_prediction_head


class GCN(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(infeat, outfeat)

    def forward(self, x, adj):
        x = torch.tanh(self.gc1(x, adj))
        return x

    def aggregation(self, x, adj):
        x = self.gc1(x, adj)
        return x


class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim))
        )
        if method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method == 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        # import pdb
        # pdb.set_trace()   # [2048, 20, 30, 64]
        x = self.net1(x)  # [2048, 20, 13, 64]
        x = self.net2(x).squeeze(2)  # [2048, 20, 64]
        x = self.pool(x).squeeze(1)  # [2048, 64]
        return x




class BGNN(object):
    def __init__(self, config):
        super(BGNN, self).__init__()
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
        for i in range(self.layer_depth):
            if i % 2 == 0:
                # 特征维度  max_num 和 max_len embedding_size的乘积
                one_gcn_layer = GCN(self.v_attr_dimensions, self.u_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer, self.u_attr_dimensions, self.v_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
            else:
                one_gcn_layer = GCN(self.u_attr_dimensions, self.v_attr_dimensions).to(self.device)
                gcn_layers.append(one_gcn_layer)
                adversarial_layers.append(
                    AdversarialLearning(one_gcn_layer, self.v_attr_dimensions, self.u_attr_dimensions, self.dis_hidden_dim, self.learning_rate,
                                        self.weight_decay, self.dropout, self.device, outfeat=1))
        return gcn_layers, adversarial_layers


    def __sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def __layer_inference(self, gcn, adversarial, real_batch_num, real_num, real_embedding, real_adj, fake_embedding,
                          step, is_training):
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
                    lossD, lossG = adversarial.forward_backward(attr_batch, gcn_output, step=step, epoch=i, iter=iter)
                # self.f_loss.write("%s %s\n" % (lossD, lossG))
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
        # self.f_loss.write("###Depth finished!\n")
        return new_real_embedding

    def adversarial_learning(self, u_previous_embedding, v_previous_embedding, u_adj, v_adj, num, is_training):
        # default start with V as input
        if num != self.u_num:
            u_adj, v_adj = generate_adjacent_matrix(num)
        if self.learning_type == 'inference':
            for i in range(self.layer_depth):
                if i % 2 == 0:
                    u_previous_embedding = self.__layer_inference(self.gcn_layers[i], self.adversarial_layers[i],
                                                                  self.batch_num_u,
                                                                  self.u_num, u_previous_embedding, u_adj,
                                                                  v_previous_embedding, i, is_training)
                else:
                    v_previous_embedding = self.__layer_inference(self.gcn_layers[i], self.adversarial_layers[i],
                                                                  self.batch_num_v,
                                                                  self.v_num, v_previous_embedding, v_adj,
                                                                  u_previous_embedding, i, is_training)
        # elif self.learning_type == 'end2end':
            # u_previous_embedding = self.__layer__gcn(u_previous_embedding, u_adj, v_previous_embedding, v_adj, 0)
        return u_previous_embedding, v_previous_embedding


class HBRBR(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HBRBR, self).__init__(config, dataset)
        self.USER_SENTS = config['USER_DOC_FIELD']
        self.ITEM_SENTS = config['ITEM_DOC_FIELD']
        self.neg_prefix = config['NEG_PREFIX']
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.geek_channels = config['max_sent_num']  # 10
        self.job_channels = config['max_sent_num']  # 10
        bgnn_u_num = config.final_config_dict.get('bgnn_u_num', 256)
        self.u_adj, self.v_adj = generate_adjacent_matrix(bgnn_u_num)
        self.emb = nn.Embedding(len(dataset.wd2id.keys()), self.embedding_size, padding_idx=0)

        # define layers and loss
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

        self.pred_head = get_prediction_head(config['prediction_head'], self.embedding_size)
        self.bgnn = BGNN(config)

        self.loss = BPRLoss()
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, geek_sents, job_sents, is_training=True):
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        geek_vec = self.geek_layer(geek_vec)
        job_vec = self.job_layer(job_vec)
        # print(job_vec.shape[0])
        # if is_training:
        geek_vec_bgnn, job_vec_bgnn = self.bgnn.adversarial_learning(geek_vec, job_vec, self.u_adj, self.v_adj, job_vec.shape[0], is_training)
        geek_vec = geek_vec + geek_vec_bgnn
        job_vec = job_vec + job_vec_bgnn

        x = self.pred_head(geek_vec, job_vec)
        return x

    def forward0(self, geek_sents, job_sents, is_neg = False):
        geek_vec = self.emb(geek_sents)
        job_vec = self.emb(job_sents)
        geek_vec = self.geek_layer(geek_vec)
        job_vec = self.job_layer(job_vec)

        # 使用BGNN进行特征提取
        if not is_neg:      # 正样本
            geek_graph_vec, job_graph_vec = self.bgnn.adversarial_learning(geek_vec, job_vec, self.u_adj, self.v_adj, job_vec.shape[0])
            # 特征融合
            geek_final = geek_vec + geek_graph_vec
            job_final = job_vec + job_graph_vec
            # 最终预测    fixme 要求mlp的维度是 之前的 2 倍
            x = torch.cat([geek_final * job_final, geek_final + job_final], dim=1)
        else:
            x = torch.cat([geek_vec * job_vec, geek_vec + job_vec], dim=1)
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

def generate_adjacent_matrix(u_num):
        u_node_list = [i for i in range(u_num)]
        v_node_list = [i for i in range(u_num)]
        edge_list = [(i, i) for i in u_node_list]
        B_u = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        B_u.add_nodes_from(u_node_list, bipartite=0)
        B_u.add_nodes_from(v_node_list, bipartite=1)

        # Add edges only between nodes of opposite node sets
        B_u.add_edges_from(edge_list)

        u_adjacent_matrix_np = biadjacency_matrix(B_u, u_node_list, v_node_list)
        # u_adjacent_matrix_np = u_adjacent_matrix.todense().A
        B_u.clear()

        B_v = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        B_v.add_nodes_from(v_node_list, bipartite=0)
        B_v.add_nodes_from(u_node_list, bipartite=1)

        # Add edges only between nodes of opposite node sets
        B_v.add_edges_from(edge_list)

        v_adjacent_matrix_np = biadjacency_matrix(B_v, v_node_list, u_node_list)

        # v_adjacent_matrix_np = v_adjacent_matrix.todense().A
        B_v.clear()
        return u_adjacent_matrix_np, v_adjacent_matrix_np
