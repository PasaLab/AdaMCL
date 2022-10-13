# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import heapq
import itertools
from scipy.sparse import coo_matrix
from time import time


class AdaMCL(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(AdaMCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        ##
        self.TopK =config['TopK']
        self.path = './dataset/' + str(config['dataset'])
        self.user_threshold = config['user_threshold']
        self.item_threshold = config['item_threshold']
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.info_temp = config['ssl_temp']
        self.info_reg = config['info_reg']
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.hyper_layer = config['hyper_layer']

        ## load dataset changed

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat, self.deg = self.get_norm_adj_mat()
        self.norm_adj_mat= self.norm_adj_mat.to(self.device)
        self.deg[self.deg == 0] = 1
        log_deg = np.log(self.deg)
        mean_deg = np.mean(log_deg)
        log_deg = log_deg / mean_deg
        self.log_deg = torch.tensor(log_deg).reshape(-1, 1).to(self.device).to(torch.float32)


        self.norm_myadj_mat = self.get_mydata().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def argmax_top_k(self,a, top_k=50):
        ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
        return np.array([idx for ele, idx in ele_idx], dtype=np.intc)

    def GetAdj(self, R, topK, train_N, num_users, low_pass):

        union = (R.dot(R.T))

        value = union.data
        row_pointers = union.indptr
        column_index = union.indices

        def GetValue(row_index, value, column_index, train_N):
            for i in range(len(value)):
                column = column_index[i]
                row = row_index[i]
                value[i] = value[i] / (train_N[0][row] + train_N[0][column] - value[i])
                if column == row:
                    value[i] = 0
            return value

        row_index = union.tocoo().row
        row_index = np.array(row_index)
        train_N = np.array(train_N)
        value = GetValue(row_index, value, column_index, train_N)
        norm_adj = coo_matrix((value, (row_index, column_index)), shape=(num_users, num_users))
        norm_adj.data[np.isinf(norm_adj.data)] = 0.
        norm_adj.data = np.where(norm_adj.data > low_pass, norm_adj.data, 0.)
        norm_adj.eliminate_zeros()
        norm_adj = norm_adj.tocsr()
        norm_row_pointers = norm_adj.indptr
        norm_column_index = norm_adj.indices
        norm_data = norm_adj.data
        user_topk_columns = []
        user_topk_values = []
        user_topk_rows = []
        user_neighbor = [[] for _ in range(num_users)]
        user_neighbor_value = [[] for _ in range(num_users)]
        user_neighbor_deg = [[] for _ in range(num_users)]
        max_len = 0
        for i in range(num_users):
            start = row_pointers[i]
            end = row_pointers[i + 1]
            norm_start = norm_row_pointers[i]
            norm_end = norm_row_pointers[i + 1]

            value_index = self.argmax_top_k(value[start:end], topK) + start
            column1 = column_index[value_index]
            column2 = norm_column_index[norm_start:norm_end]

            fin_column = column1 if len(column1) > len(column2) else column2
            fin_value = value[value_index] if len(column1) > len(column2) else norm_data[norm_start:norm_end]

            f_sum = np.sum(fin_value)
            user_topk_columns.extend(fin_column)
            user_topk_values.extend(fin_value / f_sum)
            user_topk_rows.extend(np.array([i]).repeat(len(fin_column)))
            user_neighbor[i] = fin_column.tolist()
            user_neighbor_value[i] = (fin_value / f_sum).tolist()
            user_neighbor_deg[i] = (train_N[0][fin_column]).tolist()

            max_len = max(max_len, len(fin_column))
        matrix = coo_matrix((user_topk_values, (user_topk_rows, user_topk_columns)), shape=(num_users, num_users))
        mc = matrix.tocsr()
        return mc, user_neighbor, user_neighbor_value, user_neighbor_deg, max_len

    def get_mydata(self):
        try:
            t3 = time()
            Myadj_mat = sp.load_npz(
                self.path + '/s_Myadj_%d_%f_%f.npz' % (
                    self.TopK, self. user_threshold, self.item_threshold))
            print('already load topK matrix', Myadj_mat.shape, time() - t3)
            R = self.interaction_matrix.tolil()
        except:
            t4 = time()
            R = self.interaction_matrix.tolil()
            topK = self.TopK
            train_N = np.sum(R, dtype=np.float32, axis=1).reshape(-1, )
            train_M = np.sum(R.T, dtype=np.float32, axis=1).reshape(-1, )
            U_u, u_neighbor, u_neighbor_values, user_topk_deg, user_max_len = self.GetAdj(R, topK, train_N, self.n_users,
                                                                                          self.user_threshold)
            I_i, i_neighbor, i_neighbor_values, item_topk_deg, item_max_len = self.GetAdj(R.T, topK, train_M,
                                                                                          self.n_items,
                                                                                          self.item_threshold)
            Myadj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            Myadj_mat = Myadj_mat.tolil()

            # Myadj_mat[:self.n_users, :self.n_users] = U_u.tolil()
            # Myadj_mat[self.n_users:, self.n_users:] = I_i.tolil()

            key = 20
            for i in range(key):
                Myadj_mat[int(self.n_users * i / key):int(self.n_users * (i + 1.0) / key), 0:self.n_users] = \
                    U_u[int(self.n_users * i / key):int(self.n_users * (i + 1.0) / key)]

                Myadj_mat[self.n_users + int(self.n_items * i / key):self.n_users + int(self.n_items * (i + 1.0) / key),
                self.n_users:] = \
                    I_i[int(self.n_items * i / key):int(self.n_items * (i + 1.0) / key)]

            Myadj_mat = Myadj_mat.todok()
            Myadj_mat.setdiag(0)
            Myadj_mat = Myadj_mat.tocsr()

            print('already creat topK matrix', Myadj_mat.shape, time() - t4)
            sp.save_npz(self.path + '/s_Myadj_%d_%f_%f.npz' % (
                self.TopK, self.user_threshold, self.item_threshold),
                        Myadj_mat)

        L = Myadj_mat.tocoo().astype(np.float32)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        deg = np.array(sumArr.flatten())[0]
        np.savetxt(self.path+'/deg.txt', deg,fmt='%d')
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL,deg

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        embeddings_y = []
        embeddings_z = []
        for layer_idx in range(1, self.n_layers + 1):
            all_embeddings1 = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            all_embeddings2 = torch.sparse.mm(self.norm_myadj_mat, all_embeddings)

            embeddings_y.append(all_embeddings1)
            embeddings_z.append(all_embeddings2)

            a_1 = torch.nn.functional.normalize(all_embeddings1, p=2, dim=1)
            b_1 = torch.nn.functional.normalize(all_embeddings1 + all_embeddings2, p=2, dim=1)
            sim_1 = torch.multiply(a_1, b_1).sum(dim=1).reshape(-1,1)
            sim_1 = torch.clamp(sim_1, min = 0.0 )
            beta = self.gamma / (layer_idx + sim_1 * self.log_deg)
            all_embeddings2 = torch.multiply(all_embeddings2, beta)
            all_embeddings = all_embeddings1 + all_embeddings2
            embeddings_list.append(all_embeddings)

        hyper_layer_embedding = embeddings_list[self.hyper_layer]

        user_all_embeddings_x, item_all_embeddings_x = torch.split(hyper_layer_embedding, [self.n_users, self.n_items])

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        lightgcn_all_embeddings_y = torch.stack(embeddings_y, dim=1)
        lightgcn_all_embeddings_y = torch.mean(lightgcn_all_embeddings_y, dim=1)

        lightgcn_all_embeddings_z = torch.stack(embeddings_z, dim=1)
        lightgcn_all_embeddings_z = torch.mean(lightgcn_all_embeddings_z, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_y, item_all_embeddings_y = torch.split(lightgcn_all_embeddings_y, [self.n_users, self.n_items])
        user_all_embeddings_z, item_all_embeddings_z = torch.split(lightgcn_all_embeddings_z, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, user_all_embeddings_x, item_all_embeddings_x, \
               user_all_embeddings_y, item_all_embeddings_y, user_all_embeddings_z, item_all_embeddings_z

    def creat_infoloss(self, users, a_embeddings, b_embeddings,
                      ssl_temp, ssl_reg):
       # argument settings
       batch_users = torch.unique(users)
       user_emb1 = torch.nn.functional.normalize(a_embeddings[batch_users], p=2, dim=1)
       user_emb2 = torch.nn.functional.normalize(b_embeddings[batch_users], p=2, dim=1)
       # user
       user_pos_score = torch.multiply(user_emb1, user_emb2).sum(dim=1)
       user_ttl_score = torch.matmul(user_emb1, user_emb2.t())
       user_pos_score = torch.exp(user_pos_score / ssl_temp)
       user_ttl_score = torch.exp(user_ttl_score / ssl_temp).sum(dim=1)
       user_ssl_loss = -torch.log(user_pos_score / user_ttl_score).sum()
       ssl_loss = ssl_reg * user_ssl_loss

       return ssl_loss

    def creat_sslloss(self, user, pos_item, user_all_embeddings, item_all_embeddings, user_all_embeddings_x,
                      item_all_embeddings_x, user_all_embeddings_y, item_all_embeddings_y, user_all_embeddings_z,
                      item_all_embeddings_z):
        user_ssl = self.creat_infoloss(user, user_all_embeddings, user_all_embeddings_x, self.ssl_temp, self.ssl_reg)

        item_ssl = self.creat_infoloss(pos_item, item_all_embeddings, item_all_embeddings_x, self.ssl_temp,
                                       self.ssl_reg)

        user_infoy = self.creat_infoloss(user, user_all_embeddings_y, user_all_embeddings, self.info_temp,
                                         self.info_reg)
        item_infoy = self.creat_infoloss(pos_item, item_all_embeddings_y, item_all_embeddings, self.info_temp,
                                         self.info_reg)

        user_infoz = self.creat_infoloss(user, user_all_embeddings_z, user_all_embeddings, self.info_temp,
                                         self.info_reg)
        item_infoz = self.creat_infoloss(pos_item, item_all_embeddings_z, item_all_embeddings, self.info_temp,
                                         self.info_reg)

        return user_ssl + item_ssl,  user_infoy + item_infoy + self.alpha * (user_infoz + item_infoz)

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, \
        user_all_embeddings_x, item_all_embeddings_x, \
        user_all_embeddings_y, item_all_embeddings_y, \
        user_all_embeddings_z, item_all_embeddings_z = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings,require_pow=True)

        ssl_loss,info_loss = self.creat_sslloss(user, pos_item, user_all_embeddings, item_all_embeddings,
                                                user_all_embeddings_x,item_all_embeddings_x,
                                                user_all_embeddings_y, item_all_embeddings_y,
                                                user_all_embeddings_z, item_all_embeddings_z)

        return mf_loss + self.reg_weight * reg_loss,ssl_loss,info_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, \
        user_all_embeddings_x, item_all_embeddings_x, \
        user_all_embeddings_y, item_all_embeddings_y, \
        user_all_embeddings_z, item_all_embeddings_z = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, \
            user_all_embeddings_x, item_all_embeddings_x, \
            user_all_embeddings_y, item_all_embeddings_y, \
            user_all_embeddings_z, item_all_embeddings_z = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def full_predict(self, user):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, \
            user_all_embeddings_x, item_all_embeddings_x, \
            user_all_embeddings_y, item_all_embeddings_y, \
            user_all_embeddings_z, item_all_embeddings_z = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores
