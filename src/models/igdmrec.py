# @email: zyguo0166@gmail.com
r"""
IGDMRec
# Update: 2026/02/05
"""


import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from common.abstract_recommender import GeneralRecommender

class IGDMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(IGDMREC, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.n_nodes = self.n_users + self.n_items
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32) #返回交互矩阵
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None

        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_0.1.pt')
        mm_adj_file_norm = os.path.join(dataset_path, 'mm_adj_freedomdsp_0.1norm.pt')

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        adj1, indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        adj2, indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.mm_adj = self.mm_image_weight * adj1 + (1-self.mm_image_weight) * adj2
        self.mm_adj_norm = self.mm_image_weight * image_adj + (1-self.mm_image_weight) * text_adj
        torch.save(self.mm_adj, mm_adj_file)
        torch.save(self.mm_adj_norm, mm_adj_file_norm)


    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        #mask = indices[0] != indices[1]
        #indices = indices[:, mask]
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        #mm1 = adj.to_dense()
        return adj, indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        for (i, j), value in data_dict.items():
            A[i, j] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values


    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values


    def forward(self, mm_adj, IIG, flag_train):

        if flag_train:
            adj = self.masked_adj
        else:
            adj = self.norm_adj

        h = self.item_id_embedding.weight
        all_h = [h]
        for i in range(self.n_layers):
            side_h = torch.sparse.mm(mm_adj, h)
            h = side_h
            all_h += [h]
        all_h = torch.stack(all_h, dim=1)
        all_h = all_h.mean(dim=1, keepdim=False)

        #original
        h2 = self.item_id_embedding.weight
        all_h2 = [h2]
        for i in range(self.n_layers):
            side_h2 = torch.sparse.mm(IIG, h2)
            h2 = side_h2
            all_h2 += [h2]
        all_h2 = torch.stack(all_h2, dim=1)
        all_h2 = all_h2.mean(dim=1, keepdim=False)


        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(ego_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings, h, h2



