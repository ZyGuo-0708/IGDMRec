

import os
import itertools
import torch
from torch.utils.data import DataLoader, dataset, Dataset
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import torch.nn.functional as F

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator

from models import diffusion_process as dp
from models.model import CDNet


class CustomDataset(Dataset):
    def __init__(self, IIG, adj_session):
        self.IIG = IIG
        self.adj_session_d = adj_session
    def __len__(self):
        return len(self.IIG)
    def __getitem__(self, idx):
        return self.IIG[idx], self.adj_session_d[idx]


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)
        # config
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate_bpr = config['learning_rate_bpr']
        self.learning_rate_diff = config['learning_rate_diff']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']#True
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.train_batch_size = config['train_batch_size']
        self.reweight = config['reweight']
        self.sampling_steps = config['sampling_steps']
        self.sampling_noise = config['sampling_noise']
        self.loss_weight = config['loss_weight']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        # load item graph
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_0.1.pt')  # multimodal I-I path
        mm_norm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_0.1norm.pt') # norm multimodal I-I path
        self.IIG  = torch.load(mm_adj_file).to_dense() # original

        self.IIG_norm  = torch.load(mm_norm_adj_file)
        self.n_items = self.IIG.size()[0]
        self.item_graph_dict = np.load(os.path.join(dataset_path, 'item_graph_dict_10drop3_wo_PageRank.npy'),
                                       allow_pickle=True).item()
        self.adj_session, self.session_adj_norm = self.get_session_adj()
        self.adj_session_d = self.adj_session.to_dense().float()

        # load model
        self.CDNet = CDNet(config, self.n_items, norm=config['norm']).to(
            self.device)
        self.DiffProcess = dp.DiffusionProcess(config, config['noise_schedule'], config['noise_scale'], config['noise_min'], config['noise_max'],
                                               config['steps'], self.device).to(self.device)
        # initialize trainer
        self.start_epoch = 0
        self.cur_step = 0
        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_bpr)
        self.optimizer2 = optim.Adam(self.CDNet.parameters(), lr=self.learning_rate_diff)
        lr_scheduler = config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        scheduler2 = optim.lr_scheduler.LambdaLR(self.optimizer2, lr_lambda=fac)
        self.lr_scheduler = scheduler
        self.lr_scheduler2 = scheduler2
        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.tau = config['tau']
        self.weight_cl = config['weight_cl']
        # diff top-k
        self.diff_topk = 10
        self.has_built_graph_once = False
        self.epoch_idx = 1
    def get_session_adj(self):
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in self.item_graph_dict.keys():
                item_graph_sample = self.item_graph_dict[i][0]
                item_graph_weight = self.item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])
        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0).to(self.device)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), (self.n_items, self.n_items))
        # norm
        return adj, self.compute_normalized_laplacian(indices, (self.n_items, self.n_items))

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
        return adj, indices, self.compute_normalized_laplacian(indices, adj_size)

    def _train_epoch(self, train_data, dataloader,epoch_idx, loss_func=None):
        self.model.train()
        self.CDNet.train()
        total_loss = None
        loss_batches = []

        need_refresh = (epoch_idx % self.epoch_idx == 0) or (not self.has_built_graph_once)  # 5 per epoch
        if need_refresh:
            # train diffusion
            for i in range(5):
                for batch_idx, (IIG_batch, adj_session_d_batch) in enumerate(dataloader):
                    IIG_batch = IIG_batch.to(self.device)
                    adj_session_d_batch = adj_session_d_batch.to(self.device)
                    self.optimizer2.zero_grad()
                    ii_terms = self.DiffProcess.caculate_losses(self.CDNet, IIG_batch, adj_session_d_batch,
                                                                self.reweight)
                    diff_loss = ii_terms["loss"].mean()
                    loss = diff_loss
                    loss.backward()
                    self.optimizer2.step()

            self.CDNet.eval()
            with torch.no_grad():
                half_r = self.n_items // 2
                self.IIG1 = self.IIG[:half_r]
                self.IIG2 = self.IIG[half_r:]
                self.adj_session_d1 = self.adj_session_d[:half_r]
                self.adj_session_d2 = self.adj_session_d[half_r:]
                i_sample1 = self.DiffProcess.p_sample(self.CDNet, self.IIG1, self.adj_session_d1, self.sampling_steps,
                                                      self.sampling_noise)
                i_sample2 = self.DiffProcess.p_sample(self.CDNet, self.IIG2, self.adj_session_d2, self.sampling_steps,
                                                      self.sampling_noise)
                i_sample = torch.cat((i_sample1, i_sample2), dim=0)
                self.denoised_II = self.get_DIFF_II(i_sample)
            self.has_built_graph_once = True

        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()

            users = interaction[0]
            pos_items = interaction[1]
            neg_items = interaction[2]

            # GET u&i according to adj
            ui_u_emb, ui_i_emb, ii_emb, ii_original = self.model.forward(self.denoised_II, self.IIG_norm, 1)

            ia_embeddings = ui_i_emb + ii_emb
            ia_embeddings_o = ui_i_emb + ii_original

            u_g_embeddings = ui_u_emb[users]
            pos_i_g_embeddings = ia_embeddings[pos_items]
            neg_i_g_embeddings = ia_embeddings[neg_items]

            bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                          neg_i_g_embeddings)

            clLoss1 = self.contrastLoss(ia_embeddings_o, ia_embeddings, pos_items, self.tau)
            regloss = calcRegloss(self.model) * 1e-7

            losses = bpr_loss  + regloss +  self.weight_cl* (clLoss1)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
        return total_loss, loss_batches

    def contrastLoss(self, embeds1, embeds2, nodes, temp):
        embeds1 = F.normalize(embeds1, p=2)
        embeds2 = F.normalize(embeds2, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
        return -torch.log(nume / deno).mean()

    def mask_dia_(self, IIG):
        dia_indices = torch.arange(IIG.size(0))
        IIG[dia_indices, dia_indices] = 0
        return IIG

    def get_DIFF_II(self, ii_terms):
        _, knn_ind = torch.topk(ii_terms, self.diff_topk, dim=-1)
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.diff_topk)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        diff_adj = self.compute_normalized_laplacian(indices, ii_terms.size())
        return diff_adj


    def find_common_indices(self, indices, IIG_non_zero_indices):

        mask = torch.zeros(indices.shape[1], dtype=torch.bool, device=indices.device)

        for i in range(indices.shape[1]):
            pair = indices[:, i]
            for j in range(IIG_non_zero_indices.shape[1]):
                if torch.equal(pair, IIG_non_zero_indices[:, j]):
                    mask[i] = True
                    break
        common_indices = indices[:, mask]

        return common_indices
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def _valid_epoch(self, valid_data, epoch_idx):
        valid_result = self.evaluate(valid_data, epoch_idx)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'


    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        # load diff data
        dataset = CustomDataset(IIG=self.IIG, adj_session=self.adj_session_d)
        dataloader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=0)
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, dataloader,epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()
            self.lr_scheduler2.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                test_1 = 2 # without figure
                valid_score, valid_result = self._valid_epoch(valid_data, test_1)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data, epoch_idx)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, epoch_idx, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()
        # batch full users
        batch_matrix_list = []
        with torch.no_grad():
            #ui_u_emb, ui_i_emb, ii_emb = self.model.forward()
            for batch_idx, batched_data in enumerate(eval_data):
                # predict: interaction without item ids
                user = batched_data[0]
                ui_u_emb, ui_i_emb, ii_emb, _ = self.model.forward(self.denoised_II,self.IIG_norm, 0)
                restore_item_e = ui_i_emb + ii_emb
                scores = torch.matmul(ui_u_emb[user], restore_item_e.transpose(0, 1))
                masked_items = batched_data[1]
                # mask out pos items
                scores[masked_items[0], masked_items[1]] = -1e10
                # rank and get top-k
                _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
                batch_matrix_list.append(topk_index)
            save_path = './plots'
            return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

def calcRegloss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return  ret