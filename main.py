import pickle, time, os
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from param import args
from DataHander import DataHandler
from models.model import SDNet, GCNModel

from utils import save_model, fix_random_seed_as
from tqdm import tqdm

from models import diffusion_process as dp
from Utils.Utils import *
import logging
import sys

from sklearn.utils.extmath import randomized_svd
import scipy.sparse

class Coach():
    def __init__(self, handler):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.handler = handler
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user, self.n_item = self.handler.n_user, self.handler.n_item
        self.uiGraph = self.handler.ui_graph.to(self.device)
        self.uuGraph = self.handler.uu_graph.to(self.device)
        self.GCNModel = GCNModel(args, self.n_user, self.n_item).to(self.device)

        ### Build Diffusion process###
        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.DiffProcess = dp.DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
                                               args.steps, self.device).to(self.device)

        self.optimizer1 = torch.optim.Adam([
            {'params': self.GCNModel.parameters(), 'weight_decay': 0},
        ], lr=args.lr)
        self.optimizer2 = torch.optim.Adam([
            {'params': self.SDNet.parameters(), 'weight_decay': 0},
        ], lr=args.difflr)

        self.scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optimizer1,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(
            self.optimizer2,
            step_size=args.decay_step,
            gamma=args.decay
        )

        self.train_loss = []
        self.his_recall = []
        self.his_ndcg = []

    def sparse_svd(self,sparse_tensor):
        print("svding...")
        U, _, _ = randomized_svd(sparse_tensor, n_components=args.n_hid, n_iter=args.svdlayer, random_state=None)
        print("svd completely")
        U = torch.from_numpy(U)
        U = U.to(self.device)
        U = U.to(torch.float32)
        return U

    def build_user_common_item_matrix(self, user_item_coo):
        user_item_csr = user_item_coo.tocsr()
        user_common = user_item_csr @ user_item_csr.T
        user_common.setdiag(0)
        user_common = user_common.tocoo()
        return user_common

    def train(self):
        args = self.args
        self.save_history = True
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.dataset + '/'
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        best_recall, best_ndcg, best_epoch, wait = 0, 0, 0, 0
        uuuEmbeds = self.sparse_svd(self.handler.dataset['trust'])
        uciEmbeds_train = self.build_user_common_item_matrix(self.handler.dataset['train'])
        uciEmbeds_train = self.sparse_svd(uciEmbeds_train)
        uciEmbeds_test = self.build_user_common_item_matrix(self.handler.dataset['test'])
        uciEmbeds_test = self.sparse_svd(uciEmbeds_test)

        train_times = []
        test_times = []
        start_time = time.time()

        for self.epoch in range(1, args.n_epoch + 1):
            epoch_losses,train_time_elapsed = self.train_one_epoch(uuuEmbeds,uciEmbeds_train)
            train_times.append(train_time_elapsed)
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses
            ), flush=True)
            if self.epoch % 5 == 0:
                recall, ndcg ,test_time_elapsed = self.test(self.testloader, uuuEmbeds,uciEmbeds_train)
                test_times.append(test_time_elapsed)
                # Record the history of recall and ndcg
                self.his_recall.append(recall)
                self.his_ndcg.append(ndcg)
                cur_best = recall + ndcg > best_recall + best_ndcg
                if cur_best:
                    best_recall, best_ndcg, best_epoch = recall, ndcg, self.epoch
                    wait = 0
                else:
                    wait += 1
                logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, args.topk, recall, args.topk, ndcg))
                if args.model_dir and cur_best:
                    desc = args.save_name
                    perf = ''  # f'N/R_{ndcg:.4f}/{hr:.4f}'
                    fname = f'{args.desc}_{desc}_{perf}.pth'

                    save_model(self.GCNModel, self.SDNet, os.path.join(args.model_dir, fname),
                               self.optimizer1, self.optimizer2)
            if self.save_history:
                self.saveHistory()

            if wait >= args.patience:
                print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                break
        avg_train_time = sum(train_times) / len(train_times)
        avg_test_time = sum(test_times) / len(test_times) if test_times else 0
        print(f'Best  Recall@{args.topk} {best_recall:.6f}, NDCG@{args.topk} {best_ndcg:.6f},', flush=True)
        logger.info(f'Average training time per epoch: {avg_train_time:.2f}s')
        logger.info(f'Average testing time per epoch: {avg_test_time:.2f}s')

    def train_one_epoch(self,uuuEmbeds,uciEmbeds_train):
        self.SDNet.train()
        self.GCNModel.train()

        dataloader = self.train_loader
        epoch_losses = [0] * 3
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)
        since = time.time()

        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long().cuda()
            pos_idx = pos_idx.long().cuda()
            neg_idx = neg_idx.long().cuda()
            uiEmbeds = self.GCNModel.forward(self.uiGraph, True)

            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]
            user = uEmbeds[user_idx]
            pos = iEmbeds[pos_idx]
            neg = iEmbeds[neg_idx]

            uu_terms = self.DiffProcess.caculate_losses(self.SDNet, uuuEmbeds[user_idx], uciEmbeds_train,user_idx,args.reweight)
            uuelbo = uu_terms["loss"].mean()
            user = user + uu_terms["pred_xstart"]
            diffloss = uuelbo
            scoreDiff = pairPredict(user, pos, neg)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch_size
            regLoss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(
                neg) ** 2) * args.reg) / args.batch_size
            loss = bprLoss + regLoss
            losses = [bprLoss.item(), regLoss.item()]

            loss = diffloss + loss
            losses.append(diffloss.item())

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer1.step()
            self.optimizer2.step()

            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]
        if self.scheduler1 is not None:
            self.scheduler1.step()
            self.scheduler2.step()

        epoch_losses = [sum(epoch_losses)] + epoch_losses
        time_elapsed = time.time() - since
        print('Training complete in {:.4f}s'.format(
            time_elapsed))
        return epoch_losses,time_elapsed

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def test(self, dataloader,uuuEmbeds,uciEmbeds_test):
        self.SDNet.eval()
        self.GCNModel.eval()
        Recall, NDCG = [0] * 2
        num = dataloader.dataset.__len__()
        since = time.time()
        with torch.no_grad():
            uiEmbeds = self.GCNModel.forward(self.uiGraph, True)
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trnMask = batch
                user_idx = user_idx.long().to(self.device)
                uEmbeds = uiEmbeds[:self.n_user]
                iEmbeds = uiEmbeds[self.n_user:]
                user = uEmbeds[user_idx]
                # user_predict = self.DiffProcess.p_sample(self.SDNet, uuuEmbeds[user_idx], args.sampling_steps, user_idx,
                #                                          uciEmbeds_test,args.sampling_noise)
                user_predict = self.DiffProcess.p_sample(self.SDNet, uuuEmbeds[user_idx], args.sampling_steps, user_idx,
                                                         uciEmbeds_test, args.sampling_noise)
                user = user + user_predict
                trnMask = trnMask.to(self.device)
                allPreds = t.mm(user, t.transpose(iEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                _, topLocs = t.topk(allPreds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall += recall
                NDCG += ndcg

            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(time_elapsed))
            Recall = Recall / num
            NDCG = NDCG / num

        return Recall, NDCG,time_elapsed

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['Recall'] = self.his_recall
        history['NDCG'] = self.his_ndcg
        ModelName = "SDR"
        desc = args.save_name
        perf = ''  # f'N/R_{ndcg:.4f}/{hr:.4f}'
        fname = f'{args.desc}_{desc}_{perf}.his'

        with open('./History/' + args.dataset + '/' + fname, 'wb') as fs:
            pickle.dump(history, fs)


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    fix_random_seed_as(args.seed)

    handler = DataHandler()
    handler.LoadData()
    app = Coach(handler)
    app.train()
    print(args)

