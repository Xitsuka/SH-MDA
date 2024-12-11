import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import math
from torch.utils.tensorboard import SummaryWriter
import utils
import models
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(742)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MDA():
    def __init__(self, model=models.MDA(), source_loaders=0, target_loader=0, batch_size=64, iteration=10000, lr=0.001,
                    ce_loss_parameter=0.1,
                    dis_loss_parameter=0.1,
                    top_k=8):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.ce_loss_parameter=ce_loss_parameter
        self.dis_loss_parameter=dis_loss_parameter
        self.top_k=top_k
    def __getModel__(self):
        return self.best_model_wts

    def train(self):
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        correct = 0
        LEARNING_RATE = self.lr
        progress_bar = tqdm(range(1,self.iteration + 1), desc="Training")
        for i in progress_bar:
            self.model.train()
            CommonNetwork_params = [p for n, p in self.model.named_parameters() if 'CommonNetwork' in n]
            BFE_params = [p for n, p in self.model.named_parameters() if 'BFE' in n]
            BTC_params = [p for n, p in self.model.named_parameters() if 'BTC' in n]
            dis_params = [p for n, p in self.model.named_parameters() if 'dis' in n]
            domainator_params = [p for n, p in self.model.named_parameters() if 'domainator' in n]

            optimizer = torch.optim.Adam([
                {'params': CommonNetwork_params, 'lr': LEARNING_RATE/10},
                {'params': BFE_params, 'lr': LEARNING_RATE},
                {'params': BTC_params, 'lr': LEARNING_RATE},
                {'params': dis_params, 'lr': LEARNING_RATE},
            ])
            optimizer_d=torch.optim.Adam([{'params': domainator_params, 'lr': LEARNING_RATE/10}])
            domain_data=[]
            domain_label=[]
            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(
                    device), source_label.to(device)
                target_data = target_data.to(device)
                domain_data.append(source_data)
                domain_label.append((torch.full((len(source_data),), j, dtype=torch.long)).to(device))
                optimizer.zero_grad()
                cls_loss, mmd_loss,dis_loss,ce_loss= self.model(source_data, number_of_source=len(
                    source_iters), data_tgt=target_data, label_src=source_label, mark=j)
                gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
                loss = cls_loss+gamma * mmd_loss+self.ce_loss_parameter*ce_loss+self.dis_loss_parameter*dis_loss#(3,7,11,12,13)
                loss.backward()
                optimizer.step()
            domain_data=torch.cat(domain_data, dim=0)
            domain_label=torch.cat(domain_label, dim=0)
            optimizer_d.zero_grad()
            domain_loss=self.model(data_src=domain_data,label_src=domain_label,mark=15,number_of_source=0)
            domain_loss.backward()
            optimizer_d.step()
            t_correct = self.test()
            if t_correct > correct:
                correct = t_correct
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
            accuracy = round(100. * correct.item() / len(self.target_loader.dataset), 2)
            progress_bar.set_postfix({'Accuracy': f'{accuracy}%'})
        return 100. * correct / len(self.target_loader.dataset)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)

        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders))
                domain_pred = torch.round(F.softmax(self.model.domainator(data)/10, dim=1)*100)/100
                topk_values, topk_indices = torch.topk(domain_pred, self.top_k, dim=1)

                result = torch.zeros_like(domain_pred)
                #
                for idx in range(domain_pred.size(0)):
                    result[idx, topk_indices[idx]] = topk_values[idx]

                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)*result[:, i].unsqueeze(1)
                pred = sum(preds)/len(preds)
                test_loss += nn.CrossEntropyLoss()(F.softmax(pred,
                                        dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                target=target.data.squeeze()
                correct += pred.eq(target).cpu().sum()
        return correct

def cross_session(data, label, session_id, subject_id, category_number, batch_size, iteration, lr,hybrid_parameter,ce_loss_parameter,dis_loss_parameter):
    train_idxs = list(range(3))
    del train_idxs[session_id]
    test_idx = session_id

    target_data, target_label = copy.deepcopy(data[test_idx][subject_id]), copy.deepcopy(label[test_idx][subject_id])
    source_data, source_label = copy.deepcopy(data[train_idxs][:, subject_id]), copy.deepcopy(label[train_idxs][:, subject_id])

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MDA(model=models.MDA( number_of_source=len(source_loaders), number_of_category=category_number,hybrid_parameter=hybrid_parameter),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    ce_loss_parameter = ce_loss_parameter,
                    dis_loss_parameter = dis_loss_parameter
    )
    acc = model.train()
    # torch.save(model.__getModel__(),
    #            'H:/model_save/seediv-cross-session/session{}_subject{}.pth'.format(session_id+1, subject_id + 1))
    print('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))

    return acc




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SH-MDA parameters')
    parser.add_argument('--dataset', type=str, default='seed4',
                        help='the dataset used for SH-MDA, "seed3" or "seed4"')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--session_id_main', type=int, default=0,
                        help='training session data, integer,0,1,2')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--hybrid_parameter', type=int, default=0.6,
                        help='hybrid parameter, 0.8 for the cross-subject 0.6 for the cross-session')
    parser.add_argument('--ce_loss_parameter', type=int, default=0.1,
                        help='ce_loss_parameter')
    parser.add_argument('--dis_loss_parameter', type=int, default=0.1,
                        help='dis_loss parameter')
    parser.add_argument('--top_k', type=int, default=2,
                        help='top_k')

    args = parser.parse_args()
    dataset_name = args.dataset
    session_id_main=args.session_id_main
    hybrid_parameter=args.hybrid_parameter
    ce_loss_parameter = args.ce_loss_parameter
    dis_loss_parameter = args.dis_loss_parameter
    top_k=args.top_k

    print('Model name: SH-MDA. Dataset name: ', dataset_name)
    data, label = utils.load_data(dataset_name)
    data_tmp = copy.deepcopy(data)
    label_tmp = copy.deepcopy(label)
    for i in range(len(data_tmp)):
        for j in range(len(data_tmp[0])):
            data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    trial_total, category_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)


    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch*3394/batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch*820/batch_size)
    print('Iteration: {}'.format(iteration))

    csesn = []
    for subject_id_main in range(15):
           csesn.append(cross_session(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
                                   batch_size, iteration, lr,hybrid_parameter,ce_loss_parameter,dis_loss_parameter))
    print("Cross-session: ", csesn)
    print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))


