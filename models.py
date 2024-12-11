import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.neg()  # 反转梯度
        return grad_input
class discrimtor(torch.nn.Module):
    def __init__(self,categolory,lamda=0.8):
        super(discrimtor, self).__init__()
        self.categolory=categolory
        self.mlp = nn.Sequential(nn.Linear(64,32),
                                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                 nn.Linear(32,self.categolory),
                                 )
        self.lamda=lamda

    def forward(self, x):
        output=self.mlp(x)
        return GradientReversalLayer.apply(output) * self.lamda

class domainator(torch.nn.Module):
    def __init__(self,categolory):
        super(domainator, self).__init__()
        self.categolory=categolory
        self.mlp = nn.Sequential(nn.Linear(310,256),
                                 nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                 nn.Linear(256, 128),
                                 nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                 nn.Linear(128, 64),
                                 nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                 nn.LeakyReLU(negative_slope=0.01, inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(64, self.categolory),

                                 )
    def forward(self, x):
        output=self.mlp(x)
        return output


class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


    def forward(self, x):#lkr 83.618
        x = torch.reshape(x, (-1, 62, 5))
        out = self.module(torch.flatten(x, 1))
        return out



class BFE(nn.Module):
    def __init__(self):
        super(BFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-05,affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class MDA(nn.Module):
    def __init__(self, number_of_source=15, number_of_category=4, hybrid_parameter=0.8):
        super(MDA, self).__init__()
        self.CommonNetwork = CFE()
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(number_of_source):
            exec('self.BFE' + str(i) + '= BFE()')
            exec('self.BTC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

        self.cent = ConditionalEntropyLoss()
        self.hybrid_parameter=hybrid_parameter
        self.dis = discrimtor(14)
        self.domainator=domainator(14)
    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        mmd_loss = 0
        if self.training == True:
            if mark==15:
                domain_pred = F.softmax(self.domainator(data_src), dim=1)
                domain_pred_loss = nn.CrossEntropyLoss()(domain_pred, label_src)
                return domain_pred_loss

            domain_label = (torch.full((len(data_src),), mark, dtype=torch.long)).to(self.device)
            ratio = self.hybrid_parameter
            cosine_sim = F.cosine_similarity(data_src.unsqueeze(1), data_tgt.unsqueeze(0), dim=2)
            ab = F.softmax(cosine_sim, dim=1)
            top_indices = ab.argmax(dim=1)
            b_similar = data_tgt[top_indices]
            feature_mix = ratio * data_src + (1 - ratio) * b_similar
            data_src=torch.cat((data_src[:32,:],feature_mix[32:,:]))
            data_src_CFE = self.CommonNetwork(data_src)
            data_tgt_CFE = self.CommonNetwork(data_tgt)
            domain_pre = F.softmax(self.dis(data_src_CFE),dim=1)
            domain_loss = nn.CrossEntropyLoss()(domain_pre, domain_label)
            BFE_name = 'self.BFE' + str(mark)
            data_tgt_BFE = eval(BFE_name)(data_tgt_CFE)
            BFE_name = 'self.BFE' + str(mark)
            data_src_BFE = eval(BFE_name)(data_src_CFE)
            mmd_loss += utils.mmd_linear(data_src_BFE, data_tgt_BFE)
            DSC_name = 'self.BTC' + str(mark)
            pred_src = eval(DSC_name)(data_src_BFE)
            pred_tgt = eval(DSC_name)(data_tgt_BFE)
            ce_loss = self.cent(pred_tgt)
            cls_loss = nn.CrossEntropyLoss()(F.softmax(
                pred_src, dim=1), label_src.squeeze())
            return cls_loss, mmd_loss,domain_loss,ce_loss

        else:
            if mark==1:

                data_CFE = self.CommonNetwork(data_src)
                BFE_name = 'self.BFE' + str(number_of_source)
                DSC_name = 'self.BTC' + str(number_of_source)
                feature_BFE = eval(BFE_name)(data_CFE)
                return eval(DSC_name)(feature_BFE)
            data_CFE = self.CommonNetwork(data_src)
            pred = []
            for i in range(number_of_source):

                BFE_name = 'self.BFE' + str(i)
                DSC_name = 'self.BTC' + str(i)
                feature_BFE_i = eval(BFE_name)(data_CFE)

                pred.append(eval(DSC_name)(feature_BFE_i))

            return pred

class MDA_tsne(nn.Module):
    def __init__(self, number_of_source=15, number_of_category=4, hybrid_parameter=0.8):
        super(MDA_tsne, self).__init__()
        self.CommonNetwork = CFE()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(number_of_source):
            exec('self.BFE' + str(i) + '=BFE()')
            exec('self.BTC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

        self.cent = ConditionalEntropyLoss()
        self.hybrid_parameter = hybrid_parameter
        self.dis = discrimtor(14)
        self.domainator = domainator(14)

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        mmd_loss = 0
        if self.training == True:
            # common feature extractor
            if mark == 15:
                domain_pred = F.softmax(self.domainator(data_src), dim=1)
                domain_pred_loss = nn.CrossEntropyLoss()(domain_pred, label_src)
                return domain_pred_loss

            domain_label = (torch.full((len(data_src),), mark, dtype=torch.long)).to(self.device)
            ratio = self.hybrid_parameter

            cosine_sim = F.cosine_similarity(data_src.unsqueeze(1), data_tgt.unsqueeze(0), dim=2)
            ab = F.softmax(cosine_sim, dim=1)
            top_indices = ab.argmax(dim=1)
            b_similar = data_tgt[top_indices]

            feature_mix = ratio * data_src + (1 - ratio) * b_similar
            data_src = torch.cat((data_src[:32, :], feature_mix[32:, :]))
            data_src_CFE = self.CommonNetwork(data_src)
            data_tgt_CFE = self.CommonNetwork(data_tgt)
            domain_pre = F.softmax(self.dis(data_src_CFE), dim=1)
            domain_loss = nn.CrossEntropyLoss()(domain_pre, domain_label)
            BFE_name = 'self.BFE' + str(mark)
            data_tgt_BFE = eval(BFE_name)(data_tgt_CFE)
            BFE_name = 'self.BFE' + str(mark)
            data_src_BFE = eval(BFE_name)(data_src_CFE)

            mmd_loss += utils.mmd_linear(data_src_BFE, data_tgt_BFE)
            DSC_name = 'self.BTC' + str(mark)  ###lstm+cnn+fc
            pred_src = eval(DSC_name)(data_src_BFE)
            pred_tgt = eval(DSC_name)(data_tgt_BFE)
            ce_loss = self.cent(pred_tgt)

            cls_loss = nn.CrossEntropyLoss()(F.softmax(
                pred_src, dim=1), label_src.squeeze())
            return cls_loss, mmd_loss, domain_loss, ce_loss

        else:
            data_CFE = self.CommonNetwork(data_src)
            pred = []
            feature_BFE = []
            for i in range(number_of_source):
                BFE_name = 'self.BFE' + str(i)
                DSC_name = 'self.BTC' + str(i)
                feature_BFE_i = eval(BFE_name)(data_CFE)
                feature_BFE.append(feature_BFE_i)
                pred.append(eval(DSC_name)(feature_BFE_i))

            return pred, feature_BFE