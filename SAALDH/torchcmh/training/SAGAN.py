import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
from torch.optim.adamax import Adamax
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models.SAGAN import resnet18, resnet34, get_MS_Text,triplet_loss
from torchcmh.models.SAGAN.spectral import SpectralNorm
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.utils import calc_map_k, calc_neighbor_new, calc_agreement
from torchcmh.dataset import single_data
from torch import autograd
import numpy as np

# @article{SAALDH,
# title = {Self-attention and adversary learning deep hashing network for cross-modal retrieval},
# journal = {Computers & Electrical Engineering},
# volume = {93},
# pages = {107262},
# year = {2021},
# issn = {0045-7906},
# doi = {https://doi.org/10.1016/j.compeleceng.2021.107262},
# url = {https://www.sciencedirect.com/science/article/pii/S0045790621002457},
# author = {Shubai Chen and Song Wu and Li Wang and Zhenyang Yu}


__all__ = ['train']

class SAGAN(TrainBase):
    def __init__(self, data_name: str, img_dir: str, bit: int, img_net, visdom=True, batch_size=64, cuda=True,
                 **kwargs):
        super(SAGAN, self).__init__("SAGAN", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ['cosine_margin_loss','cosine_quantization_loss','graph_loss','loss_adver','loss']
        self.parameters = {'fusion num': 4,  'lambda': 1, 'margin': 0.3, 'alpha': 2 ** np.log2(bit / 32), 'margin_t':2.0}
        self.lr = {'img': 10**(-1.5), 'txt': 10**(-1.5),'gan': 0.001}
        self.max_epoch = 500
        self.lr_decay_freq = 1
        self.lr_decay = (10 ** (-1.5) - 1e-6) / self.max_epoch

        self.num_train = len(self.train_data)
        self.img_model = img_net(bit, self.parameters['fusion num'])
        self.txt_model = get_MS_Text(self.train_data.get_tag_length(), bit, self.parameters['fusion num'])
        ##GAN Module
        self.emd_dim = self.bit
        self.img_discriminator = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, self.emd_dim, kernel_size=(self.emd_dim, 1))),
            nn.ReLU(True),

            SpectralNorm(nn.Conv2d(self.emd_dim, 256, 1)),
            nn.ReLU(True),

            SpectralNorm(nn.Conv2d(256, 1, 1))
        )

        self.txt_discriminator = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, self.emd_dim, kernel_size=(self.emd_dim, 1))),
            nn.ReLU(True),

            SpectralNorm(nn.Conv2d(self.emd_dim, 256, 1)),
            nn.ReLU(True),

            SpectralNorm(nn.Conv2d(256, 1, 1))
        )

        self.train_L = self.train_data.get_all_label()
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.img_discriminator =self.img_discriminator.cuda()
            self.txt_discriminator =self.txt_discriminator.cuda()
            self.train_L = self.train_L.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
        self.B = torch.sign(self.F_buffer + self.G_buffer)

        #GAN_optimizer
        self.optimizer_dis = {
            'img': Adamax(self.img_discriminator.parameters(), lr= self.lr['img'], betas=(0.5, 0.9), weight_decay=0.0001),
            'txt': Adamax(self.txt_discriminator.parameters(), lr= self.lr['img'], betas=(0.5, 0.9), weight_decay=0.0001)
        }
        self.optimizer = torch.optim.SGD([
            {'params': self.img_model.parameters(), 'lr': self.lr['img']},
            {'params': self.txt_model.parameters(),'lr': self.lr['txt']}
        ])
        self._init()

    def dis_img(self, f_x):
        is_img = self.img_discriminator(f_x.unsqueeze(1).unsqueeze(-1))
        return is_img.squeeze()

    def dis_txt(self, f_y):
        is_txt = self.txt_discriminator(f_y.unsqueeze(1).unsqueeze(-1))
        return is_txt.squeeze()


    def train(self, num_works=4):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works,
                                  shuffle=False,
                                  pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            self.train_data.both_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor
                image = data['img']  # type: torch.Tensor
                tag=data['txt']
                if self.cuda:
                    image = image.cuda()
                    tag = tag.cuda()
                    sample_L = sample_L.cuda()

                #output from img_net
                middle_hash_img, hash_img = self.img_model(image)
                # output from img_net
                middle_hash_txt, hash_txt= self.txt_model(tag)

                hash_img_layers = middle_hash_img
                hash_txt_layers = middle_hash_txt
                f_img=hash_img
                f_txt=hash_txt
                hash_img = torch.tanh(hash_img)
                hash_txt = torch.tanh(hash_txt)
                hash_img_layers.append(hash_img)
                hash_txt_layers.append(hash_txt)

                self.G_buffer[ind, :] = hash_txt.data
                self.F_buffer[ind, :] = hash_img.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)
#############################################################GAN####################################################
                #####
                # train img discriminator
                #####

                D_img_real = self.dis_img(f_img.detach())
                D_img_real = -D_img_real.mean()
                self.optimizer_dis['img'].zero_grad()
                D_img_real.backward()

                # train with fake
                D_img_fake = self.dis_img(f_txt.detach())
                D_img_fake = D_img_fake.mean()
                D_img_fake.backward()

                # train with gradient penalty
                alpha = torch.rand(self.batch_size,self.emd_dim).cuda()
                interpolates = alpha * f_img.detach() + (1 - alpha) * f_txt.detach()
                interpolates.requires_grad_()
                disc_interpolates = self.dis_img(interpolates)

                gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                          grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                          create_graph=True,retain_graph=True , only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                # 10 is gradient penalty hyperparameter
                img_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
                img_gradient_penalty.backward()

                loss_D_img = D_img_real - D_img_fake
                self.optimizer_dis['img'].step()

                #####
                # train txt discriminator
                #####
                D_txt_real = self.dis_txt(f_txt.detach())
                D_txt_real = -D_txt_real.mean()
                self.optimizer_dis['txt'].zero_grad()
                D_txt_real.backward()

                # train with fake
                D_txt_fake = self.dis_txt(f_img.detach())
                D_txt_fake = D_txt_fake.mean()
                D_txt_fake.backward()

                # train with gradient penalty
                alpha = torch.rand(self.batch_size, self.emd_dim).cuda()
                interpolates = alpha * f_txt.detach() + (1 - alpha) * f_img.detach()
                interpolates.requires_grad_()
                disc_interpolates = self.dis_txt(interpolates)
                gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                          grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                # 10 is gradient penalty hyperparameter
                txt_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
                txt_gradient_penalty.backward()

                loss_D_txt = D_txt_real - D_txt_fake
                self.optimizer_dis['txt'].step()

                #####
                # train generators
                #####
                # update img network (to generate txt features)
                domain_output = self.dis_txt(f_img)
                loss_G_txt = -domain_output.mean()

                # update txt network (to generate img features)
                domain_output = self.dis_img(f_txt)
                loss_G_img = -domain_output.mean()

                loss_adver = loss_G_txt + loss_G_img
#############################################################GAN####################################################
                cosine_margin_loss,cosine_quantization_loss,intra, variance_loss = self.object_function(f_img,f_txt,hash_img_layers, hash_txt_layers, hash_img, hash_txt,
                                                                                                  sample_L,  G, F, ind)

                loss = cosine_margin_loss+cosine_quantization_loss+variance_loss+loss_adver+intra

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_store['cosine_margin_loss'].update(cosine_margin_loss.item(), (self.batch_size * self.num_train))
                self.loss_store['cosine_quantization_loss'].update(cosine_quantization_loss.item(), (self.batch_size * self.num_train))
                self.loss_store['variance_loss'].update(variance_loss.item(), (self.batch_size * self.num_train))
                self.loss_store['intra'].update(intra.item(), (self.batch_size * self.num_train))
                self.loss_store['loss_adver'].update(loss_adver.item(), (self.batch_size * self.num_train))
                self.loss_store['loss'].update(loss.item())
            self.print_loss(epoch)
            self.plot_loss("loss")
            self.reset_loss()
            weight_img = self.img_model.weight.weight  # type: torch.Tensor
            weight_img = torch.mean(weight_img, dim=1)
            for i in range(weight_img.shape[0]):
                self.plotter.plot("img ms weight", 'part' + str(i), weight_img[i].item())

            weight = self.txt_model.weight.weight  # type: torch.Tensor
            weight = torch.mean(weight, dim=1)
            for i in range(weight.shape[0]):
                self.plotter.plot("txt ms weight", 'part' + str(i), weight[i].item())

            self.B = torch.sign(self.F_buffer + self.G_buffer)
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()


    def object_function(self, f_img,f_txt,hash_layers1, hash_layers2, final_hash1, final_hash2, label,G, F, ind):
        S_inter = calc_neighbor(label, self.train_L)
        inter_loss_img = calc_inter_loss(hash_layers1, S_inter, G,
                                         self.parameters['alpha'])
        inter_loss_txt = calc_inter_loss(hash_layers2, S_inter, F,
                                         self.parameters['alpha'])
        inter_loss = 0.5 * (inter_loss_img + inter_loss_txt)
        # intra_loss of img and txt
        intra_loss_1 = calc_inter_loss(hash_layers1, S_inter, F,
                                       self.parameters['alpha'])
        intra_loss_2 = calc_inter_loss(hash_layers2, S_inter, G,
                                       self.parameters['alpha'])

        intra_loss = (intra_loss_1 + intra_loss_2) * self.parameters['lambda']

        intra = intra_loss + inter_loss
        # S_cos=calc_neighbor_new(label,self.train_L,f_img,f_txt)

        #NOTE:margin is the hyperparameters
        criterion_tri_cos=triplet_loss.TripletHardLoss(dis_metric='cos', reduction='sum')
        loss1 = criterion_tri_cos(final_hash1, label, target=final_hash2, margin=self.parameters['margin'])
        loss2 = criterion_tri_cos(final_hash2, label, target=final_hash1, margin=self.parameters['margin'])
        cosine_margin_loss=loss1+loss2

        theta1 = functional.cosine_similarity(torch.abs(F), torch.ones_like(F).cuda())
        theta2 = functional.cosine_similarity(torch.abs(G), torch.ones_like(G).cuda())
        cosine_quantization_loss = torch.sum(1 / (1 + torch.exp(theta1))) + torch.sum(1 / (1 + torch.exp(theta2)))

        quantization_loss1 = torch.mean(torch.sum(torch.pow(final_hash2 - final_hash1, 2), dim=1))
        graph_loss = quantization_loss1 / self.bit


        return cosine_margin_loss, cosine_quantization_loss,intra, graph_loss


    @staticmethod
    def bit_scalable(img_model, txt_model, qB_img, qB_txt, rB_img, rB_txt, dataset, to_bit=[64, 32, 16]):
        def get_rank(img_net, txt_net):
            from torch.nn import functional as F
            w_img = img_net.weight.weight
            w_txt = txt_net.weight.weight
            w_img = F.softmax(w_img, dim=0)
            w_txt = F.softmax(w_txt, dim=0)
            w = torch.cat([w_img, w_txt], dim=0)
            w = torch.sum(w, dim=0)
            _, ind = torch.sort(w)
            return ind

        hash_length = qB_img.size(1)
        rank_index = get_rank(img_model, txt_model)
        dataset.query()
        query_label = dataset.get_all_label()
        dataset.retrieval()
        retrieval_label = dataset.get_all_label()

        def calc_map(ind):
            qB_img_ind = qB_img[:, ind]
            qB_txt_ind = qB_txt[:, ind]
            rB_img_ind = rB_img[:, ind]
            rB_txt_ind = rB_txt[:, ind]
            mAPi2t = calc_map_k(qB_img_ind, rB_txt_ind, query_label, retrieval_label)
            mAPt2i = calc_map_k(qB_txt_ind, rB_img_ind, query_label, retrieval_label)
            return mAPi2t, mAPt2i

        print("bit scalable from 128 bit:")
        for bit in to_bit:
            if bit >= hash_length:
                continue
            bit_ind = rank_index[hash_length - bit: hash_length]
            mAPi2t, mAPt2i = calc_map(bit_ind)
            print("%3d: i->t %4.4f| t->i %4.4f" % (bit, mAPi2t, mAPt2i))


def train(dataset_name: str, img_dir: str, bit: int, img_net_name='resnet34', visdom=True, batch_size=128, cuda=True, **kwargs):
    img_net = resnet34 if img_net_name == 'resnet34' else resnet18
    trainer = SAGAN(dataset_name, img_dir, bit, img_net, visdom, batch_size, cuda, **kwargs)
    trainer.train()

def calc_inter_loss(hash1_layers, S1, O, alpha):
    inter_loss1 = 0
    for index, hash1_layer in enumerate(hash1_layers):
        theta = 1.0 / alpha * torch.matmul(hash1_layer, O.t())
        logloss = -torch.mean(S1 * theta - torch.log(1 + torch.exp(theta)))
        if torch.isinf(logloss):
            print("the log loss is inf in hash1 of layer %d, with the max of theta is %3.4f" % (index, torch.max(theta).data))
        inter_loss1 += logloss
    return inter_loss1


def calc_contrastive_loss(hash1_layers, hash2_layers, F,G):

    pass




