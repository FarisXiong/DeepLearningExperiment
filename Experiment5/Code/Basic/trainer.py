from dataset import getDataset
from torch.utils.data import DataLoader
from models import *
from config import config
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torch import autograd


class Trainer():
    def __init__(self, config:config):
        if config.model == 'GAN':
            self.trainer = GANTrainer(config)
        elif config.model == 'WGAN':
            self.trainer = WGANTrainer(config)
        elif config.model == 'WGAN-GP':
            self.trainer = WGANGPTrainer(config)
        else:
            print("Choose GAN/WGAN/WGAN-GP")
            exit(0)
    def train(self):
        self.trainer.train()



class GANTrainer():
    def __init__(self, config:config):
        self.dataset, self.ground_truth = getDataset(config)
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, shuffle=True)
        self.epoch = config.epoch
        self.device = config.device
        self.D_model = BasicDiscriminator().to(self.device)
        self.G_model = BasicGenerator().to(self.device)
        self.Glr = config.Glr
        self.Dlr = config.Dlr
        self.D_optimizer = torch.optim.Adam(self.D_model.parameters(), config.Dlr)
        self.G_optimizer = torch.optim.Adam(self.G_model.parameters(), config.Glr)
        self.model_saved_D = config.model_saved_D
        self.model_saved_G = config.model_saved_G
        self.images_saved = config.images_saved
        self.dk = config.dk
        self.gk = config.gk

    def train(self):
        total_batch = 0
        loss = torch.nn.BCELoss()
        for epoch_index in range(self.epoch):
            print("Epoch [{}/{}]:".format(epoch_index+1, self.epoch))
            for batch_index, Trains in enumerate(self.train_dataloader):
                total_batch += 1

                # 训练 Discriminator 分类器
                origin = Trains['data'].to(self.device)
                D_Loss_sum = float(0)
                G_Loss_sum = float(0)
                ones = torch.normal(mean=1, std=0.05, size=(origin.shape[0], 1), device=self.device)
                zeroes = torch.normal(mean=0, std=0.05, size=(origin.shape[0], 1), device=self.device)

                # 生成随机分布，再使用 Generator 生成数据
                noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                noise = self.G_model(noise_randn)

                # 训练 Discriminator
                for i in range(self.dk):
                    self.D_optimizer.zero_grad()
                    origin_labels = self.D_model(origin)
                    D_Loss = loss(origin_labels, ones)
                    noise_outputs = self.D_model(noise.detach())
                    D_Loss += loss(noise_outputs, zeroes)
                    # 计算 Discriminator 损失函数
                    # outputs = torch.cat((origin_outputs.detach(), noise_outputs), dim=0)
                    # labels = torch.cat((origin_labels.detach(), noise_labels), dim=0)
                    # D_Loss = torch.nn.BCELoss()(outputs.squeeze(), labels)

                    D_Loss.backward()
                    self.D_optimizer.step()
                    D_Loss_sum += D_Loss.item()
                    # self.visual_iter(origin.detach().cpu(), noise_.detach().cpu(), str(total_batch)+'_'+str(i))

                # 训练 Generator 生成器
                for i in range(self.gk):
                    self.G_optimizer.zero_grad()
                    noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                    noise = self.G_model(noise_randn)
                    noise_outputs = self.D_model(noise)
                    G_Loss = loss(noise_outputs, ones)

                    G_Loss.backward()
                    self.G_optimizer.step()
                    G_Loss_sum += G_Loss.item()

                # save model
                torch.save(self.D_model.state_dict(), self.model_saved_D)
                torch.save(self.G_model.state_dict(), self.model_saved_G)
                print("Iter:{:4d} DiscriminatorLoss:{:.8f} GeneratorLoss:{:.8f}".format(total_batch, D_Loss_sum / self.dk,G_Loss_sum / self.gk))
                if total_batch % 10 == 0:
                    # 可视化训练过程
                    self.visual(noise.detach().cpu(), total_batch)

    def border(self):
        """
        画出模型决策边界
        """
        x = np.arange(-1, 2.1, 0.1)
        y = np.arange(-0.5, 2.1, 0.1)
        xx, yy = np.meshgrid(x, y)
        inputs = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(self.device)
        outputs = self.D_model(inputs)
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs.reshape(xx.shape)
        plt.contour(xx, yy, outputs, levels=[0.5], colors=['blue'])

    def visual(self, inputs, index):
        """
        可视化训练过程
        :param: inputs generate生成的数据
        :param: index 图像编号
        """
        inputs = np.transpose(inputs.numpy())
        plt.style.use('ggplot')
        # plt.figure(figsize=(20, 5))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((-1, 2))
        plt.ylim((-0.5, 2))
        plt.scatter(x=self.ground_truth[0], y=self.ground_truth[1], s=5, label='ground truth')
        plt.scatter(x=inputs[0], y=inputs[1], s=5, label='generate')
        self.border()
        plt.legend()
        plt.grid(True)
        # plt.tight_layout()
        plt.savefig(self.images_saved + str(index) + '.png')
        plt.cla()
        # plt.show()






class WGANTrainer():
    def __init__(self, config:config):
        self.dataset, self.ground_truth = getDataset(config)
        self.batch_size = config.batch_size
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, shuffle=True)
        self.epoch = config.epoch
        self.device = config.device
        self.D_model = WGANDiscriminator().to(self.device)
        self.G_model = WGANGenerator().to(self.device)
        self.Glr = config.Glr
        self.Dlr = config.Dlr
        self.D_optimizer = torch.optim.RMSprop(self.D_model.parameters(), config.Dlr)
        self.G_optimizer = torch.optim.RMSprop(self.G_model.parameters(), config.Glr)
        self.model_saved_D = config.model_saved_D
        self.model_saved_G = config.model_saved_G
        self.images_saved = config.images_saved
        self.dk = config.dk
        self.gk = config.gk
        self.clip_value = config.clip_value


    def train(self):
        total_batch = 0
        one = torch.FloatTensor([1]).to(self.device)
        mone = torch.FloatTensor([-1]).to(self.device)

        for epoch_index in range(self.epoch):
            print("Epoch [{}/{}]:".format(epoch_index+1, self.epoch))
            for batch_index, Trains in enumerate(self.train_dataloader):
                for p in self.D_model.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                total_batch += 1

                # 训练 Discriminator 分类器
                origin = Trains['data'].to(self.device)
                D_Loss_sum = float(0)
                G_Loss_sum = float(0)

                # 生成随机分布，再使用 Generator 生成数据
                noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                noise = self.G_model(noise_randn)

                # 训练 Discriminator
                for i in range(self.dk):
                    self.D_optimizer.zero_grad()
                    origin_labels = self.D_model(origin)
                    D_Loss = origin_labels.mean(0)
                    D_Loss.backward(one)
                    D_Loss_sum += D_Loss.item()

                    noise_outputs = self.D_model(noise.detach())
                    D_Loss = noise_outputs.mean(0)
                    D_Loss.backward(mone)
                    D_Loss_sum -= D_Loss.item()

                    self.D_optimizer.step()

                # 训练 Generator 生成器
                for i in range(self.gk):
                    self.G_optimizer.zero_grad()
                    noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                    noise = self.G_model(noise_randn)
                    noise_outputs = self.D_model(noise)
                    G_Loss = noise_outputs.mean(0)
                    G_Loss_sum += G_Loss.item()

                    G_Loss.backward(one)
                    self.G_optimizer.step()
                    G_Loss_sum += G_Loss.item()

                # save model
                torch.save(self.D_model.state_dict(), self.model_saved_D)
                torch.save(self.G_model.state_dict(), self.model_saved_G)
                print("Iter:{:4d} DiscriminatorLoss:{:.8f} GeneratorLoss:{:.8f}".format(total_batch, D_Loss_sum / self.dk, G_Loss_sum / self.gk))

                if total_batch % 10 == 0:
                    # 可视化训练过程
                    self.visual(noise.detach().cpu(), total_batch)

    def border(self):
        """
        画出模型决策边界
        """
        x = np.arange(-1, 2.1, 0.1)
        y = np.arange(-0.5, 2.1, 0.1)
        xx, yy = np.meshgrid(x, y)
        inputs = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(self.device)
        outputs = self.D_model(inputs)
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs.reshape(xx.shape)
        plt.contour(xx, yy, outputs, levels=[0.], colors=['blue'])

    def visual(self, inputs, index):
        """
        可视化训练过程
        :param: inputs generate生成的数据
        :param: index 图像编号
        """
        inputs = np.transpose(inputs.numpy())
        plt.style.use('ggplot')
        # plt.figure(figsize=(20, 5))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((-1, 2))
        plt.ylim((-0.5, 2))
        plt.scatter(x=self.ground_truth[0], y=self.ground_truth[1], s=5, label='ground truth')
        plt.scatter(x=inputs[0], y=inputs[1], s=5, label='generate')
        self.border()
        plt.legend()
        plt.grid(True)
        # plt.tight_layout()
        plt.savefig(self.images_saved + str(index) + '.png')
        plt.cla()
        # plt.show()



class WGANGPTrainer():
    def __init__(self, config:config):
        self.dataset, self.ground_truth = getDataset(config)
        self.batch_size = config.batch_size
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, shuffle=True)
        self.epoch = config.epoch
        self.device = config.device
        self.D_model = WGANDiscriminator().to(self.device)
        self.G_model = WGANGenerator().to(self.device)
        self.Glr = config.Glr
        self.Dlr = config.Dlr
        self.D_optimizer = torch.optim.RMSprop(self.D_model.parameters(), config.Dlr)
        self.G_optimizer = torch.optim.RMSprop(self.G_model.parameters(), config.Glr)
        self.model_saved_D = config.model_saved_D
        self.model_saved_G = config.model_saved_G
        self.images_saved = config.images_saved
        self.dk = config.dk
        self.gk = config.gk
        self.lambda_value = config.lambda_value

    def calculate_gradient_penalty(self, ground_truth, noise):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(ground_truth.size()).to(self.device)
        x = (alpha * ground_truth + (1 - alpha) * noise).to(self.device)
        x_outputs = self.D_model(x)
        gradients = autograd.grad(outputs=x_outputs, inputs=x, grad_outputs=torch.ones(x_outputs.size()).to(self.device), create_graph=True, retain_graph=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_value
        return gradient_penalty

    def train(self):
        total_batch = 0
        one = torch.FloatTensor([1]).to(self.device)
        mone = torch.FloatTensor([-1]).to(self.device)

        for epoch_index in range(self.epoch):
            print("Epoch [{}/{}]:".format(epoch_index+1, self.epoch))
            for batch_index, Trains in enumerate(self.train_dataloader):
                total_batch += 1

                # 训练 Discriminator 分类器
                origin = Trains['data'].to(self.device)
                D_Loss_sum = float(0)
                G_Loss_sum = float(0)

                # 生成随机分布，再使用 Generator 生成数据
                noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                noise = self.G_model(noise_randn)

                # 训练 Discriminator
                for i in range(self.dk):
                    self.D_optimizer.zero_grad()
                    origin_labels = self.D_model(origin)
                    D_Loss = origin_labels.mean(0)
                    D_Loss.backward(one)
                    D_Loss_sum += D_Loss.item()

                    noise_ = torch.autograd.Variable(noise.detach(), requires_grad=True)
                    noise_outputs = self.D_model(noise_)
                    D_Loss = noise_outputs.mean(0)
                    D_Loss.backward(mone)
                    D_Loss_sum -= D_Loss.item()

                    gradient_penalty = self.calculate_gradient_penalty(origin, noise_)
                    gradient_penalty.backward()
                    D_Loss_sum += gradient_penalty.item()
                    self.D_optimizer.step()

                # 训练 Generator 生成器
                for i in range(self.gk):
                    self.G_optimizer.zero_grad()
                    noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                    noise = self.G_model(noise_randn)
                    noise_outputs = self.D_model(noise)
                    G_Loss = noise_outputs.mean(0)
                    G_Loss_sum += G_Loss.item()

                    G_Loss.backward(one)
                    self.G_optimizer.step()
                    G_Loss_sum += G_Loss.item()

                # save model
                torch.save(self.D_model.state_dict(), self.model_saved_D)
                torch.save(self.G_model.state_dict(), self.model_saved_G)
                print("Iter:{:4d} DiscriminatorLoss:{:.8f} GeneratorLoss:{:.8f}".format(total_batch, D_Loss_sum / self.dk, G_Loss_sum / self.gk))

                if total_batch % 10 == 0:
                    # 可视化训练过程
                    self.visual(noise.detach().cpu(), total_batch)

    def border(self):
        """
        画出模型决策边界
        """
        x = np.arange(-1, 2.1, 0.1)
        y = np.arange(-0.5, 2.1, 0.1)
        xx, yy = np.meshgrid(x, y)
        inputs = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(self.device)
        outputs = self.D_model(inputs)
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs.reshape(xx.shape)
        plt.contour(xx, yy, outputs, levels=[0.], colors=['blue'])

    def visual(self, inputs, index):
        """
        可视化训练过程
        :param: inputs generate生成的数据
        :param: index 图像编号
        """
        inputs = np.transpose(inputs.numpy())
        plt.style.use('ggplot')
        # plt.figure(figsize=(20, 5))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((-1, 2))
        plt.ylim((-0.5, 2))
        plt.scatter(x=self.ground_truth[0], y=self.ground_truth[1], s=5, label='ground truth')
        plt.scatter(x=inputs[0], y=inputs[1], s=5, label='generate')
        self.border()
        plt.legend()
        plt.grid(True)
        # plt.tight_layout()
        plt.savefig(self.images_saved + str(index) + '.png')
        plt.cla()
        # plt.show()



class WGANGP_Trainer():
    def __init__(self, config:config):
        self.dataset, self.ground_truth = getDataset(config)
        self.batch_size = config.batch_size
        self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=config.batch_size, shuffle=True)
        self.epoch = config.epoch
        self.device = config.device
        self.D_model = WGANDiscriminator().to(self.device)
        self.G_model = WGANGenerator().to(self.device)
        self.Glr = config.Glr
        self.Dlr = config.Dlr
        self.D_optimizer = torch.optim.Adam(self.D_model.parameters(), config.Dlr)
        self.G_optimizer = torch.optim.Adam(self.G_model.parameters(), config.Glr)
        self.model_saved_D = config.model_saved_D
        self.model_saved_G = config.model_saved_G
        self.images_saved = config.images_saved
        self.dk = config.dk
        self.gk = config.gk
        self.lambda_value = config.lambda_value
        self.clip_value = config.clip_value

    def calculate_gradient_penalty(self, ground_truth, noise):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(ground_truth.size()).to(self.device)
        x = (alpha * ground_truth + (1 - alpha) * noise).to(self.device)
        x_outputs = self.D_model(x)
        gradients = autograd.grad(outputs=x_outputs, inputs=x, grad_outputs=torch.ones(x_outputs.size()).to(self.device), create_graph=True, retain_graph=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_value
        return gradient_penalty

    def train(self):
        total_batch = 0
        one = torch.FloatTensor([1]).to(self.device)
        mone = torch.FloatTensor([-1]).to(self.device)

        for epoch_index in range(self.epoch):
            print("Epoch [{}/{}]:".format(epoch_index+1, self.epoch))
            for batch_index, Trains in enumerate(self.train_dataloader):
                for p in self.D_model.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)
                total_batch += 1

                # 训练 Discriminator 分类器
                origin = Trains['data'].to(self.device)
                D_Loss_sum = float(0)
                G_Loss_sum = float(0)

                # 生成随机分布，再使用 Generator 生成数据
                noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                noise = self.G_model(noise_randn)

                # 训练 Discriminator
                for i in range(self.dk):
                    self.D_optimizer.zero_grad()
                    origin_labels = self.D_model(origin)
                    D_Loss = origin_labels.mean(0)
                    D_Loss.backward(mone)
                    D_Loss_sum -= D_Loss.item()

                    noise_ = torch.autograd.Variable(noise.detach(), requires_grad=True)
                    noise_outputs = self.D_model(noise_)
                    D_Loss = noise_outputs.mean(0)
                    D_Loss.backward(one)
                    D_Loss_sum += D_Loss.item()

                    # gradient_penalty = self.calculate_gradient_penalty(origin, noise_)
                    # gradient_penalty.backward()
                    # D_Loss_sum += gradient_penalty.item()

                    self.D_optimizer.step()

                # 训练 Generator 生成器
                for i in range(self.gk):
                    self.G_optimizer.zero_grad()
                    noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                    noise = self.G_model(noise_randn)
                    noise_outputs = self.D_model(noise)
                    G_Loss = noise_outputs.mean(0)
                    G_Loss_sum += G_Loss.item()

                    G_Loss.backward(mone)
                    self.G_optimizer.step()
                    G_Loss_sum += G_Loss.item()

                # save model
                torch.save(self.D_model.state_dict(), self.model_saved_D)
                torch.save(self.G_model.state_dict(), self.model_saved_G)
                print("Iter:{:4d} DiscriminatorLoss:{:.8f} GeneratorLoss:{:.8f}".format(total_batch, D_Loss_sum / self.dk, G_Loss_sum / self.gk))

                if total_batch % 10 == 0:
                    # 可视化训练过程
                    self.visual(noise.detach().cpu(), total_batch)



    def train_(self):
        total_batch = 0
        one = torch.FloatTensor([1]).to(self.device)
        mone = torch.FloatTensor([-1]).to(self.device)

        for epoch_index in range(self.epoch):
            print("Epoch [{}/{}]:".format(epoch_index+1, self.epoch))
            for batch_index, Trains in enumerate(self.train_dataloader):
                total_batch += 1

                # 训练 Discriminator 分类器
                origin = Trains['data'].to(self.device)
                D_Loss_sum = float(0)
                G_Loss_sum = float(0)

                # 生成随机分布，再使用 Generator 生成数据
                noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                noise = self.G_model(noise_randn)

                # 训练 Discriminator
                for i in range(self.dk):
                    self.D_optimizer.zero_grad()
                    origin_labels = self.D_model(origin)
                    D_Loss_origin = origin_labels.mean(0)
                    D_Loss_origin.backward(one)

                    noise_ = torch.autograd.Variable(noise.detach(), requires_grad=True)
                    noise_outputs = self.D_model(noise_)
                    D_Loss_noise = noise_outputs.mean(0)
                    D_Loss_noise.backward(mone)

                    gradient_penalty = self.calculate_gradient_penalty(origin, noise_)
                    gradient_penalty.backward()

                    D_Loss = D_Loss_noise - D_Loss_origin + gradient_penalty
                    D_Loss_sum += D_Loss.item()

                    self.D_optimizer.step()

                # 训练 Generator 生成器
                for i in range(self.gk):
                    self.G_optimizer.zero_grad()
                    noise_randn = torch.randn(origin.shape[0], 2).to(self.device)
                    noise = self.G_model(noise_randn)
                    noise_outputs = self.D_model(noise)
                    G_Loss = noise_outputs.mean(0)
                    G_Loss.backward(one)
                    self.G_optimizer.step()
                    G_Loss_sum += G_Loss.item()

                # save model
                torch.save(self.D_model.state_dict(), self.model_saved_D)
                torch.save(self.G_model.state_dict(), self.model_saved_G)
                print("Iter:{:4d} DiscriminatorLoss:{:.8f} GeneratorLoss:{:.8f}".format(total_batch, D_Loss_sum / self.dk, G_Loss_sum / self.gk))
                if total_batch % 10 == 0:
                    # 可视化训练过程
                    self.visual(noise.detach().cpu(), total_batch)

    def border(self):
        """
        画出模型决策边界
        """
        x = np.arange(-1, 2.1, 0.1)
        y = np.arange(-0.5, 2.1, 0.1)
        xx, yy = np.meshgrid(x, y)
        inputs = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(self.device)
        outputs = self.D_model(inputs)
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs.reshape(xx.shape)
        plt.contour(xx, yy, outputs, levels=[0], colors=['blue'])

    def visual(self, inputs, index):
        """
        可视化训练过程
        :param: inputs generate生成的数据
        :param: index 图像编号
        """
        inputs = np.transpose(inputs.numpy())
        plt.style.use('ggplot')
        # plt.figure(figsize=(20, 5))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((-1, 2))
        plt.ylim((-0.5, 2))
        plt.scatter(x=self.ground_truth[0], y=self.ground_truth[1], s=5, label='ground truth')
        plt.scatter(x=inputs[0], y=inputs[1], s=5, label='generate')
        self.border()
        plt.legend()
        plt.grid(True)
        # plt.tight_layout()
        plt.savefig(self.images_saved + str(index) + '.png')
        plt.cla()
        # plt.show()