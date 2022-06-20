from gensim.models.fasttext import FastText
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config


class textRNN(nn.Module):
    def __init__(self, config: config):
        super(textRNN, self).__init__()
        self.vocab_size = config.weight.shape[0]
        self.device = config.device
        self.params = torch.Tensor(config.weight).to(self.device)
        self.batch_size = config.batch_size
        self.embedding_size = config.vector_size
        self.hidden_size = config.hidden_size
        self.get_params(self.embedding_size, self.hidden_size, self.device)
        self.theta = config.theta

    def get_params(self, embedding_size, hidden_size, device):
        """
        获取RNN的参数

        Args:
            embedding_size: 词嵌入的维度
            hidden_size: 隐藏层大小
            device: 训练设备

        Returns:
            参数的元组
        """
        # 输入的维度假设为词嵌入的维度
        num_inputs = embedding_size
        # 输出的维度假设为10
        num_outputs = 10

        # 使用正态分布初始化权重
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.1

        # 隐藏层的参数
        W = normal((num_inputs, hidden_size))
        U = normal((hidden_size, hidden_size))
        b_h = torch.zeros(hidden_size, device=device)

        # 输出层参数
        V = normal((hidden_size, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)

        # 附加梯度
        self.params = (self.params, W, U, b_h, V, b_q)
        for param in self.params:
            param.requires_grad_(True)

    def grad_clipping(self):
        """裁剪梯度"""
        theta = self.theta
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in self.params))
        if norm > theta:
            for param in self.params:
                param.grad[:] *= theta / norm

    def forward(self, inputs):
        weights, W, U, b_h, V, b_q = self.params
        H = [torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)]
        inputs_embedding = F.embedding(inputs, weights)
        inputs = inputs_embedding.permute(1, 0, 2)
        for X in inputs:
            H.append(torch.tanh(torch.mm(X, W) + torch.mm(H[len(H) - 1], U) + b_h))
        Y = torch.mm(H[len(H) - 1], V) + b_q
        H.clear()
        Y = F.softmax(Y, dim=1)
        return Y

    def save(self, path):
        torch.save(self.params, path)

    def load(self, path):
        self.params = torch.load(path)

class textGRU(nn.Module):
    def __init__(self, config: config):
        super(textGRU, self).__init__()
        # 词表大小
        self.vocab_size = config.weight.shape[0]
        # 训练设备
        self.device = config.device
        # 设置embedding参数
        self.params = torch.Tensor(config.weight).to(self.device)
        # batch_size
        self.batch_size = config.batch_size
        # 词嵌入的维度
        self.embedding_size = config.vector_size
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取参数
        self.get_params(self.embedding_size, self.hidden_size, self.device)

    def get_params(self, embedding_size, hidden_size, device):
        num_inputs = embedding_size
        num_outputs = 10
        num_hiddens = hidden_size

        def init_params(shape):
            return nn.init.orthogonal_(torch.randn(size=shape)).to(device)

        # 重置门参数
        W_xr, W_hr, b_r = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.zeros(num_hiddens, device=device))
        # 更新门参数
        W_xz, W_hz, b_z = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.ones(num_hiddens, device=device))
        # 候选记忆元参数
        W_xc, W_hc, b_c = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.zeros(num_hiddens, device=device))
        # 输出层参数
        W_hq = init_params((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        self.params = (self.params, W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xc, W_hc, b_c, W_hq, b_q)
        for param in self.params:
            param.requires_grad_(True)

    def forward(self, inputs):

        H = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)
        weights, W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xc, W_hc, b_c, W_hq, b_q = self.params
        inputs_embedding = F.embedding(inputs, weights)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)
        for X in inputs_embedding:
            r = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
            z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
            h = torch.tanh(torch.mm(X, W_xc) + torch.mm(H * r, W_hc) + b_c)
            H = z * H + (1 - z) * h
        Y = torch.mm(H, W_hq) + b_q
        Y = F.softmax(Y, dim=1)
        return Y
    def save(self, path):
        torch.save(self.params, path)

    def load(self, path):
        self.params = torch.load(path)

class textLSTM(nn.Module):
    def __init__(self, config: config):
        super(textLSTM, self).__init__()
        # 词表大小
        self.vocab_size = config.weight.shape[0]
        # 训练设备
        self.device = config.device
        # 设置embedding参数
        self.params = torch.Tensor(config.weight).to(self.device)
        # batch_size
        self.batch_size = config.batch_size
        # 词嵌入的维度
        self.embedding_size = config.vector_size
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取参数
        self.get_params(self.embedding_size, self.hidden_size, self.device)

    def get_params(self, embedding_size, hidden_size, device):
        num_inputs = embedding_size
        num_outputs = 10
        num_hiddens = hidden_size

        def init_params(shape):
            return nn.init.orthogonal_(torch.randn(size=shape)).to(device)

        # 输入门参数
        W_xi, W_hi, b_i = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.zeros(num_hiddens, device=device))
        # 遗忘门参数
        W_xf, W_hf, b_f = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.ones(num_hiddens, device=device))
        # 输出门参数
        W_xo, W_ho, b_o = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.zeros(num_hiddens, device=device))
        # 候选记忆元参数
        W_xc, W_hc, b_c = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                           torch.zeros(num_hiddens, device=device))
        # 输出层参数
        W_hq = init_params((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        self.params = (self.params, W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q)
        for param in self.params:
            param.requires_grad_(True)

    def forward(self, inputs):
        H = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)
        C = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)

        weights, W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = self.params
        inputs_embedding = F.embedding(inputs, weights)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)
        for X in inputs_embedding:
            I = torch.sigmoid((torch.mm(X, W_xi) + torch.mm(H, W_hi) + b_i))
            F_ = torch.sigmoid((torch.mm(X, W_xf) + torch.mm(H, W_hf) + b_f))
            O = torch.sigmoid((torch.mm(X, W_xo) + torch.mm(H, W_ho) + b_o))
            C_tilda = torch.tanh(torch.mm(X, W_xc) + torch.mm(H, W_hc) + b_c)
            C = F_ * C + I * C_tilda
            H = O * torch.tanh(C)
        Y = torch.mm(H, W_hq) + b_q
        Y = F.softmax(Y, dim=1)
        return Y

    def save(self, path):
        torch.save(self.params, path)

    def load(self, path):
        self.params = torch.load(path)

class textBiLSTM(nn.Module):
    def __init__(self, config: config):
        super(textBiLSTM, self).__init__()
        # 词表大小
        self.vocab_size = config.weight.shape[0]
        # 训练设备
        self.device = config.device
        # 设置embedding参数
        self.params = torch.Tensor(config.weight).to(self.device)
        # batch_size
        self.batch_size = config.batch_size
        # 词嵌入的维度
        self.embedding_size = config.vector_size
        # 隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取参数
        self.get_params(self.embedding_size, self.hidden_size, self.device)

    def get_params(self, embedding_size, hidden_size, device):
        num_inputs = embedding_size
        num_outputs = 10
        num_hiddens = hidden_size

        def init_params(shape):
            return nn.init.orthogonal_(torch.randn(size=shape)).to(device)

        # 正向输入门参数
        W_xi_f, W_hi_f, b_i_f = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.zeros(num_hiddens, device=device))
        # 正向遗忘门参数
        W_xf_f, W_hf_f, b_f_f = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.ones(num_hiddens, device=device))
        # 正向输出门参数
        W_xo_f, W_ho_f, b_o_f = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.zeros(num_hiddens, device=device))
        # 正向候选记忆元参数
        W_xc_f, W_hc_f, b_c_f = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.zeros(num_hiddens, device=device))

        # 反向输入门参数
        W_xi_b, W_hi_b, b_i_b = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.zeros(num_hiddens, device=device))
        # 反向遗忘门参数
        W_xf_b, W_hf_b, b_f_b = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.ones(num_hiddens, device=device))
        # 反向输出门参数
        W_xo_b, W_ho_b, b_o_b = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.zeros(num_hiddens, device=device))
        # 反向候选记忆元参数
        W_xc_b, W_hc_b, b_c_b = (init_params((num_inputs, num_hiddens)), init_params((num_hiddens, num_hiddens)),
                                 torch.zeros(num_hiddens, device=device))

        # 输出层参数
        W_hq = init_params((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        self.params = (
            self.params, W_xi_f, W_hi_f, b_i_f, W_xf_f, W_hf_f, b_f_f, W_xo_f, W_ho_f, b_o_f, W_xc_f, W_hc_f, b_c_f,
            W_xi_b, W_hi_b, b_i_b, W_xf_b, W_hf_b, b_f_b, W_xo_b, W_ho_b, b_o_b, W_xc_b, W_hc_b, b_c_b, W_hq, b_q)
        for param in self.params:
            param.requires_grad_(True)

    def forward(self, inputs):
        H_f = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)
        H_b = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)
        C_f = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)
        C_b = torch.zeros((inputs.shape[0], self.hidden_size), device=self.device)
        weights, W_xi_f, W_hi_f, b_i_f, W_xf_f, W_hf_f, b_f_f, W_xo_f, W_ho_f, b_o_f, W_xc_f, W_hc_f, b_c_f, W_xi_b, W_hi_b, b_i_b, W_xf_b, W_hf_b, b_f_b, W_xo_b, W_ho_b, b_o_b, W_xc_b, W_hc_b, b_c_b, W_hq, b_q = self.params
        inputs_embedding = F.embedding(inputs, weights)
        inputs_embedding = inputs_embedding.permute(1, 0, 2)
        for X in inputs_embedding:
            I = torch.sigmoid((torch.mm(X, W_xi_f) + torch.mm(H_f, W_hi_f) + b_i_f))
            F_ = torch.sigmoid((torch.mm(X, W_xf_f) + torch.mm(H_f, W_hf_f) + b_f_f))
            O = torch.sigmoid((torch.mm(X, W_xo_f) + torch.mm(H_f, W_ho_f) + b_o_f))
            C_tilda_f = torch.tanh(torch.mm(X, W_xc_f) + torch.mm(H_f, W_hc_f) + b_c_f)
            C_f = F_ * C_f + I * C_tilda_f
            H_f = O * torch.tanh(C_f)
        for X in list(reversed(inputs_embedding)):
            I = torch.sigmoid((torch.mm(X, W_xi_b) + torch.mm(H_b, W_hi_b) + b_i_b))
            F_ = torch.sigmoid((torch.mm(X, W_xf_b) + torch.mm(H_b, W_hf_b) + b_f_b))
            O = torch.sigmoid((torch.mm(X, W_xo_b) + torch.mm(H_b, W_ho_b) + b_o_b))
            C_tilda_b = torch.tanh(torch.mm(X, W_xc_b) + torch.mm(H_b, W_hc_b) + b_c_b)
            C_b = F_ * C_b + I * C_tilda_b
            H_b = O * torch.tanh(C_b)
        H = H_f + H_b
        Y = torch.mm(H, W_hq) + b_q
        Y = F.softmax(Y, dim=1)
        return Y

    def save(self, path):
        torch.save(self.params, path)

    def load(self, path):
        self.params = torch.load(path)



class WeatherLSTM(nn.Module):
    def __init__(self, config):
        super(WeatherLSTM, self).__init__()
        self.device = config.device
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        # 正向
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # 输入门
        self.gate_forward_i_W = nn.Linear(self.input_size, self.hidden_size)
        self.gate_forward_i_U = nn.Linear(self.hidden_size, self.hidden_size)
        # 遗忘门
        self.gate_forward_f_W = nn.Linear(self.input_size, self.hidden_size)
        self.gate_forward_f_U = nn.Linear(self.hidden_size, self.hidden_size)
        # 输出门
        self.gate_forward_o_W = nn.Linear(self.input_size, self.hidden_size)
        self.gate_forward_o_U = nn.Linear(self.hidden_size, self.hidden_size)
        # 候选参数
        self.gate_forward_c_W = nn.Linear(self.input_size, self.hidden_size)
        self.gate_forward_c_U = nn.Linear(self.hidden_size, self.hidden_size)

        # 输出参数
        self.outputlayer = nn.Linear(self.hidden_size, 1)

        # 导师训练过程输入参数
        self.relu = nn.ReLU(inplace=False)
        self.input = nn.Linear(1, self.input_size)


    def forward(self, inputs):
        trains = inputs[0]
        h = torch.zeros((trains.shape[0], self.hidden_size), device=self.device)
        c = torch.zeros((trains.shape[0], self.hidden_size), device=self.device)
        trains = trains.permute(1, 0, 2)
        for X in trains:
            it = self.sigmoid(self.gate_forward_i_W(X)+self.gate_forward_i_U(h))
            ft = self.sigmoid(self.gate_forward_f_W(X)+self.gate_forward_f_U(h))
            ot = self.sigmoid(self.gate_forward_o_W(X)+self.gate_forward_o_U(h))
            ct_cilda = self.tanh(self.gate_forward_c_W(X)+self.gate_forward_c_U(h))
            c = ft * c + it * ct_cilda
            h = ot * self.tanh(c)

        y1 = self.outputlayer(h)
        X = inputs[1].permute(1,0)
        for i in range(self.output_size-1):
            x = X[i]
            x = x.view(-1, 1)
            x = self.input(x)
            it = self.sigmoid(self.gate_forward_i_W(x) + self.gate_forward_i_U(h))
            ft = self.sigmoid(self.gate_forward_f_W(x) + self.gate_forward_f_U(h))
            ot = self.sigmoid(self.gate_forward_o_W(x) + self.gate_forward_o_U(h))
            ct_cilda = self.tanh(self.gate_forward_c_W(x) + self.gate_forward_c_U(h))
            c = ft * c + it * ct_cilda
            h = ot * self.tanh(c)
            y = self.outputlayer(h)
            y1 = torch.cat((y1, y), 1)
        return y1



