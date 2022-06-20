import torch.nn as nn

class BasicDiscriminator(nn.Module):
    def __init__(self):
        super(BasicDiscriminator, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc3(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc4(outputs)
        outputs = self.sigmoid(outputs)
        return outputs

class BasicGenerator(nn.Module):
    def __init__(self):
        super(BasicGenerator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.generate = nn.Sequential(
            *block(2, 256, False),
            *block(256, 256),
            *block(256, 256),
            nn.Linear(256, 2),
            # nn.Tanh()
        )

    def forward(self, inputs):
        outputs = self.generate(inputs)
        return outputs




class WGANDiscriminator(nn.Module):
    def __init__(self):
        super(WGANDiscriminator, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc3(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc4(outputs)
        outputs = self.tanh(outputs)
        return outputs

class WGANGenerator(nn.Module):
    def __init__(self):
        super(WGANGenerator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.generate = nn.Sequential(
            *block(2, 256, False),
            *block(256, 256),
            *block(256, 256),
            nn.Linear(256, 2),
            # nn.Tanh()
        )

    def forward(self, inputs):
        outputs = self.generate(inputs)
        return outputs



class WGANGPDiscriminator(nn.Module):
    def __init__(self):
        super(WGANGPDiscriminator, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc3(outputs)
        outputs = self.relu(outputs)
        outputs = self.fc4(outputs)
        outputs = self.tanh(outputs)
        return outputs






