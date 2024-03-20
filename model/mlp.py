import torch.nn as nn
import torch.utils.data


class MultiHeadMLP(nn.Module):
    def __init__(self, layers):
        super(MultiHeadMLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(layers[0], layers[1]), nn.GELU())
        self.layer2 = nn.Sequential(nn.Linear(layers[1], layers[2]), nn.GELU())
        self.layer3 = nn.Sequential(nn.Linear(layers[2], layers[3]), nn.GELU())
        self.layers_final1 = nn.Linear(layers[3], layers[4])
        self.layers_final2 = nn.Linear(layers[3], layers[4])
        # self.layers_final3 = nn.Linear(layers[3], layers[4])

    def forward(self, x, noise):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if noise:
            y = self.layers_final1(x)
        else:
            y = self.layers_final2(x)
        return y


class MT_MLP(nn.Module):
    def __init__(self, layers):
        super(MT_MLP, self).__init__()
        # student network
        self.sub_net1 = MLP(layers)
        # teacher network
        self.sub_net2 = MLP(layers)
        # detach the teacher model
        for param in self.sub_net2.parameters():
            param.detach_()

    def forward(self, x, cur_iter, step):
        if not self.training:
            s_out = self.sub_net1(x)
            return s_out

        # copy the parameters from teacher to student
        if cur_iter == 1:
            for t_param, s_param in zip(self.sub_net2.parameters(), self.sub_net1.parameters()):
                t_param.data.copy_(s_param.data)

        s_out = self.sub_net1(x)
        with torch.no_grad():
            t_out = self.sub_net2(x)
        if step == 1:
            self._update_ema_variables(ema_decay=0.99)

        return s_out, t_out

    def _update_ema_variables(self, ema_decay=0.99):
        for t_param, s_param in zip(self.sub_net2.parameters(), self.sub_net1.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=(1 - ema_decay))


class MultiMLP(nn.Module):
    def __init__(self, layers):
        super(MultiMLP, self).__init__()
        self.sub_net1 = MLP(layers)
        self.sub_net2 = MLP(layers)
        self.sub_net3 = MLP(layers)
        self.sub_net4 = MLP(layers)
        self.sub_net5 = MLP(layers)

    def forward(self, x):
        y1 = self.sub_net1(x)
        y2 = self.sub_net2(x)
        y3 = self.sub_net3(x)
        # y4 = self.sub_net4(x)
        # y5 = self.sub_net5(x)
        return y1, y2, y3


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):

            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            linear_layers.append(nn.Dropout(0.1))
            # linear_layers.append(nn.LayerNorm(layers[i + 1]))
            linear_layers.append(nn.Dropout(0.1))
            linear_layers.append(nn.GELU())

        linear_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*linear_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1
        # 在本情况中，所有元素都被丢弃
        if dropout == 1:
            return torch.zeros_like(X)
        # 在本情况中，所有元素都被保留
        if dropout == 0:
            return X
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout)


    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    from thop import profile
    from thop import clever_format
    from ptflops import get_model_complexity_info

    model = MLP([16, 128, 1280, 4800, 40000])
    print(model)
    x = torch.randn(1, 1, 20)
    flops, params = profile(model, (x,))
    flops, params = clever_format([flops, params], '%.3f')
    print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    flops, params = get_model_complexity_info(model, (1, 20), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)

    print(model)
    xyz = torch.rand(4, 20)
    y = model(xyz)
    for param in model.parameters():
        print(param.dtype)
