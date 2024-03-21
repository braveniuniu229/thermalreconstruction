import torch.nn as nn
import torch.utils.data
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        linear_layers = []
        for i in range(len(layers) - 2):

            linear_layers.append(nn.Linear(layers[i], layers[i + 1]))
            # linear_layers.append(nn.Dropout(0.1))
            # linear_layers.append(nn.LayerNorm(layers[i + 1]))

            linear_layers.append(nn.Tanh())

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




    def forward(self, x):
        x = self.layers(x)
        return x