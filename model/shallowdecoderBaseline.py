import torch.nn as nn
import torch
class Shallow_decoder(nn.Module):
    def __init__(self, outputlayer_size, n_sensors):
        super(Shallow_decoder, self).__init__()

        self.n_sensors = n_sensors
        self.outputlayer_size = outputlayer_size

        self.learn_features = nn.Sequential(
            nn.Linear(n_sensors, 55),
            nn.ReLU(True),
            nn.BatchNorm1d(55),
        )

        self.learn_coef = nn.Sequential(
            nn.Linear(55, 65),
            nn.ReLU(True),
            nn.BatchNorm1d(65),
        )


        self.learn_dictionary = nn.Sequential(
            nn.Linear(65, self.outputlayer_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.learn_features(x)
        x = self.learn_coef(x)
        x = self.learn_dictionary(x)
        return x
if __name__ == "__main__":
    device = torch.device("cuda")
    model = shallow_decoder(outputlayer_size=4096,n_sensors=16)
    model.to(device)
    x = torch.randn(5,16).to(device)
    out = model(x)
    print(out.shape)
