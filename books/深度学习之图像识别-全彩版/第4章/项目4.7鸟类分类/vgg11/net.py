#coding:utf8
import torch
import torchvision

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class VGG11(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = torchvision.models.vgg11(pretrained=True)
        print(self.model)
        num_ftrs = self.model.classifier._modules["6"].in_features
        self.model.classifier._modules["6"] = torch.nn.Linear(num_ftrs, 200)

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        #print(X.shape)
        X = self.model(X)
        assert X.size() == (N, 200)
        return X

if __name__ == '__main__':
    model = VGG11()
    model.eval()
    input = torch.randn((1, 3, 224, 224))
    output = model(input)
    print(output.shape)
