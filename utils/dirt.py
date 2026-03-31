import torch
import torchvision.io as io
import torchvision.transforms.v2 as v2
from torchvision.models.resnet import resnet18

class dirty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights='DEFAULT')
        resnet.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

        checkpoint = torch.load("./data-yolo/best-dirty.pth", weights_only=True)
        resnet.load_state_dict(checkpoint["model_state"])
        self.model = resnet
        self.mean=[0.485, 0.456, 0.406] #it's for ImageNet
        self.std=[0.229, 0.224, 0.225]
        self.transforms = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(self.mean, self.std),
                                 v2.Resize((128,128)),
                                ])

    def forward(self,x):
        res = torch.stack(self.transforms(x))
        y = self.model(res).argmax(dim=1)
        return y
