import torch
import sys
import models
from models import *
model = Simpleconv5(nclass=20,inplanes=32,kernel=5)

checkpoint = torch.load(sys.argv[1])
model.load_state_dict(checkpoint['state_dict'])
torch.save(model,'model.pth')
