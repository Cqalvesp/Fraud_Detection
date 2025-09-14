import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Class
class FraudDetectionModel(nn.Module):
    
    def __init__(self, input_attributes=30, hl1=16, hl2=8):
        super().__init__()
        self.connection1 = nn.Linear(input_attributes, hl1)
        self.connection2 = nn.Linear(hl1, hl2)
        self.output = nn.Linear(hl2, 1)

    def forward(self, x):
        x = F.relu(self.connection1(x))
        x = F.relu(self.connection2(x))
        return self.output(x)

    
  


