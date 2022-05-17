from platform import release
from re import M
import torch.nn as nn
import torch

class reshapeModule(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        return x.view(-1, 32, 1020, 59)

class VAE(nn.Module):
    def __init__(self, inputdim, latenvector):
        super().__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=inputdim, out_channels=32,kernel_size=(3,3)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=32),
                nn.Conv2d(in_channels= 32, out_channels = 32, kernel_size=(3,3)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=32),
                nn.Flatten()
            )
            
        self.mu = nn.Linear(in_features=1925760, out_features=latenvector)
        self.log_var= nn.Linear(in_features =1925760 , out_features =latenvector)
        self.normalDistribution = torch.distributions.Normal(0,1)
        
        
        self.decoder = nn.Sequential(

                nn.Linear(in_features = latenvector, out_features = 1925760 ),
                reshapeModule(),
           
                nn.ConvTranspose2d(in_channels = 32, out_channels= 32,\
                                   kernel_size=(3,3)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=32),

                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3,3)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=1)
            )
            
            
    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        normal = self.normalDistribution.sample(sigma.shape)       
        z = mu + normal*sigma
        return z 

    def forward(self, x):
        enc_output = self.encoder(x)
        mu = self.mu(enc_output)
        log_var = self.log_var(enc_output)
        z =  self.reparameterization(mu, log_var)
        dec_output = self.decoder(z)
        return torch.sigmoid(dec_output), mu, log_var



