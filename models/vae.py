"""
    Encoder  class
    Distribution class
    Decoder Class
"""
from syslog import LOG_AUTHPRIV
from turtle import forward
from regex import P
import torch.nn as nn
import torch

class reshapeModule(nn.ModuleList):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.view(-1, 128, 5,1)

class Encoder(nn.ModuleList):
    def __init__(self, latendim) -> None:
        super().__init__()
        # create a sequential model 
     
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, return_indices=True)
        self.batchnorm1 = nn.BatchNorm2d(num_features= 32)
       
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        # defining mu and variance 
        self.mu = nn.Linear(in_features= 640 , out_features= latendim)
        self.var = nn.Linear(in_features= 640 , out_features= latendim)
        # sampling e value from normal distribution 


    def forward(self, x):
        # encoding
        #1st encode
        
        x = self.conv1(x)
        x = self.relu(x)
        x1 = x.size()
        x, indices1 = self.pool1(x)
        x = self.batchnorm1(x)
        
        #2nd encoder
       
        x = self.conv2(x)
        x = self.relu(x)
        x2 = x.size()
        x, indices2 = self.pool1(x)
        x = self.batchnorm2(x)
        
        #3rd encoder
        
        x = self.conv3(x)
        x = self.relu(x)
        x3 = x.size()
        x, indices3 = self.pool2(x)
        x = self.batchnorm3(x)
        #print(x.shape)
        # getting mean & variance 
        x = self.flatten(x)
        mu = self.mu(x)
        var = self.var(x)

        return mu, var, indices1, x1,  indices2, x2, indices3, x3
class representation(nn.ModuleList):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,mu, log_var, norm):
        sigma = torch.exp(log_var/2)
        normalized_dist = norm.sample(sigma.shape)
        z = sigma*normalized_dist + mu
        return z


class Decoder(nn.ModuleList):
    def __init__(self, latenvector):
        super().__init__()
        self.linear = nn.Linear(in_features=latenvector, out_features=640 )
        self.reshape = reshapeModule()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.deconv1_batchnorm = nn.BatchNorm2d(num_features=64)
        self.deconv1_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2,padding=0)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.deconv2_batchnorm = nn.BatchNorm2d(num_features=32)
        self.deconv2_unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.deconv3_batchnorm =  nn.BatchNorm2d(num_features=1)




    def forward(self, z, ind1, input1, ind2, input2, ind3, input3):
        z = self.linear(z)
       
        reshaped_z = self.reshape(z)
       

        # 1st decoder
        x = self.deconv1_unpool(reshaped_z, ind3, output_size= input3)
        x = self.deconv1(x)
        x = self.deconv1_batchnorm(x)
      


        #2nd decoder
        x = self.deconv2_unpool(x, ind2, output_size=input2)
        x = self.deconv2(x)
        x = self.deconv2_batchnorm(x)

        
        #3rd decoder 
     
        x = self.deconv2_unpool(x, ind1, output_size = input1)
        x = self.deconv3(x)
        x = self.deconv3_batchnorm(x)


        return x
       
    


class VAE(nn.ModuleList):
    def __init__(self, latentdim):
        super().__init__()
        self.normal_distribution = torch.distributions.Normal(0,1)
        self.encoder = Encoder(latentdim)
        self.bottleneck = representation()
        self.decoder  = Decoder(latentdim)

    def forward(self, input):
        mu, var, ind1, x1,  ind2, x2, ind3, x3 = self.encoder(input)
        z  = self.bottleneck(mu, var,self.normal_distribution)
        decoder_output  = self.decoder(z, ind1, x1,  ind2, x2,  ind3, x3)
        return torch.sigmoid(decoder_output), z, mu, var 


