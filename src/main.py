from unittest import TestLoader
import config
#from vae import VAE
from vae import VAE
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from customdataloader import dataset
all_previous_loss = []
counter = 0
def earlystopping(devloss):
        global counter 
        global all_previous_loss
        if counter != 3:
            # we will store the value 
            all_previous_loss.append(devloss.item())
            counter +=1
        
        elif counter >=3:
            index = np.argmax(all_previous_loss)
            if index != 0:
                # it means that loss is started increasing 
                return True
            else:
                all_previous_loss.pop(0)
                counter -=1

def loss(output, input, mean, log_var):
    """
        we need to define two type of loss function 
        1. reconstruction loss 
        2. Mean Squared Error Loss
    """
    reconLoss = nn.MSELoss(reduction='none')
    kl = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var), axis=1)
    recon = reconLoss(input, output).view(config.BATCH_SIZE, -1).sum(axis=1)
    loss_val = recon.mean() + kl.mean()
    return loss_val



if __name__ == "__main__":

    traindata = config.loadpickle(config.SAVE_FEATURES_TRAIN)
    testdata  = config.loadpickle(config.SAVE_FEATURES_TEST)

    # now we can send the values in custom data loader
    traindataset = dataset(traindata)
    testdataset  = dataset(testdata)
    # now we can pass the dataset inside the data loader
    traindataloader =  DataLoader(traindataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    testdataloader  =  DataLoader(testdataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    # define the optimizer
    vae = VAE(config.LATENT_SPACE)
    optimizer = torch.optim.SGD(vae.parameters(), lr= config.LR)
    train_loss = []
    test_loss  = []
    for epoch in tqdm(range(config.ITERATION)):
        total_loss = []
        vae.train()
        for data in traindataloader:
            x, _ = data
          
            
            output,  z, mean, log_var = vae(x)
            lossfn = loss(output, x, mean, log_var)
            lossfn.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss.append(lossfn.item())
    
        if epoch % 5 == 0 and epoch !=0:
            vae.eval()
            with torch.no_grad():
                total_dev_loss = []
                for data in testdataloader:
                    x, _ = data
                    output,  z, mean, log_var = vae(x)
                    lossfn = loss(output, x, mean, log_var)
                    total_dev_loss.append(lossfn.item())

            # if earlystopping(np.mean(total_dev_loss)):
            #     break
            train_loss.append(np.mean(total_loss))
            test_loss.append(np.mean(total_dev_loss))
            print(f"Train Loss {np.mean(total_loss)} & Test loss {np.mean(total_dev_loss)}")

    torch.save({
        "model_state_dict": vae.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
    }, config.SAVE_MODEL)