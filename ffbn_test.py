
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import Dataset
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from utils import *

class BFNN(nn.Module):
    def __init__(self):
        super(BFNN,self).__init__()
        
        # Inputs to hidden layer linear transformation
        self.batch_norm_v2=nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
        self.batch_norm_v1=nn.BatchNorm1d(128)
        self.batch_norm1=nn.BatchNorm1d(256)
        self.hidden1 = nn.Linear(128, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        

        
    def forward(self, x,perfect_CSI,SNR_input):
        # Pass the input tensor through each of our operations
        x = self.batch_norm_v2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.batch_norm_v1(x)
        # print(x.shape)
        x = F.relu(self.hidden1(x))
        # print(x.shape)
        x = self.batch_norm1(x)
        x = F.relu(self.hidden2(x))
        # print(x.shape)
        x=self.hidden3(x)
        # V_RF
        x1=torch.cos(x)
        x2=torch.sin(x)
        VRF = torch.cat((x1, x2), 1)
        hv=torch.bmm(perfect_CSI.view(-1, 1, 128), VRF.view(-1, 128, 1)) 
        hv=self.flatten(hv)
        print(hv.shape)
        print(SNR_input.shape)

        tmp_input=torch.as_tensor(1+SNR_input/64*torch.pow(torch.abs(hv), 2),dtype=torch.float32)

        rate=torch.log(tmp_input)/ torch.log(torch.tensor(2.0,dtype=torch.float32))
        # torch.log(a)
        # print(VRF.shape)
        # print(perfect_CSI.shape)
        # print(x1.shape)
        return -rate

def my_loss(output, target):
    # loss = torch.mean((output - target)**2)
    loss=output
    return loss

if __name__ == "__main__":

    params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 2}
    NUM_EPOCHS = 10
    BEST_MODEL_PATH = 'best_model.pth'
    best_accuracy = 0.0
    model=BFNN()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    path = 'train_set/example/test'
    training_set = Dataset(path)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    # model.load_state_dict(BEST_MODEL_PATH)
    model=torch.load(BEST_MODEL_PATH)
    model.eval()

    
    path = 'train_set/example/test'  # the path of the dictionary containing pcsi.mat and ecsi.mat
    H, H_est = mat_load(path)
    # use the estimated csi as the input of the BFNN
    H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
    # H denotes the perfect csi
    H = np.squeeze(H)
    H=np.hstack((H.real,H.imag))
    # generate  SNRs associated with different samples
    SNR = np.power(10, np.random.randint(-20, 20, [H.shape[0], 1]) / 10)
    H_input=torch.from_numpy(H_input).to(dtype=torch.float32) 
    H=torch.from_numpy(H).to(dtype=torch.float32) 
    rate = []
    for snr in range(-20, 25, 5):
        SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
        SNR=torch.from_numpy(SNR)
        
        outputs=model(H_input, H, SNR)
        y=outputs.mean()
        rate.append(-y)
    print(rate)

    plt.title("The result of BFNN")
    plt.xlabel("SNR(dB)")
    plt.ylabel("Spectral Efficiency")
    plt.plot(range(-20, 25, 5), rate)
    plt.show()


