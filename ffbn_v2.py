
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_v1 import Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import mat_load
import numpy as np

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

        tmp_input=torch.as_tensor(1+SNR_input/64*torch.pow(torch.abs(hv), 2),dtype=torch.float64)

        rate=torch.log(tmp_input)/ torch.log(torch.tensor(2.0,dtype=torch.float64))
        return -rate

def my_loss(output, target):    #函数的作用，还有output表示什么，前面好像没有出现过呢？
    """
    求模型的损失函数，调用在下面：

    loss = my_loss(outputs, batch[2])
    loss.mean().backward()
    """
    # loss = torch.mean((output - target)**2)
    loss=output

    return loss

if __name__ == "__main__":

    params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 2}
    NUM_EPOCHS = 10
    BEST_MODEL_PATH = 'best_model.pth'    #这里保存的是什么？是模型的参数吗？保存的是模型的参数

    best_accuracy = 0.0
    model=BFNN()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    path = 'train_set/example/train'
    training_set = Dataset(path)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    for epoch in range(NUM_EPOCHS):
        for batch in training_generator:
            optimizer.zero_grad()
            # res=torch.tensor(batch[0], dtype=torch.double)
            # batch[0].dtype=torch.double
            res=batch[0].to(dtype=torch.float32)    #res表示为什么呢？表示的是dataloader里面的H_input
            # print(res)
            outputs = model(res,batch[1].to(dtype=torch.float32) ,batch[2])
            loss = my_loss(outputs, batch[2])
            # loss.requres_grad = True
            loss.mean().backward()
            optimizer.step()
            scheduler.step(loss.mean())
        torch.save(model,BEST_MODEL_PATH)    #这里save的是啥？保存的是best_model.pth


    # -----------------------
    #  Test Your Model
    # -----------------------

    path = 'train_set/example/train'  # the path of the dictionary containing pcsi.mat and ecsi.mat
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
    # model.load_weights('./0db.h5')
    rate = []
    for snr in range(-20, 25, 5):
        SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
        SNR=torch.from_numpy(SNR)
        
        outputs=model(H_input, H, SNR)
        y=outputs.mean()
        rate.append(-y)
    print(rate)


