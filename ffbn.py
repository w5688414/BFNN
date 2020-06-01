
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import Dataset
import torch.nn.functional as F


# def Rate_func(temp):
#     h, v, SNR_input = temp
#     hv = backend.batch_dot(
#         tf.cast(h, tf.complex64), tf.transpose(a=v, perm=[1, 0]))
#     rate = tf.math.log(tf.cast(1 + SNR_input / Nt * tf.pow(tf.abs(hv), 2), tf.float32)) / tf.math.log(2.0)
#     return -rate

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

    params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 2}
    NUM_EPOCHS = 10
    BEST_MODEL_PATH = 'best_model.pth'

    best_accuracy = 0.0
    model=BFNN()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    path = 'train_set/example/train'
    training_set = Dataset(path)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    for epoch in range(NUM_EPOCHS):
        for batch in training_generator:
            optimizer.zero_grad()
            # res=torch.tensor(batch[0], dtype=torch.double)
            # batch[0].dtype=torch.double
            res=batch[0].to(dtype=torch.float32) 
            # print(res)
            outputs = model(res,batch[1].to(dtype=torch.float32) ,batch[2])
            loss = my_loss(outputs, batch[2])
            # loss.requres_grad = True
            loss.mean().backward()
            optimizer.step()
        torch.save(model,BEST_MODEL_PATH)

