import torch
import numpy as np
from utils import mat_load
from pytorch_complex_tensor import ComplexTensor

class Dataset(torch.utils.data.Dataset):

    def __init__(self,path):
        H, H_est = mat_load(path)
        H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
        H = np.squeeze(H)
        SNR = np.power(10, np.random.randint(-20, 20, [H.shape[0], 1]) / 10)
        res=np.hstack((H.real,H.imag))
        self.labels = res
        self.list_X = [H_input,res, SNR]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):

        # X = [self.list_X[0][index],self.list_X[1][index],self.list_X[2][index]]
        y = self.labels[index]
        # return X, y
        return self.list_X[0][index],self.list_X[1][index],self.list_X[2][index],y

def collate_fn(batch):
    # batch.sort(key=lambda x: len(x[1]), reverse=True)
    # img, label = zip(*batch)
    # pad_label = []
    # lens = []
    # max_len = len(label[0])
    # for i in range(len(label)):
    #     temp_label = [0] * max_len
    #     temp_label[:len(label[i])] = label[i]
    #     pad_label.append(temp_label)
    #     lens.append(len(label[i]))
    print(len(batch))
    # print(len(batch[0]))
    # print(len(batch[1]))
    batch_x=[]
    batch_y=[]
    for i in range(len(batch)):
        x=batch[i][0]
        label=batch[i][1]
        batch_x.append(x)
        batch_y.append(label)

    # batch_x = ComplexTensor(batch_x)
    batch_y=torch.FloatTensor(batch_y)
    # print(batch_x.shape)
    print(batch_y.shape)
    return batch_x,batch_y

if __name__ == "__main__":
    params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 2}
    path = 'train_set/example/train'
    training_set = Dataset(path)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    for batch in training_generator:
        print(len(batch))
        # print(batch)
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        # print(batch[1])
        # print(batch[3])
        # print(len(local_batch))
        # print(len(local_labels))
        # print(local_batch[0].shape)
        # print(local_labels[0].shape)
    # path = 'train_set/example/train'
    # H, H_est = mat_load(path)
    # H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
    # H = np.squeeze(H)
    # SNR = np.power(10, np.random.randint(-20, 20, [H.shape[0], 1]) / 10)