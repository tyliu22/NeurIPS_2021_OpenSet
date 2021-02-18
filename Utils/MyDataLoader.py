import torch
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader

import numpy as np

class subDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.IntTensor(self.Label[index])
        return data, label

def main():
    source_data = np.random.rand(10,20)
    source_label = np.random.randint(0,2,(10,1))

    torch_data = subDataset(source_data, source_label)
    # create DataLoader iterator
    # create DataLoader，batch_size = 2，shuffle=False，num_workers= 4：
    train_dataloader = DataLoader.DataLoader(torch_data, batch_size=5, shuffle=False, num_workers=1)

    data, label = iter(train_dataloader).next()
    print('data shape', data.shape)
    print('label shape', label.shape)
    #
    for i, item in enumerate(train_dataloader, 0):
        data, label = item
        print(i)
    print('End')

if __name__ == "__main__":
    main()