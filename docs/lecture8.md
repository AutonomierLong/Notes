# Lecture 8: Deep Learning Software

## CPU vs GPU

> In deep learning, NVIDIA is dominant, we dont't consider AMD.

???notes "Comparison between CPU and GPU"
    ![linear](./images/Lec08/1%20(2).png){: width="600px" .center}
    

## Deep Learning Frameworks

> 2017年的课程略有些过时了, 这里我主要记录一下Pytorch.

The point of deep learning frameworks

+ Easily build big computational graphs.
+ Easily compute gradients in computational graphs.
+ Run it all efficiently on GPU.

In PyTorch, we define `Variable` to start building a computational graph.

![linear](./images/Lec08/1%20(3).png){: width="600px" .center}

> Calling `c.backward()` computes all gradients.

> Run on GPU by casting to `.cuda()`, just like: `x = variable(torch.randn(N, D).cuda(), requires_grad=True)`.

### PyTorch

Three levels of abstraction:

1. **Tensor**: Imperative ndarray(命令式编程范式下使用的多维数组), but runs on GPU.

    ![linear](./images/Lec08/1%20(4).png){: width="600px" .center}

    > To run on GPU, just cast tensors to a cuda datatype! `dtype = torch.cuda.FloatTensor`.

2. **Variable**: Node in a computational graph, stores data and gradient.

    ![linear](./images/Lec08/1%20(5).png){: width="600px" .center}

    + New Autograd Functions
        <br> You can define your own autograd functions by writing forward and backward for Tensors.
        ![linear](./images/Lec08/1%20(7).png){: width="600px" .center}

3. **Module**: A neural network layer, may store state or learnable weights.

    ![linear](./images/Lec08/1%20(8).png){: width="600px" .center}

    + We can also use an optimizer for different update rules:
    `optimizer = torch.optim.Adam(model.prameters(), lr = learning_rate)`
    ![linear](./images/Lec08/1%20(9).png){: width="600px" .center}
    最后需要使用`optimizer.step()`对参数进行更新.
    
    + Define New Modules
        ![linear](./images/Lec08/1%20(10).png){: width="600px" .center}
        No need to define backward, autograd will handle it.

    + Dataset Loader
        ![linear](./images/Lec08/1%20(11).png){: width="600px" .center}
        Iterate over loader to form minibatches.
        
在PyTorch中，自定义数据集需要继承`torch.utils.data.Dataset`类，并实现三个方法：`__init__`、`__len__`和`__getitem__`。以下是一个自定义数据集类的基本示例，包括如何使用PyTorch的DataLoader进行数据加载(假设我们从CSV文件中读取数据):

```python
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 读取CSV文件
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.scaler = StandardScaler()

        # 假设数据集的最后一列是标签，其余列是特征
        self.features = self.data_frame.iloc[:, :-1].values
        self.labels = self.data_frame.iloc[:, -1].values

        # 标准化特征
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        # 返回数据集的大小
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取指定索引的数据和标签
        features = self.features[idx]
        label = self.labels[idx]

        # 如果有数据转换操作，应用转换
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
```

!!!info "Pretrained Models"
    ![linear](./images/Lec08/1%20(1).png){: width="600px" .center}
