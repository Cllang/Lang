# 作者：陈朗
# 2023年08月09日17时38分52秒
import torch
import random

# 1，2，6，7 题对应代码
def generate_data_for_Georg(true_R, number):
    I = torch.rand(number,1) #生成电流值，物理意义均为正，在此用(0,1)均匀分布
    U = torch.matmul(I, true_R) #U=IR
    #U += torch.rand(U.shape) #引入(0,1)分布的噪声（以防电压负值）
    U += torch.normal(0, 0.1, U.shape)
    return I, U

def split_data_into_batch(batch_size, I, U):# 载入数据集，分成batch
    n_data = len(U)
    data_index = list(range(n_data)) # 创建整个数据集的索引
    random.shuffle(data_index) # 打乱索引，便于随机抽取batch
    for i in range(0, n_data, batch_size):
        batch_index = torch.tensor(data_index[i:min(i + batch_size, n_data)])
        yield I[batch_index], U[batch_index] #用yield不用return便于训练时的迭代

def model_for_Georg(I, R): # 欧姆定律模型~
    return torch.matmul(I, R)

def loss_for_Georg(predicted_U, U): # 采用的是交叉熵
    return 0.5/len(U)*(predicted_U - U).norm()

def sgd_for_Georg(param, learning_rate): #小批量随机梯度下降优化算法
    with torch.no_grad(): #优化算法更新参数不能算在计算图中，所以先声明一下
        param -= learning_rate * param.grad
        param.grad.zero_()

true_R = torch.tensor([3.5])
I, U = generate_data_for_Georg(true_R, 3000)
learning_rate = 0.1
# 6、这个学习率会很慢，在6个epoch内学不到比较准确的R
#learning_rate = 0.01
batch_size = 30
# 7、如果样本个数不能被整除，在epoch中的最后一个Batch的样本个数不到batchsize
#batch_size = 7 # 我代码里的loss是按照取出batch的大小算的，即使不能整除也不会像视频里那样出错~~~
epoch_size = 10
model = model_for_Georg
optimizer = sgd_for_Georg
loss = loss_for_Georg

R = torch.normal(0, 0.01, size = true_R.shape, requires_grad = True)
# 1、无隐藏层时权重可以初始化为0 ，但是后续如果有隐藏层权重初始化为0会导致训练过程中所有隐藏层权重都是相等的
#R = torch.tensor([0], dtype = torch.float32, requires_grad = True) # 也可以训练
for epoch in range(epoch_size):
    for i, u in split_data_into_batch(batch_size, I, U):
        l = loss(model(i, R), u)
        l.backward() #反向传播计算梯度
        optimizer(R, learning_rate) #梯度下降优化
    with torch.no_grad(): #计算epochloss时，只是检验一下，不需要算在计算图里
        train_loss = loss(model(I, R), U)
    print(f'train loss for epoch {epoch} is {train_loss} \n')
print('实际的电阻值 = ', true_R, '\n', '训练学习到的电阻值 = ', R)