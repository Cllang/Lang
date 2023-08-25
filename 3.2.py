# 作者：陈朗
# 2023年08月09日17时35分35秒
import torch
import random


def dianliu_dianya_data(true_R, num_examples):
    I = torch.rand(num_examples, 1)  # 随机电流
    U = torch.matmul(I, true_R)  # U = IR
    # U += torch.rand(U.shape) #引入(0,1)分布的噪声（以防电压负值）
    U += torch.normal(0, 0.1, U.shape)
    return I, U


# def data_iter(I, U, batch_size):  # 载入数据集，分成batch
#     # 随机样本，意思是把顺序随机，可以把所有样本涉及到
#     num_examples = len(U)
#     indices = list(range(num_examples))
#     random.shuffle(indices)

#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
#         yield I[batch_indices], U[batch_indices]  # 生成器，方便迭代

def split_data_into_batch(batch_size, I, U):  # 载入数据集，分成batch
    n_data = len(U)
    data_index = list(range(n_data))  # 创建整个数据集的索引
    random.shuffle(data_index)  # 打乱索引，便于随机抽取batch
    for i in range(0, n_data, batch_size):
        batch_index = torch.tensor(data_index[i:min(i + batch_size, n_data)])
        yield I[batch_index], U[batch_index]  # 用yield不用return便于训练时的迭代


def oumu(I, R):  # 欧姆定律模型
    return torch.matmul(I, R)


def loss1(predicted_U, U):  # 交叉熵损失函数
    return 0.5 / len(U) * (predicted_U - U).norm()


def sgd_for_Georg(param, learning_rate):  # 小批量随机梯度下降优化算法
    with torch.no_grad():  # 优化算法更新参数不能算在计算图中，所以先声明一下
        param -= learning_rate * param.grad
        param.grad.zero_()


true_R = torch.tensor([3.5])
I, U = dianliu_dianya_data(true_R, 3000)
lr = 0.1

R = torch.normal(0, 0.01, size=true_R.shape, requires_grad=True)
# 1、无隐藏层时权重可以初始化为0 ，但是后续如果有隐藏层权重初始化为0会导致训练过程中所有隐藏层权重都是相等的
# true_R = torch.tensor([0],dtype = torch.float32,requires_grad = True)  # 这样也可以训练


num_epochs = 10
batch_size = 30
net = oumu
loss = loss1
optimizer = sgd_for_Georg

for epoch in range(num_epochs):
    for i, u in split_data_into_batch(batch_size, I, U):
        l = loss(net(i, R), u)  # I和U的小批量损失
        l.backward()
        optimizer(R, lr)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_loss = loss(net(I, R), U)
        print(f'epoch {epoch + 1}, loss {train_loss}')
print('实际的电阻值 = ', true_R, '\n', '训练学习到的电阻值 = ', R)
