import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter

"""
WGAN-Based Oversampling

适用场景：
- 该算法通过Wasserstein GAN生成少数类样本，用于类别不平衡问题。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- sampling_strategy: dict，采样策略，默认为‘auto’，例如{0: 10, 1: 7}表示类别0采样10个，类别1采样7个。
- random_state: int，随机种子，确保结果可重复，默认为None。
- z_dim: int，噪声向量的维度。
- epochs: int，训练WGAN的轮次。
- batch_size: int，批量大小。

返回值：
- resampled_data: ndarray，生成的合成样本数据。
- resampled_labels: ndarray，生成的合成样本对应的标签。
"""

# WGAN生成器
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, img_dim)
        )

    def forward(self, z):
        return self.fc(z)

# WGAN判别器
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

# WGAN训练过程
def train_wgan(generator, discriminator, dataloader, z_dim, epochs=200, lr=0.00005):
    optimizer_g = optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):  # 修正此处，移除标签部分
            real_data = real_data[0]  # 只提取数据部分
            optimizer_d.zero_grad()

            # 判别器对真实数据评分
            real_labels = torch.ones(real_data.size(0), 1)  # 真实标签

            output_real = discriminator(real_data)
            loss_real = criterion(output_real, real_labels)

            # 判别器对生成数据评分
            z = torch.randn(real_data.size(0), z_dim)  # 噪声
            fake_data = generator(z)
            fake_labels = torch.zeros(real_data.size(0), 1)  # 生成数据标签

            output_fake = discriminator(fake_data)
            loss_fake = criterion(output_fake, fake_labels)

            # 总判别器损失
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # 每隔几步训练生成器
            if i % 5 == 0:
                optimizer_g.zero_grad()
                z = torch.randn(real_data.size(0), z_dim)  # 噪声
                fake_data = generator(z)
                output_fake = discriminator(fake_data)
                loss_g = criterion(output_fake, real_labels)  # 生成器目标是欺骗判别器

                loss_g.backward()
                optimizer_g.step()

        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {loss_d.item()}, G Loss: {loss_g.item()}")


# WGAN-Based Oversampling
def wgan_based_oversampling(data, labels, sampling_strategy, z_dim=100, epochs=100, batch_size=32, random_state=None):
    # 创建生成器和判别器
    generator = Generator(z_dim, data.shape[1])
    discriminator = Discriminator(data.shape[1])

    # 采样策略
    resampled_data = data.copy()
    resampled_labels = labels.copy()

    # 按类别分配数据
    unique_classes = np.unique(labels)
    class_samples = {cls: data[labels == cls] for cls in unique_classes}

    # 训练WGAN并为每个类别生成数据
    for cls in unique_classes:
        target_count = sampling_strategy.get(cls, len(class_samples[cls]))
        current_count = len(class_samples[cls])

        if target_count > current_count:
            n_samples_to_generate = target_count - current_count
            class_data = class_samples[cls]

            # 将类数据转化为Tensor并创建DataLoader
            class_data_tensor = torch.tensor(class_data, dtype=torch.float32)
            dataset = TensorDataset(class_data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 训练WGAN
            train_wgan(generator, discriminator, dataloader, z_dim, epochs)

            # 生成合成样本
            z = torch.randn(n_samples_to_generate, z_dim)  # 生成噪声
            synthetic_data = generator(z).detach().numpy()

            # 添加合成样本到结果数据
            resampled_data = np.vstack([resampled_data, synthetic_data])
            resampled_labels = np.hstack([resampled_labels, [cls] * n_samples_to_generate])

    return resampled_data, resampled_labels


def OverWGAN(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'WGANBased{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    # 从表单中获取可调参数
    z_dim = int(request.form.get("WGAN_z", 10))
    epochs = int(request.form.get("WGAN_epochs", 50))
    batch_size = int(request.form.get("WGAN_batch_size", 10))

    labels = label.ravel()
    # 调用函数进行过采样
    data, label = wgan_based_oversampling(
        data, labels, label_count_dict, z_dim=z_dim, epochs=epochs, batch_size=batch_size
    )

    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Now counts:", label_count_dict)

    return data, label


# 测试WGAN-Based Oversampling
if __name__ == '__main__':
    # 创建示例数据
    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 10))
    data_class_1 = np.random.normal(loc=2, scale=0.2, size=(5, 10))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(4, 10))

    # 合并数据和标签
    data = np.vstack((data_class_0, data_class_1, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2] * 4)

    # 设置采样策略
    sampling_strategy = {0: 15, 1: 10, 2: 10}

    # 调用 WGAN-Based Oversampling 进行过采样
    resampled_data, resampled_labels = wgan_based_oversampling(
        data, labels, sampling_strategy, z_dim=10, epochs=50, batch_size=3
    )

    print("Resampled labels:", Counter(resampled_labels))
    print("Resampled data:", resampled_data)
