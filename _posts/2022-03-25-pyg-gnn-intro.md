---
layout: article
title: PyTorch Geometric 教程1 - 介绍
key: pyg-gnn-intro
tags: PyG, GNN
category: blog
pageview: true
date: 2022-03-25 10:00:00 +08:00
---

组合优化中有一种问题叫路由问题 (routing problem), 代表是旅行商人问题 (Travelling Salesman Problem, TSP). 而这类问题, 本身的数据结构就是图 (Graph) 结构, 在构建和求解上, GNN **似乎** 具有天然的优势. 当下 GNN 大火, 有两个库是最热门的: [Deep Graph Library (DGL)](https://www.dgl.ai/) 和 [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/en/latest/). 这两个库都很好用, 差别也不特别大 (DGL官网是有中文教程的). 但是PyG相对来说更基础一些, 教程与支持也更多一些. 因此笔者打算在自我学习之余, 翻译, 理解并整理官方的英文教程. 如果有错误, 或者疑问, 十分欢迎留言或者私信讨论!

---

# 下载并引入库

```
# Install required packages.
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Helper function for visualization.
%matplotlib inline
import torch
import networkx as nxc
import matplotlib.pyplot as plt

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
```

# GNN 介绍
图神经网络旨在泛化经典的深度学习概念至非常规的数据结构 (与图像或文字不同), 使得神经网络能够对研究对象与其关系进行推理. 举个例子, 我们可以使用一个简单的 **神经消息传递架构 (neural message passing scheme)** 来实现这一需求. 在某个图中 $G = (V, E)$, 所有节点 $v \in {V}$ 的特征向量 $\mathbf{x}_v^{(\ell)}$ 都会被从它们的 邻居节点 (neighbors) $\mathcal{N}(v)$ 聚集 (aggregate) 本地信息, 来进行迭代地更新:

$$\mathbf{x}_v^{(\ell + 1)} = f^{(\ell + 1)}_{\theta} \left( \mathbf{x}_v^{(\ell)}, \left\{ \mathbf{x}_w^{(\ell)} : w \in \mathcal{N}(v) \right\} \right)$$

```
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
```

输出结果:
```
Dataset: KarateClub():
======================
Number of graphs: 1
Number of features: 34
Number of classes: 4
```

在初始化 `KarateClub` 数据集之后, 我们首先可以检查它的一些属性. 例如, 我们可以看到这个数据集只有一张图, 并且每个节点都有一个34维的特征向量 (用来 **唯一** 描述Karate俱乐部的每个成员). 另外, 这张图只有4个类别, 这代表每个节点所属的社区. 现在让我们看看这个图的更多细节:

```
data = dataset[0]  # 获取第一个图对象.

print(data)
print('==============================================================')

# 查看图的数据.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
```

输出结果:
```
Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
==============================================================
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Number of training nodes: 4
Training node label rate: 0.12
Has isolated nodes: False
Has self-loops: False
Is undirected: True
```

在PyG内, 每张图都由某个 `Data` 对象表示, 该对象拥有所有描述 图表征 (graph representation) 的信息. 不难发现, 这个 `data` 对象拥有4个属性: 
1. 代表节点特征的 `x`: 34个节点, 每个节点有一个34维的特征向量
2. 代表图连通性 (connectivity) 的 `edge_index`: 每条边, 表示源 (source) 节点到目标 (destination) 节点是否相连
3. 节点标签 `y`: 每个节点都只有一个类别
4. 训练用掩码 `train_mask`: 描述了我们已知节点的社区分配

总的来说, 我们只知道4个节点的 真实 (groud-truth) 标签, 我们的任务是推断剩余节点的社区分配. data 对象也提供一些便利函数来推断一些图下基础属性. 比如, 我们可以轻易判断在图中是否存在孤立节点, 也就是其与任意其他节点都没有边相连; 该图是否包含 **自循环 (self-loop)**, 数学上我们可以表示为 $(v, v) \in E$; 该图是否为 **无向 (undirected)** 图, 也就是对每条边 $(v, w) \in E$, 同样存在 (w, v) \in \mathcal{E} . 让我们接下来检查一下 `edge_index` 的属性:

```
edge_index = data.edge_index
print(edge_index.t())
```

输出结果:
```
tensor([[ 0,  1],
        [ 0,  2],
        [ 0,  3],
        ...
        [33, 30],
        [33, 31],
        [33, 32]])
```

通过打印出 `edge_index`, 我们就可以理解 PyG 是怎样在内部表达图连通性的了. 我们可以看到对于每条边, `edge_index` 都拥有两个索引的 元组 (tuple), 其中第一个值是源节点的索引, 而第二个值则是对应边的目标节点的索引.

这种表达也被叫做 **COO 格式 (coordinate format)**, 它通常被用作表达稀疏矩阵. 与在密集表达内持有邻接信息 $\mathbf{A} \in \{0, 1\}^{|V| \times |V|}$ 不同, PyG 稀疏地表达图, 也就是只持有 $\mathbf{A}$ 里非零的坐标或值. 而且, PyG 并不区分有向和无向图, 它将无向图视作有向图的一个特例. 我们可以将这个图转化为 `networkx` 库格式, 来可视化它:

```
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)
```

输出结果:

![Graph Visualization](/fig/pyg-gnn-intro-graph-visual.png)

# 实现 GNN

在学习PyG怎么处理数据之后, 现在该实现第一个我们的 GNN 了! 让我们使用最简单的 GNN 运算器 (operator) 之一, [图卷积层 (Graph Convolutional Network, GCN)](https://arxiv.org/abs/1609.02907) 定义为:

$$\mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)} $$

其中 $\mathbf{W}^{(\ell + 1)}$ 代表一个可训练的权重矩阵, 大小为 `[num_output_features, num_input_features]`, 而 $c_{w,v}$ 指的是每条边的固定的归一化系数. PyG 通过 `GCNConv` 来实现这一次层, `GCNConv` 的输入为节点特征表达 `x` 和 COO图连通性表达 `edge_index`. 与常规的 PyTorch 神经网络结构相似, 可以通过定义一个 `torch.nn.Module` 类来构建我们的第一个图神经网络:

```
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)
```

输出结果:
```
GCN(
  (conv1): GCNConv(34, 4)
  (conv2): GCNConv(4, 4)
  (conv3): GCNConv(4, 2)
  (classifier): Linear(in_features=2, out_features=4, bias=True)
)
```

在这里, 我们首先在 `__init__` 里初始化构建模块, 并在 `forward` 里定义计算流程. 我们先定义并堆叠 **3图卷积层**, 其对应聚集每个节点周围 3-hop 的邻居信息 (也就是说所有节点最多距离三次 "跳跃" 这么远). 另外, `GCNConv` 层减少节点特征维度至 2, 也就是 $34 \to 4 \to 4 \to 2$. 注意, 每层 `GCNConv` 都被 [tanh 非线性函数](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html?highlight=tanh#torch.nn.Tanh) 增强. 

之后, 我们应用单层 **线性变换 (linear transformation)** `torch.nn.Linear` 来作为分类器, 将节点映射到4个类或社区中的一个. 最后, 我们同时返回最终分类器的输出和 GNN 产生的最终的 节点嵌入 (node embeddings). 

# 嵌入 Karate 俱乐部网络

让我们来看看 GNN 产生的节点嵌入. 在这里, 我们传递初始节点特征 `x` 和 图连通性信息 `edge_index` 到模型中, 并可视化它的 2D 嵌入.

```
model = GCN()

_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize_embedding(h, color=data.y)
```

输出结果:
```
Embedding shape: [34, 2]
```
![Node Embedding 0](/fig/pyg-gnn-intro-node-embedding-0.png)

值得注意的是, 即使在训练模型权重之前, 该模型也产生了与图的社区结构非常相似的节点嵌入. 相同颜色的节点代表它们所属社区相同. 虽然我们的模型权重是完全随机初始化的, 而且到目前为止我们还没有进行任何训练, 我们任然可以观察到这些节点已经在嵌入空间中被 聚类 (cluster) 了. 因此, 我们可以得出结论, GNN 引入了很强的 诱导偏差 (inductive bias), 这些偏差导致输入图中彼此想接近的节点出现类似的嵌入. 

# 基于 Karate 俱乐部网络的训练

当然我们可以做得更好, 既然我们模型中的所有东西都是可微分的且参数化的, 我们可以添加一些标签来训练这个模型, 并观测嵌入是如何反应的. 在这里, 我们利用 **半监督 (semi-supervised)**, 或者叫 **直推学习 (transductive learning)** 过程: 我们只针对每个类的一个节点进行训练, 但允许使用完整的图数据输入. 
训练过程也与大部分 PyTorch 常规模型相似. 除了定义网络架构, 我们还需要定义一个 损失标准 (loss critertion), 在这里我们选择 `CrossEntropyLoss`, 并初始化一个随机梯度优化器 (stochastic gradient optimizer), 我们选择 `Adam`. 之后我们执行多次优化, 每轮包含一个 前向 (forward) 和 后向 (backward) 传播, 以此来计算从前向传播推导出的模型参数. 如果你并不熟 悉PyTorch, 请移步至 [官方教程 a good introduction on how to train a neural network in PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-loss-function-and-optimizer).

现在, 我们的半监督场景是由下面的函数获得:

```
loss = criterion(out[data.train_mask], data.y[data.train_mask])
```

当我们为所有节点计算节点嵌入时, 我们只用训练节点来计算损失. 在这里, 我们筛选出分类器的输出 `out` 和真实标签 `data.y` 来包含在 `train_mask` 里的节点. 让我们看看代码是什么:

```
import time

model = GCN()
criterion = torch.nn.CrossEntropyLoss()                              # 定义损失标准
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)            # 定义优化器

def train(data):
    optimizer.zero_grad()                                            # 所有梯度归零
    out, h = model(data.x, data.edge_index)                          # 执行单次前向传递
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 基于训练节点单独计算损失
    loss.backward()                                                  # 后向传播以推导梯度
    optimizer.step()                                                 # 基于梯度更新参数
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    if epoch % 10 == 0:
        visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)
```

输出结果:

![Node Embedding 2](/fig/pyg-gnn-intro-node-embedding-2.png)

![Node Embedding 3](/fig/pyg-gnn-intro-node-embedding-3.png)

![Node Embedding 4](/fig/pyg-gnn-intro-node-embedding-4.png)

![Node Embedding 5](/fig/pyg-gnn-intro-node-embedding-5.png)

不难发现, 我们的3层GCN模型成功地区分了社区(线性的), 并准确地分类了大部分的节点.

# 参考

Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016). https://arxiv.org/abs/1609.02907