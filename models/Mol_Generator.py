import torch as th
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_sort_pool
from models.EGNN_Block import EGNNlayer


class ResidualBlock(nn.Module):
    """
    3D 卷积版本的残差模块 (Residual Block)，和 ResNet 的基本残差单元类似，
    只不过是用在 三维数据（体数据，比如医学图像、分子密度图、点云体素化结果）
    模块的作用: 
        保证 信息可以绕过卷积层直接传递（残差思想）
        减少深层网络的梯度消失问题
        能够训练更深的 3D 卷积网络
    """

    def __init__(self, channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.padding = padding   # 用于提取 3D 特征。卷积核大小是 3×3×3
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)  # 批归一化，加快训练并稳定收敛
        self.relu = nn.ReLU(inplace=True)
        # conv1 和 conv2 的 stride 都是相同的，意味着残差分支（identity）和主分支的空间分辨率必须保持一致，否则会维度不匹配
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity      # 残差连接
        out = self.relu(out)
        return out


class GrowthPointPredictionModel(nn.Module):

    def __init__(self, dim_in=10, dim_out=3, dim_edge_feat=7, num_layers=2):
        """
        dim_in=10: 节点 (原子/点) 的初始特征维度是 10
        dim_out=3: 最终分类输出的类别数是 3 (例如预测三种不同的 “生长点” 类别)
        dim_edge_feat=7: 边的特征维度是 7 (例如化学键类型、距离编码等)
        num_layers=2: 堆叠 EGNN 的层数
        """
        super(GrowthPointPredictionModel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_edge_feat = dim_edge_feat
        self.num_layers = num_layers
        self.EGNN = EGNNlayer(self.dim_in, self.dim_in, self.dim_in, edge_feat_size=self.dim_edge_feat)
        # 等变图神经网络层（Equivariant GNN），同时更新节点特征 x 和节点坐标 coords。它保证了对旋转、平移的不变性
        self.fc = nn.Linear(self.dim_in, self.dim_in)  # 把节点特征维度保持在 dim_in
        self.fc2 = nn.Linear(self.dim_in, self.dim_out)  # 把特征映射到 dim_out（即分类类别数）        
    
    def _make_layer(self, channels, num_blocks):   # 这里没有用上
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, edge_index, x, coords, edge_feat):
        for i in range(self.num_layers):
            x_out, coords_out = self.EGNN(edge_index, x, coords, edge_feat)
            # edge_index: 图的边信息（邻接关系）  x: 节点特征（dim_in 维）  coords: 节点坐标（3D 坐标） edge_feat: 边特征（dim_edge_feat 维）
            x_out = x_out.relu()
        x = self.fc(x_out)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class TorsionAnglePredictionModel(nn.Module):

    """
    多模态融合模型：同时利用
    分子片段 (frag graph)
    核心骨架 (core graph)
    三维网格 (grid, 体素化密度或电子密度？)
    来预测  分子的扭转角
    """

    def __init__(self, frag_dim_in=32, core_dim_in=35,  # 片段/核心图的节点特征维度
                 egnn_dim_tmp=1024, egnn_dim_out=256, egnn_num_layers=7, dim_edge_feat=7,  # EGNN 参数
                 cnn_dim_in=1, cnn_dim_tmp=512, stride=2, padding=3,  # 3D CNN 参数, CNN 通道逐步扩展到 512
                 dim_tmp=128, dim_out=36, num_layers=6):   # MLP 参数  把扭转角离散成 36 个 bin，每个 10°
        super(TorsionAnglePredictionModel, self).__init__()
        self.frag_dim_in = frag_dim_in
        self.core_dim_in = core_dim_in
        self.egnn_dim_tmp = egnn_dim_tmp
        self.egnn_dim_out = egnn_dim_out
        self.egnn_num_layers = egnn_num_layers
        self.dim_edge_feat = dim_edge_feat
        self.cnn_dim_in = cnn_dim_in
        self.cnn_dim_tmp = cnn_dim_tmp
        self.stride = stride
        self.padding = padding
        self.dim_tmp = dim_tmp
        self.dim_out = dim_out
        self.num_layers = num_layers
        # 处理分子片段
        self.layer_frag = EGNNlayer(self.frag_dim_in, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 处理核心骨架
        self.layer_core = EGNNlayer(self.core_dim_in, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 将中间输出结果堆叠 多层 EGNN
        self.layer1 = EGNNlayer(self.egnn_dim_out, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        # 融合 片段 graph + 核心 graph + 3D CNN 特征后，逐层映射到 36 类
        self.fc = nn.Linear(self.egnn_dim_out * 2 + self.cnn_dim_tmp * 3, self.dim_tmp * 2)
        self.fc2 = nn.Linear(self.dim_tmp * 2, self.dim_tmp * 2)
        self.fc3 = nn.Linear(self.dim_tmp * 2, self.dim_tmp)
        self.fc1 = nn.Linear(self.dim_tmp, self.dim_out)
        self.bn1 = nn.BatchNorm1d(self.dim_tmp)

        # 3D CNN 部分，连续 4 层 3D 卷积 + 3D 残差模块：
        self.bn3d = nn.BatchNorm3d(self.cnn_dim_in)
        self.conv3D1 = nn.Conv3d(self.cnn_dim_in, self.cnn_dim_tmp // 8, kernel_size=7, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D2 = nn.Conv3d(self.cnn_dim_tmp // 8, self.cnn_dim_tmp // 4, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D3 = nn.Conv3d(self.cnn_dim_tmp // 4, self.cnn_dim_tmp // 2, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D4 = nn.Conv3d(self.cnn_dim_tmp // 2, self.cnn_dim_tmp, kernel_size=3, stride=self.stride, padding=self.padding, bias=False) 
        self.layer3D1 = self._make_layer(self.cnn_dim_tmp // 8)
        self.layer3D2 = self._make_layer(self.cnn_dim_tmp // 4)
        self.layer3D3 = self._make_layer(self.cnn_dim_tmp // 2)
        self.layer3D4 = self._make_layer(self.cnn_dim_tmp)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 最后 AdaptiveAvgPool3d 压缩成全局向量

    def _make_layer(self, channels, num_blocks=2):   # 堆叠多个 3D 残差块
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, edge_index, x, coords, edge_feat, batch,  # 片段图
                edge_index1, x1, coords1, edge_feat1, batch1,   # 核心图
                grid):                                          # 三维网格
        # 片段 graph 特征提取：
        x_out, coords_out = self.layer_frag(edge_index, x, coords, edge_feat)
        x_out = x_out.relu()
        for i in range(self.egnn_num_layers):
                                   # layer1 被复用 在 frag/core graph 上，这样参数是共享的
            x_out, coords_out = self.layer1(edge_index, x_out, coords_out, edge_feat)  # EGNN 提取空间不变特征
            x_out = x_out.relu()
        readout = global_sort_pool(x_out.squeeze(dim=0), batch, k=3)  # 把节点特征池化为图级别表示（k=3，取前 3 个特征拼接？）

        # 核心 graph 特征提取：
        x_out1, coords_out1 = self.layer_core(edge_index1, x1, coords1, edge_feat1)
        x_out1 = x_out1.relu()
        for i in range(self.egnn_num_layers):
                                   # layer1 被复用 在 frag/core graph 上，这样参数是共享的
            x_out1, coords_out1 = self.layer1(edge_index1, x_out1, coords_out1, edge_feat1)  # EGNN 提取空间不变特征
            x_out1 = x_out1.relu()
        readout1 = global_sort_pool(x_out1, batch1, k=3) 

        # 3D CNN 特征提取，逐层卷积 → 残差块 → 下采样 → 全局池化，输出展平成一个向量：
        grid = self.bn3d(grid)
        grid = self.conv3D1(grid)
        grid = self.layer3D1(grid)  # 每次 conv3D 之后立刻跟一个 ResidualBlock，目的是让网络在提取特征的同时还能保持残差信息传递
        grid = self.conv3D2(grid)
        grid = self.layer3D2(grid)
        grid = self.conv3D3(grid)
        grid = self.layer3D3(grid)   
        grid = self.conv3D4(grid)
        grid = self.layer3D4(grid)
        grid = self.avgpool(grid)
        grid = th.flatten(grid, 1)

        readout = th.cat((readout, readout1, grid), dim=1)  # 多模态融合，把 片段图、核心图、3D CNN 特征拼接
        readout = self.fc(readout)
        for i in range(self.num_layers):
            readout = self.fc2(readout).relu() + readout   # 残差式 MLP
        readout = self.fc3(readout).relu()
        readout = self.bn1(readout)
        readout = self.fc1(readout)
        readout = F.softmax(readout, dim=1)   # 输出 36 维向量 → softmax 得到角度类别概率分布
        return readout