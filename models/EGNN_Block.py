import torch as th
from torch import nn
from torch_geometric.nn import MessagePassing

# 等变图神经网络 (Equivariant Graph Neural Network, EGNN)
class EGNNlayer(MessagePassing):
    """
    基于 PyTorch Geometric 的 MessagePassing。它能同时更新 节点特征 (x) 和 节点坐标 (coords),
    保证网络对 平移、旋转等操作保持等变性
    in_size: 输入节点特征的维度
    hid_size: 隐藏层的神经元数量
    out_size: 输出节点特征的维度
    edge_feat_size: 边特征的维度, 默认为 0 (即如果没有边特征)
    """

    def __init__(self, in_size, hid_size, out_size, edge_feat_size=0):
        super(EGNNlayer, self).__init__()
        #        边
        self.phi_e = nn.Sequential(nn.Linear(2 * in_size + edge_feat_size + 1,  # x_i, x_j（两个节点特征拼接）+ edge_feat + dist（节点间距离）
                                              hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, out_size),
                                  nn.ReLU())
        #        坐标
        self.phi_x = nn.Sequential(nn.Linear(out_size, hid_size),   # 输入来自 phi_e 的输出，表示边对节点坐标更新的影响
                                  nn.ReLU(),
                                  nn.Linear(hid_size, 1),  # 输出：一个标量 w（沿着边方向更新坐标的强度）
                                  nn.ReLU())
        #      节点特征更新网络
        self.phi_h = nn.Sequential(nn.Linear(in_size + out_size,  # 输入：原始节点特征 x + 来自 phi_e 的输出
                                             hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, out_size),  # 输出：更新后的节点特征 x_out
                                  nn.ReLU())
    
    def forward(self, edge_index, x, coords, edge_feat):

        rela_diff = coords[edge_index[0]] - coords[edge_index[1]]  # 两个节点的相对坐标差
        dist = th.norm(coords[edge_index[0]] - coords[edge_index[1]], dim = 1, keepdim = True)
        
        edge_feat = th.cat([edge_feat, dist], 1)   # 把距离拼到边特征里
        
        x_out, coords_out = self.propagate(edge_index, x = x, coords = coords,    # 然后调用 propagate（PyG 的消息传递机制）
                                           edge_feat = edge_feat, dist = dist, rela_diff = rela_diff)                
        return x_out, coords_out
        
    def propagate(self, edge_index, size = None, **kwargs):
        """
        在图结构数据上进行消息传递和节点更新
        edge_index: 表示图的边索引, 指示图中节点之间的连接关系
        size: 可选参数, 通常表示图的大小 (节点数)
        **kwargs: 允许传递额外的关键字参数
        self._check_input、self._collect、self.inspector.distribute、self.aggregate 
        都是 PyG 的 MessagePassing 基类里提供的封装工具
        """        
        size = self._check_input(edge_index, size)  # 验证 edge_index 和 size 的有效性，如果 size=None，PyG 会自动推断
        # 收集消息传递所需的特征:
        coll_dict = self._collect(self._user_args, edge_index, size,   # 把 全局特征拆分成消息传递需要的 x_i, x_j, edge-level 特征
                                         kwargs)
        
        # 分发消息参数
        msg_kwargs = self.inspector.distribute('message', coll_dict)  # self.inspector 会分析 message 和 update 函数的参数列表，然后自动把 coll_dict 里对应的张量分发给它们
        # 聚合函数的参数
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        msg = self.message(**msg_kwargs)   # 调用 message 方法！！！ 计算消息
        
        w = self.phi_x(msg)   # 先用 phi_x(msg) 得到权重 w
        rela_diff = kwargs["rela_diff"]
        aggr_w = self.aggregate(w * rela_diff, **aggr_kwargs)  # 沿着边方向更新坐标
        coords_out = kwargs["coords"] + aggr_w    # 聚合邻居贡献
        
        msg = self.aggregate( msg , **aggr_kwargs)
        
        x = kwargs["x"]
        x_out = self.update(x, msg)   # 调用 update 方法
        return x_out, coords_out
        
    def message(self, x_i, x_j, edge_feat):
        edge_feat = edge_feat.float()
        message = self.phi_e(th.cat([x_i, x_j, edge_feat], 1))
        return message

    def update(self, x, message):
        x_out = self.phi_h(th.cat([x, message], 1))
        return x_out