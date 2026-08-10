# @Time   : 2024
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
预测头消融实验模块

提供 5 种可插拔的预测层变体，用于替换论文公式 (11)(12) 中的 "特征拼接 + MLP":
  - DotProductHead : 纯内积 (无参数，最强可解释性)
  - BilinearHead   : 双线性评分 u^T W v (线性交互矩阵)
  - GatedHead      : 门控融合 + MLP (显式建模两侧贡献权重)
  - ElemProdMLPHead: Element-wise Product + MLP (当前 forward 实现)
  - ConcatMLPHead  : Concat(product, sum) + MLP (论文公式 11/12, forward0)

用法:
    在配置中添加 prediction_head: 'elem_mlp' 即可切换。
    可选值: dot | bilinear | gated | elem_mlp | concat_mlp
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """预测头基类，所有变体继承此类"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, user_vec, item_vec):
        raise NotImplementedError


class DotProductHead(PredictionHead):
    """变体1: 纯内积，无任何可学习参数，最强可解释性"""
    def forward(self, user_vec, item_vec):
        return (user_vec * item_vec).sum(dim=1)


class BilinearHead(PredictionHead):
    """变体2: 双线性评分 u^T W v，单一交互矩阵，参数少、结构透明"""
    def __init__(self, dim):
        super().__init__(dim)
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, user_vec, item_vec):
        return self.bilinear(user_vec, item_vec).squeeze(1)


class GatedHead(PredictionHead):
    """变体3: 门控融合 — 显式建模 user/item 两侧对匹配分数的贡献权重"""
    def __init__(self, dim):
        super().__init__(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, user_vec, item_vec):
        gate = self.gate(torch.cat([user_vec, item_vec], dim=1))
        fused = gate * user_vec + (1 - gate) * item_vec
        return self.mlp(fused).squeeze(1)

    def get_gate_values(self, user_vec, item_vec):
        """返回门控值，用于分析 user/item 各自对匹配的贡献权重"""
        gate = self.gate(torch.cat([user_vec, item_vec], dim=1))
        return gate


class ElemProdMLPHead(PredictionHead):
    """变体4: Element-wise Product + MLP (当前 forward 实现)"""
    def __init__(self, dim):
        super().__init__(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, user_vec, item_vec):
        return self.mlp(user_vec * item_vec).squeeze(1)


class ConcatMLPHead(PredictionHead):
    """变体5: Concat(product, sum) + MLP (论文公式 11/12, forward0 实现)"""
    def __init__(self, dim):
        super().__init__(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, user_vec, item_vec):
        x = torch.cat([user_vec * item_vec, user_vec + item_vec], dim=1)
        return self.mlp(x).squeeze(1)


_PREDICTION_HEAD_REGISTRY = {
    'dot':        DotProductHead,
    'bilinear':   BilinearHead,
    'gated':      GatedHead,
    'elem_mlp':   ElemProdMLPHead,
    'concat_mlp': ConcatMLPHead,
}


def get_prediction_head(name, dim):
    """工厂函数，根据配置名创建对应的预测头

    Args:
        name (str): 预测头名称，可选 dot | bilinear | gated | elem_mlp | concat_mlp
        dim (int):  特征向量维度 (embedding_size)

    Returns:
        PredictionHead 子类实例
    """
    if name not in _PREDICTION_HEAD_REGISTRY:
        raise ValueError(
            f'Unknown prediction head: "{name}". '
            f'Available: {list(_PREDICTION_HEAD_REGISTRY.keys())}'
        )
    return _PREDICTION_HEAD_REGISTRY[name](dim)
