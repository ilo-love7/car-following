<div align="center">

# Car-Following Traffic Modelling Toolkit

**七种经典跟驰模型 + 稳定性分析 + 参数标定 + 轨迹预测，一个可复现的 Python 包。**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 46 passing](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)](./tests)

<p align="center">
  <a href="./README.md">English</a> · <b>简体中文</b>
</p>

</div>

---

一个自包含、可复现的微观交通流建模库：7 种经典跟驰模型统一接口，包含基本图解析推导、线性/环形/弦稳定性分析、有界最小二乘标定、轨迹预测、MOBIL 换道与电动车能耗。

## 特性

- **7 种跟驰模型**统一接口：IDM、OVM、FVD、Helly/ACC、GHR、Gipps、Newell
- **基本图与通行能力**由平衡间距解析推导
- **线性稳定性**有闭式判据，并与环形车道特征值交叉验证
- **弦稳定性**：Helly/ACC 给出解析传递函数
- **有界最小二乘标定**，附噪声鲁棒性分析
- **轨迹预测**：CV / CA / IDM / LSTM 四个基线
- **MOBIL 换道**双车道微观仿真
- **电动车能耗**模型，含能量回收
- **NGSIM 数据适配器**，配合成 fixture 做端到端验证

## 快速开始

```bash
# 克隆
git clone https://github.com/ilo-love7/car-following.git
cd car-following

# 安装
pip install -r requirements.txt

# 跑全部实验（仅需 CPU）
python scripts/run_all.py

# 跑测试
python -m pytest tests -q
```

测试的预期输出：

```
46 passed
```

## 这个工具包能做什么

六个可直接上手的使用场景，每个都有可运行的入口（纯 CPU，不需要 GPU，也不需要私有数据）：

| 用途 | 入口 |
|---|---|
| **在同一条件下对比 7 种跟驰模型**（IDM / OVM / FVD / Helly / Newell 等） | `python scripts/run_all.py case1` · `carfollowing.models.make_model` |
| **做线性/车队稳定性分析与通行能力定位** | `carfollowing.stability.linear_stability`、`carfollowing.equilibrium.capacity_point` · `scripts/run_all.py case2 case3` |
| **用实测轨迹（NGSIM）标定模型参数** | `carfollowing.calibration` · `scripts/run_all.py case4` |
| **短时域轨迹预测与指标评估** | `carfollowing.metrics` · `scripts/run_all.py case5` |
| **基于 MOBIL 的换道建模** | `carfollowing.lanechange` · `scripts/run_all.py case6` |
| **从跟驰轨迹算电动车能耗** | `carfollowing.energy` · `scripts/run_all.py case7` |

典型用途：① 作为**新模型的对照基线**；② 作为标定类论文的**可复现实验台**；
③ 交通流课程的**教学材料**；④ 把 ACC/ADAS 逻辑放进仿真前的**健全性检查**。

全部可离线复现：`python -m pytest tests -q`（46 个测试）不需要任何数据集，CI 覆盖 Linux 与 Windows。

## 用法

```python
from carfollowing import IDM, make_model
from carfollowing.simulate import simulate_following
from carfollowing.stability import linear_stability

m = IDM(v0=30, T=1.2, s0=2.0, a=1.5, b=2.0)
print(m.equilibrium_gap(20.0))          # 平衡间距
print(linear_stability(m, 20.0))        # 线性稳定性裕度
```

## 仓库结构

```
car-following/
├── src/carfollowing/          # 核心库
│   ├── models.py              # 7 种跟驰模型
│   ├── simulate.py            # 单跟驰车 & 环形车道仿真
│   ├── equilibrium.py         # 微观基本图、通行能力
│   ├── stability.py           # 线性、环形与弦稳定性
│   ├── calibration.py         # 有界非线性最小二乘标定
│   ├── metrics.py             # 精度与安全指标
│   ├── energy.py              # 纵向电动车能耗模型
│   ├── lanechange.py          # MOBIL 换道与双车道微观仿真
│   └── datasets/
│       ├── synthetic.py       # 可复现合成基准
│       └── ngsim.py           # NGSIM 加载与事件抽取
├── scripts/                   # 实验脚本 + run_all.py
├── tests/                     # 46 个 pytest 用例
├── data/                      # 生成的 CSV / JSON / fixture
├── figures/                   # 结果图
├── reports/                   # 技术报告、验收核查表、数据字典
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## 报告

- [`reports/技术报告.md`](reports/技术报告.md) — 完整技术报告
- [`reports/验收核查表.md`](reports/验收核查表.md) — 验收核查表
- [`reports/数据来源与数据字典.md`](reports/数据来源与数据字典.md) — 数据来源与数据字典

## 许可证

MIT License，详见 [`LICENSE`](LICENSE)。

## 关联仓库

本仓库属于同一作者在电池与交通建模方向的系列工作：

| 仓库 | 内容 |
|---|---|
| [battery-cathode-ml](https://github.com/ilo-love7/battery-cathode-ml) | 正极材料筛选：带隙 / 形成能 / 电压 / 容量 + 不确定度 + 多目标决策 |
| [battery-lifetime-guide](https://github.com/ilo-love7/battery-lifetime-guide) | 从零复现 Severson 2019 循环寿命预测（含翻车记录） |
| [battery-soh-tsfm-benchmark](https://github.com/ilo-love7/battery-soh-tsfm-benchmark) | 零样本时序基础模型 vs 线性外推：三数据集 SOH 基准 |
| [battery-ecm-rul](https://github.com/ilo-love7/battery-ecm-rul) | Thevenin ECM 参数辨识 + 早期寿命预测 + PyBaMM 仿真 |
| [drift-diffusion-pn-junction](https://github.com/ilo-love7/drift-diffusion-pn-junction) | 自研 1D 漂移–扩散–泊松求解器 + 三层验证 |
| [car-following](https://github.com/ilo-love7/car-following) | 跟驰建模工具包（7 模型 / 稳定性 / 标定 / 换道 / EV 能耗） |
| [project-packer](https://github.com/ilo-love7/project-packer) | 把真实项目包装成可防守的面试经历（事实档案 + 追问树） |
