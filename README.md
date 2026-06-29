# A股选股ML Pipeline

全流程A股量化选股系统：数据获取 → 特征工程 → 模型训练 → 共识选股。

## 模型

| 模型 | 验证集年均 | 说明 |
|---|---|---|
| REG∩RF 共识 | +57.0% (CV) | 部署首选 |
| REG (256x128x64) | +46.9% (CV) | 共识空时兜底 |
| RF (100树) | +40.8% (CV) | 泛化最稳 |

CV为全行业5折逐年交叉验证。

## 快速开始

```bash
# 1. 拉取最新财报
python 选股模型数据/pull_new.py

# 2. 更新股价缓存
python 选股模型数据/update_prices.py

# 3. 构建数据集
python 选股模型数据/build_extend.py

# 4. 训练+预测 (需要GPU)
python 选股模型数据/deploy.py
```

## 数据

- 财报来源: AkShare (stock_yjbb_em + stock_zcfz_em)
- 股价来源: AkShare (stock_zh_a_hist, 前复权)
- 覆盖: 全A股6大类 ~5000只, 2019Q1至今
- 特征: 14维财务指标 + 6行业one-hot

## 依赖

```
pandas numpy scikit-learn torch akshare
```

详见 requirements.txt

## 项目结构

```
├── 选股模型数据/
│   ├── 核心流程.md          # 完整流程文档
│   ├── 数据流水线日志.md     # 实验记录
│   ├── build_extend.py      # 数据集构建
│   ├── pull_new.py          # 增量财报拉取
│   ├── update_prices.py     # 股价缓存更新
│   ├── deploy.py            # 部署训练+预测
│   ├── history_data/        # 数据+模型
│   │   ├── training_all.csv       # 原数据集(2019-2024)
│   │   ├── training_extended.csv  # 扩展数据集(2019-2026)
│   │   ├── ak_cache/        # 财报缓存(parquet)
│   │   ├── price_cache/     # 股价缓存(pkl)
│   │   └── models/          # 模型权重(pt/pkl)
│   └── ...                  # 其他实验脚本
└── README.md
```

## 部署策略

1. REG∩RF 共识优先 (高胜率, 少而精)
2. 共识为空时 REG 单独买入 (覆盖率兜底)
3. 全部通过 S1∩S3∩S4 财务掩码

## License

MIT
