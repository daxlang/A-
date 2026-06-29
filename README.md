# A股选股ML Pipeline

全流程A股量化选股系统：数据获取 → 特征工程 → 模型训练 → 共识选股。

## 模型表现

| 策略 | 验证集年均 | 验证集买入 | 数据集 | 说明 |
|---|---|---|---|---|
| REG∩RF 共识 | +57.0% | 223只 | 全行业5折CV | 部署首选，极度精选 |
| REG (256x128x64, drop0.5, wd0.02) | +46.9% | 593只 | 全行业5折CV | 单模型最强 |
| RF (100树, depth=20) | +40.8% | 425只 | 全行业5折CV | 泛化最稳，三数据集通吃 |

全行业S1234掩码内测试，5折逐年交叉验证。

## 快速开始

```bash
# 1. 拉取最新财报 (增量)
python 选股模型数据/pull_new.py

# 2. 更新股价缓存 (增量, 约25分钟)
python 选股模型数据/update_prices.py

# 3. 构建数据集 (约60分钟含首次股价拉取)
python 选股模型数据/build_extend.py

# 4. 训练+预测 (需要GPU, RTX5070约2分钟)
python 选股模型数据/deploy.py
```

## 核心流程文档

- **[核心流程.md](核心流程.md)** — 数据清洗、特征工程、训练、选股策略完整说明
- **[选股模型数据/数据流水线日志.md](选股模型数据/数据流水线日志.md)** — 全部实验记录（18个大章节，37个子实验）

---

## 文件清单

### 核心脚本

| 文件 | 用途 |
|---|---|
| `build_extend.py` | 完整数据集构建：财报合并 → 特征计算 → 股价拉取 → 估值填充 |
| `build_all.py` | 同上，但限定 2019-2024 年份（原始版本） |
| `pull_new.py` | 增量拉取最新季度 yjbb + zcfz 到 parquet 缓存 |
| `update_prices.py` | 增量更新全部股票价格缓存到最新日期 |
| `deploy.py` | 部署脚本：加载数据 → 预处理 → 训练 REG/RF → 生成预测列表 |
| `build_dataset.py` | 早期4行业数据集构建 |
| `build_full.py` | 8行业数据集构建 |
| `filter_outliers.py` | 极端值缩尾 |
| `fill_cr_ic.py` | 缺失值填补 |
| `clean_data.py` | 数据清洗 |

### 早期实验脚本

| 文件 | 对应日志 | 用处 |
|---|---|---|
| `pull_ak.py` / `pull_test.py` / `pull_rest.py` | 18.0-18.2 | AkShare 数据拉取初探 |
| `pull_history.py` / `pull_all_years.py` / `pull_all_features.py` | 18.3-18.6 | 历年财报批量拉取 |
| `expand_stocks.py` / `merge_stocks.py` | 18.7 | 股票池扩展与合并 |
| `extend_years.py` | 18.8 | 年份范围扩展 |
| `scrape_em.py` | — | 东方财富爬虫（备用） |
| `add_bs_cf_val.py` | 18.9 | BS/CF 估值补充 |

### 掩码实验 (S1/S2/S3/S4)

| 文件 | 对应日志 |
|---|---|
| `stage1_filter.py` / `stage2_train.py` | 18.12-18.13 | S1/S2 两阶段 | 
| `s2_on_s1.py` / `s2_only.py` / `s2_repair.py` | 18.15 | S2 修复 |
| `s3_check.py` / `s3_feat.py` / `s3_three.py` | 18.16-18.17 | S3 毛利率筛选 |
| `s4_check.py` | 18.27 | S4 负债率筛选 |
| `compare_pass.py` | 18.14 | S1_PASS vs S1_FAIL 对照 |
| `split_train.py` / `split_cm.py` | — | 分行业混淆矩阵 |
| `baseline_models.py` | 18.9 | 四行业 LR/RF 基线 |

### 模型架构与训练

| 文件 | 对应日志 | 用处 |
|---|---|---|
| `train_all_14.py` | 18.23 | ALL_14(128x64) 5折训练 |
| `train_full.py` / `train_full_ind.py` | 18.21-18.22 | 8行业 + 行业dummy实验 |
| `train_rand.py` | 18.24 | RAND 80/20 训练 |
| `arch_gpu.py` | 18.33 | 架构扫描: 128x64 vs 256x128 vs 256x128x64 |
| `reg_arch.py` | 18.34 | 高惩罚宽/深架构: drop0.4-0.5, wd0.01-0.02 |
| `arch_scan.py` / `arch_small.py` / `arch_fast.py` / `arch_quick.py` / `arch_compare.py` | 18.28 | 早期架构消融（子采样版） |
| `final_3layer.py` / `triple_layer.py` | — | 三层 MLP 尝试 |
| `grokking_test.py` / `grok_final.py` | — | Grokking 实验 |

### Cartography (数据制图)

| 文件 | 对应日志 | 用处 |
|---|---|---|
| `cartography.py` | 18.29 | 4行业制图 + 去Hard重训 (v1) |
| `cart_v2.py` | 18.29 | 同上 v2 (修复版) |
| `cart_all.py` / `cart_all_full.py` / `cart_all_gpu.py` / `cart_all_5fold.py` | 18.30-18.31 | 全行业制图, RAND→5折→GPU |
| `cart_reg.py` | 18.35 | CART+REG 叠加（失败） |

### 混合模型与共识

| 文件 | 对应日志 | 用处 |
|---|---|---|
| `lr_rf_all.py` | 18.32 | LR + RF 全行业5折 |
| `lr_rf_weight.py` / `lr_screen.py` | — | LR 权重实验 |
| `consensus_full.py` | — | 早期共识买入 |
| `q_consensus.py` | — | 季度共识（失败） |
| `mlp_vs_baseline.py` / `mlp_vs_rf.py` | — | MLP vs 传统模型对比 |
| `xgb_search.py` | — | XGBoost 尝试 |
| `penalty_compare.py` | 18.37 | w=0.2 惩罚（失败） |
| `w02_test.py` | 18.37 | w=0.2 两版测试 |
| `drop_ic.py` | 18.28 | interest_coverage 消融 |

### 评估与报告

| 文件 | 用处 |
|---|---|
| `classification_eval.py` / `eval_all14.py` / `eval_full.py` / `eval_rand.py` / `eval_only.py` / `eval_mae.py` | 各阶段评估脚本 |
| `eval_cart_all.py` / `compare_cart.py` / `ind_cart.py` | CART 评估对比 |
| `all12_cm.py` | 早期混淆矩阵汇总 |
| `by_industry.py` / `ind_year.py` | 分行业/逐年分析 |
| `five_class.py` / `five_class_compare.py` / `five_detail.py` / `five_penalty_correct.py` | 五分类实验 |
| `check_d14_8.py` / `retrain_d14_8.py` | 8行业验证 |
| `top5_pick.py` / `sell_strategy.py` / `strategy_backtest.py` / `rolling_cv.py` | 选股策略/回测 |
| `two_year_eval.py` / `quarterly.py` / `sp_on_all.py` | 时域评估 |
| `no_ind_test.py` | 无行业信号测试 |
| `d13_test.py` | 13维消融 |
| `ab_test.py` | A/B 测试框架 |
| `report.md` | 早期报告草稿 |
| `correlation_analysis.py` | 特征相关性分析 |
| `detailed_report.py` / `final_compare.py` / `gen_appendix.py` / `gen_appendix2.py` | 报告生成 |
| `run_final.py` / `full_pipeline.py` / `auto_pipeline.py` / `two_stage.py` | 端到端流水线 |

---

## 权重文件映射

### 全行业5折验证权重（history_data/models/）

| 权重名 | 日志 | 模型 | 说明 |
|---|---|---|---|
| `ALL_14_{2020-2024}.pt` | 18.23 | 128x64 MLP | 全行业5折基准，train_all_14.py |
| `CART_ALL_{2020-2024}.pt` | 18.31 | 128x64 MLP (去Hard76%) | 制图学去Hard后重训，GPU |
| `MLP0_5_{2020-2024}.pt` | 18.12 | 128x64 MLP | 4行业5折，早期最优 |
| `MLP0_25_{2020-2024}.pt` / `MLP0_35_{2020-2024}.pt` / `MLP1_0_{2020-2024}.pt` | 18.25-18.26 | 128x64 MLP | 4行业权重消融 |
| `ARCH_128x64_{2020-2024}.pt` | 18.33 | 128x64 MLP | 架构扫描基准 |
| `ARCH_256x128_{2020-2024}.pt` | 18.33 | 256x128 MLP (原惩罚) | 架构扫描，年+33.2% |
| `ARCH_256x128x64_{2020-2024}.pt` | 18.33 | 256x128x64 MLP (原惩罚) | 过拟合，+29.2% |
| `REG_256x128_d04_wd01_{2020-2024}.pt` | 18.34 | 256x128 MLP (高惩罚) | drop0.4, wd0.01, +44.3% |
| `REG_256x128x64_d05_wd02_{2020-2024}.pt` | 18.34 | 256x128x64 MLP (高惩罚) | drop0.5, wd0.02, **+46.9% 单模型最强** |
| `RF_ALL_{2020-2024}.pkl` | 18.32 | Random Forest (100树) | 三轮数据通吃, +40.8% |
| `LR_ALL_{2020-2024}.pkl` | 18.32 | Logistic Regression | 几乎不买，弃用 |
| `W02_128x64_{2020-2024}.pt` / `W02_REG256x128x64_{2020-2024}.pt` | 18.37 | w=0.2版 | 完全锁死，失败 |

### 4行业权重

| 权重名 | 日志 | 说明 |
|---|---|---|
| `CART_{2020-2024}.pt` | 18.29 | 4行业去Hard, +24.5% |
| `Q_MLP0_5_{2020-2024}.pt` | 18.27 | 季度预测尝试 |
| `A_[2,2,1,1,0.5]_{2020-2024}.pt` / `A_[3,3,1,1,0.5]_{2020-2024}.pt` | — | 早期惩罚矩阵实验 |

### 8行业权重

| 权重名 | 日志 | 说明 |
|---|---|---|
| `FULL_{2020-2024}.pt` | 18.21 | 8行业, 无行业dummy |
| `FULL_IND_{2020-2024}.pt` | 18.22 | 8行业, 含行业dummy |
| `D13_{2020-2024}.pt` / `D14_4_{2020-2024}.pt` / `D14_8_{2020-2024}.pt` | 18.21 | 维度消融 |

### S1/S2 掩码权重

| 权重名 | 日志 | 说明 |
|---|---|---|
| `S1_PASS_{2020-2024}.pt` | 18.13 | S1通过样本训练 |
| `S1_FAIL_{2020-2024}.pt` | 18.13 | S1未通过样本训练 |
| `S1_PASS_w0.5_{2020-2024}.pt` | 18.13 | S1+0.5惩罚版 |
| `S2_{2020-2024}.pt` / `S2_only_{2020-2024}.pt` / `S2_repair_{2020-2024}.pt` | 18.15 | S2 掩码各版本 |
| `S3_feat_{2020-2024}.pt` | 18.17 | S3 毛利率特征版 |

### 部署权重

| 权重名 | 说明 |
|---|---|
| `REG_DEPLOY.pt` / `RF_DEPLOY.pkl` | 全量历史训练，最新季预测用 |
| `cart_cache_{2020-2024}.pkl` | Cartography缓存，制图分区用 |
| `CART_ALL.pt` | RAND80/20 单权重版（早期，后被5折淘汰） |

### RAND 80/20 权重

| 权重名 | 日志 | 说明 |
|---|---|---|
| `RAND_ALL.pt` | 18.24 | 全行业RAND, +52.1% |
| `RAND_4.pt` | 18.24 | 4行业RAND, +58.8% |

---

## 数据集

| 数据集 | 文件 | 行业 | 年份 | 条数 |
|---|---|---|---|---|
| 4行业 | `training_final.csv` | 钢铁/白酒/银行/游戏 | 2019-2024 | ~3.3k |
| 8行业 | `training_full.csv` | 8类 | 2019-2024 | ~25k |
| 全行业 | `training_all.csv` | 6大类 | 2019-2024 | ~87k |
| 扩展版 | `training_extended.csv` | 6大类 | 2019-2026Q1 | ~112k |

数据源：AkShare `stock_yjbb_em` + `stock_zcfz_em`，前复权 `stock_zh_a_hist`。

## 依赖

pandas numpy scikit-learn torch akshare (详见 requirements.txt)

## 部署策略

1. REG∩RF 共识优先（高胜率，少而精，223只+57.0%）
2. 共识为空时 REG 单独兜底（覆盖率，593只+46.9%）
3. 全部通过 S1∩S3∩S4 财务健康掩码

## License

MIT
