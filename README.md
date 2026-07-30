# Quant Trader · 多品种期货自适应量化交易系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Dashboard 示例](https://jarrich-starmap.github.io/quant-dashboard/example.png)

一套面向商品期货（沪金 AU / 沪银 AG / 原油 SC）与股指期货（中证 500 IC / 中证 1000 IM）的**四层流水线自适应量化交易系统**。核心特征：

- **四层流水线**：`Data → Signal → Execution → Feedback`，各层解耦、可独立演进。
- **三路信号融合**：技术面（RSI/MACD/BB/Momentum）+ 情绪面 + 波动率面，经 **softmax(EWMA) 动态权重** 融合，EWMA 在线学习各路信号的历史胜率。
- **波动率模块**：单一 Yang-Zhang 估计器 + EWMA z-score 状态判定 + 连续 sigmoid 风险控制；仅启用 **Expansion-follow（扩张跟随）** 与 **Extreme-reversal（极端反转）** 两类模式，全部 config-gated。
- **持仓台账自愈**：以 `trades` 表为权威账本，每次运行先对账（reconcile），自动清理孤儿持仓，保证与冗余台账 `ewma_state` 一致。
- **可视化 Dashboard**：FastAPI 后端 + Chart.js 前端，展示信号得分时序、持仓、累计盈亏等。

---

## ⚠️ 免责声明

本项目**仅供学习与研究**，不构成任何投资建议。量化交易存在重大亏损风险，所有代码默认运行于 **`mode: simulation`（模拟）**。实盘交易需自行评估风险并承担全部责任。作者与贡献者对使用本系统产生的任何盈亏不承担责任。

---

## 目录结构

```
quant-dashboard/
├── main.py                  # 主入口：编排四层流水线，由 systemd timer 每分钟拉起
├── config.example.yaml      # 配置示例（复制为 config.yaml 后填写真实值）
├── requirements.txt         # Python 依赖（已锁定版本）
├── check_ts.py / test_save.py  # 调试/校验小工具
├── data/                    # Data 层：行情抓取、合约适配、情绪读取
│   ├── fetcher.py           # K 线/行情获取
│   ├── adapter.py           # 合约代码适配、换月检测
│   └── sentiment.py         # 情绪数据加载
├── signal/                  # Signal 层
│   ├── indicators.py        # RSI/MACD/BB/Momentum 指标
│   └── scorer.py            # 三路 softmax 融合 → 综合得分
├── volatility/              # 波动率信号源
│   ├── estimators.py        # Yang-Zhang 波动率估计
│   ├── avi.py               # 波动率状态（EWMA z-score）
│   ├── signal.py            # Expansion / Reversal 信号 → vol_score
│   └── integration.py       # 与 Signal 层集成
├── executor/                # Execution 层
│   ├── position.py          # 仓位计算
│   └── trader.py            # 开/平仓、滑点、止损
├── feedback/                # Feedback 层
│   ├── ewma_tracker.py      # EWMA 权重在线学习
│   └── optimizer.py         # 参数/权重优化
├── db/
│   └── models.py            # SQLite（WAL）数据模型：trades / ewma_state
├── notify/
│   └── wecom_notifier.py    # 企业微信「交易执行前」通知（支持应用消息/群机器人）
├── dashboard/
│   ├── server.py            # FastAPI 后端（:8090）
│   └── dashboard.sh         # 启动脚本
└── docs/
    └── quant-trader.html    # 系统设计文档（架构图 + 参数表）
```

> 注：各子模块以**顶层包**形式导入（`from data.adapter`、`from db.models`、`from volatility.integration` 等），运行时应将**仓库根目录**置于 `PYTHONPATH`（直接 `python main.py` 即满足）。

---

## 系统架构

[![系统架构](https://jarrich-starmap.github.io/quant-dashboard/architecture.svg)](https://jarrich-starmap.github.io/quant-dashboard/quant-trader.html)

> 点击图片查看完整设计文档（架构图、参数表与计算公式）。

---

## 快速开始

### 1. 环境

- Python 3.12+
- 推荐虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

复制示例配置并填写真实值：

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml：品种参数、手续费、情绪数据路径等
```

| 环境变量 | 对应配置项 | 说明 |
|---|---|---|
| `WECOM_CORP_ID` | `notify.wecom.corp_id` | 企业微信 corp id（应用消息模式） |
| `WECOM_CORP_SECRET` | `notify.wecom.corp_secret` | 应用 secret |
| `WECOM_AGENT_ID` | `notify.wecom.agent_id` | 应用 agentid |
| `WECOM_WEBHOOK_KEY` | `notify.wecom.webhook_key` | 群机器人 webhook key |

> `notify.wecom_notifier.py` 优先读取环境变量，其次读取 `config.yaml`。**代码库中不含任何真实密钥**——`config.example.yaml` 中的 `webhook_key` 为 `${WECOM_WEBHOOK_KEY}` 占位符。

### 3. 运行（模拟模式）

```bash
python main.py            # 默认 mode: simulation，不会真实下单
```

### 4. 启动 Dashboard

```bash
cd dashboard
bash dashboard.sh          # 启动 FastAPI（默认 :8090）
```

---

## 信号与融合

综合得分由三路经 softmax 动态加权融合：

```
final_score = softmax([ewma_tech, ewma_sent, ewma_vol], τ) · [tech_score, sent_score, vol_score]
```

- **技术面**：RSI / MACD / 布林带 / 动量，等权 `technical_weights` 合成 `tech_score ∈ [-1, 1]`。
- **情绪面**：外部 JSON 提供多空计数；超过 `max_age_hours` 视为过期，`α` 强制为 1（纯技术驱动）。
- **波动率面**：`vol_score = direction × (expansion_conf − reversal_conf)`，受强度系数限制，实盘激活率约 2%。

EWMA 对各路信号权重做在线学习（方向 ±1、PnL 用 tanh 映射为 reward），实现「历史胜率高者权重更大」的自适应。

---

## 风险控制

- **三级止损**：硬止损 / 保本触发 / 移动止损（均为小数比例，如 `0.06` = 6%）。
- **波动率连续惩罚**：以 z-score 经 sigmoid 连续映射仓位乘数、滑点惩罚、止损距离，零 `if-elif` 突变。
- **冷却恢复**：连续错误达 `max_errors_before_cooldown` 进入冷却，按 `recovery_position_scale` 逐步恢复。

---

## 交易品种

| 品种 | 类型 | 合约乘数 | 手续费模式 | 备注 |
|---|---|---|---|---|
| AU 沪金 | 贵金属 | 1000 | 固定 10 元/手 | 已开启 `dynamic_stop` / `adjust_reward` |
| AG 沪银 | 贵金属 | 15 | 比率 万分之 0.5 | |
| SC 原油 | 能源 | 1000 | 固定 20 元/手 | 波动率更高，差异化参数 |
| IC 中证500 | 股指 | 200 | 比率 万分之 0.23 | |
| IM 中证1000 | 股指 | 200 | 比率 万分之 0.23 | |

---

## 部署（生产参考）

生产环境通过 **systemd** 调度：

- `quant-trader.timer`：每分钟触发一次 `main.py`。
- `quant-dashboard.service`：常驻 `dashboard/server.py`（`uvicorn :8090`），前接 nginx 反代。

数据库使用 SQLite（WAL 模式），并加 `busy_timeout` 应对并发写。详细架构、参数含义与计算公式见 [`docs/quant-trader.html`](docs/quant-trader.html)。

---

## 开发状态

- [x] 四层流水线、三路融合、EWMA 自适应
- [x] 波动率模块 + 连续 sigmoid 风控
- [x] 持仓台账对账自愈
- [x] Dashboard 可视化

---

## License

[MIT](LICENSE) © 2026 Jarrich-Starmap
