"""
企业微信「交易执行前」通知模块
================================

设计目标
--------
在每笔交易真正下单之前，调用企业微信 API 把交易计划推送给指定成员（或群），
做到「事前预警」。关键约束：

1. 不阻塞交易：网络慢 / API 故障 / 凭证缺失 都只记录日志，绝不影响主交易流程。
2. 超时可控：默认 5s，到时直接放弃本次通知。
3. Token 跨进程复用：systemd timer 每分钟拉起一个全新 Python 进程，
   access_token 有效期 7200s，用本地文件缓存避免重复 gettoken 触发频控。
4. 零外部依赖：仅用标准库（urllib / json / threading / logging）。

支持两种推送通道（config + 环境变量自动选择）
------------------------------------------------
A. 应用消息（推荐，可精确推送给某个/某些成员）
   corpid + corpsecret + agentid + touser
   接口：/cgi-bin/message/send  (msgtype=markdown)
B. 群机器人 Webhook（最简单，推送到群聊）
   webhook_key
   接口：/cgi-bin/webhook/send?key=xxx  (msgtype=markdown)

集成位置（在你的 main.py 中，调用 trader 开仓之前）：
   from notify.wecom_notifier import notify_before_trade
   notify_before_trade(config, plan, block=True)   # 同步、超时不影响交易
   trader.open_position(...)                        # 之后才真正下单
"""

import json
import os
import time
import threading
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger("wecom_notify")

WECHAT_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"
# token 缓存文件放在项目根目录（notify/ 的上一级），所有 cron 进程共享
_TOKEN_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".wecom_token_cache.json",
)
_TOKEN_LOCK = threading.Lock()


class WeComNotifier:
    def __init__(self, config: dict):
        wc = (config or {}).get("notify", {}).get("wecom", {})

        # 敏感信息优先读环境变量，其次读 config.yaml（便于把密钥留在部署环境而非代码库）
        self.corp_id = os.environ.get("WECOM_CORP_ID") or wc.get("corp_id")
        self.corp_secret = os.environ.get("WECOM_CORP_SECRET") or wc.get("corp_secret")
        self.agent_id = os.environ.get("WECOM_AGENT_ID") or wc.get("agent_id")
        self.touser = wc.get("touser", "@all")
        self.webhook_key = os.environ.get("WECOM_WEBHOOK_KEY") or wc.get("webhook_key")
        self.timeout = int(wc.get("timeout", 5))
        self.enabled = bool(wc.get("enabled", True))

        if not self.enabled:
            logger.info("[WeCom] 通知已禁用 (notify.wecom.enabled=false)")
            return

        # 通道自动选择：有 webhook_key 且未配置 corpid → webhook；否则 app
        forced = wc.get("mode")
        if forced == "webhook" or (self.webhook_key and not self.corp_id):
            self.mode = "webhook"
        elif forced == "app" or self.corp_id:
            self.mode = "app"
        else:
            self.mode = "webhook" if self.webhook_key else "app"

        if self.mode == "app" and not (self.corp_id and self.corp_secret and self.agent_id):
            logger.warning("[WeCom] 应用消息模式缺少 corpid/corpsecret/agentid，通知将跳过")
            self.enabled = False
        if self.mode == "webhook" and not self.webhook_key:
            logger.warning("[WeCom] Webhook 模式缺少 webhook_key，通知将跳过")
            self.enabled = False

        if self.enabled:
            logger.info("[WeCom] 通知已启用，通道=%s，接收人=%s", self.mode, self.touser)

    # ----------------------------- token 管理 -----------------------------
    def _load_token(self) -> dict:
        try:
            with open(_TOKEN_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_token(self, token: str, expires_in: int):
        try:
            with open(_TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "access_token": token,
                        # 提前 5 分钟过期，避免边界失效
                        "expire_at": int(time.time()) + max(expires_in, 0) - 300,
                    },
                    f,
                )
        except Exception as e:  # 缓存失败不影响交易
            logger.warning("[WeCom] 保存 token 缓存失败: %s", e)

    def _get_token(self):
        cached = self._load_token()
        if cached.get("access_token") and cached.get("expire_at", 0) > time.time():
            return cached["access_token"]
        url = f"{WECHAT_API_BASE}/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"
        with _TOKEN_LOCK:  # 进程内防并发；跨进程的极小概率重复 gettoken 无害
            cached = self._load_token()
            if cached.get("access_token") and cached.get("expire_at", 0) > time.time():
                return cached["access_token"]
            data = self._http_get(url)
            if data.get("errcode") == 0:
                self._save_token(data["access_token"], int(data.get("expires_in", 7200)))
                return data["access_token"]
            logger.error("[WeCom] 获取 access_token 失败: %s", data)
            return None

    # ----------------------------- HTTP 原语 -----------------------------
    def _http_get(self, url: str) -> dict:
        req = Request(url, headers={"User-Agent": "quant-notify/1.0"})
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "quant-notify/1.0"},
        )
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ----------------------------- 消息构建 -----------------------------
    def build_message(self, plan: dict) -> str:
        """把交易计划渲染成企业微信 markdown 文本（<=4096 字节）。"""
        event = plan.get("event", "OPEN")
        icon = {"OPEN": "📈", "CLOSE": "📉", "STOP": "⚠️", "FORCE_CLOSE": "🛑"}.get(event, "🔔")
        event_cn = {
            "OPEN": "开仓预警",
            "CLOSE": "平仓通知",
            "STOP": "止损/止盈通知",
            "FORCE_CLOSE": "时段结束强平",
        }.get(event, event)
        dir_cn = {"LONG": "做多", "SHORT": "做空", "NEUTRAL": "中性"}.get(
            plan.get("direction"), plan.get("direction", "")
        )
        lines = [
            f"{icon} **交易通知 · {event_cn}**",
            f">合约：<font color=\"info\">{plan.get('symbol', '')}</font>",
            f">方向：{dir_cn} {plan.get('direction', '')}",
        ]
        if plan.get("entry_price") is not None:
            lines.append(f">开仓价格：{plan['entry_price']}")
        if plan.get("exit_price") is not None:
            lines.append(f">平仓价格：{plan['exit_price']}")
        if plan.get("position_size") is not None:
            lines.append(f">仓位：{plan['position_size']} 手")
        if plan.get("final_score") is not None:
            lines.append(f">综合得分：{plan['final_score']} ｜ Alpha：{plan.get('alpha')}")
        if plan.get("vol_score") is not None:
            lines.append(f">波动得分：{plan['vol_score']}")
        if plan.get("pnl") is not None:
            pnl = plan["pnl"]
            arrow = "▲" if pnl >= 0 else "▼"
            lines.append(f">盈亏：{arrow} {pnl:+.2f}")
        lines.append(f">时间：{plan.get('cycle_time', '')}")
        return "\n".join(lines)

    # ----------------------------- 对外发送 -----------------------------
    def notify(self, plan: dict) -> bool:
        if not self.enabled:
            return False
        try:
            content = self.build_message(plan)
            if self.mode == "app":
                token = self._get_token()
                if not token:
                    return False
                url = f"{WECHAT_API_BASE}/message/send?access_token={token}"
                payload = {
                    "touser": self.touser,
                    "msgtype": "markdown",
                    "agentid": int(self.agent_id),
                    "markdown": {"content": content},
                }
            else:
                url = f"{WECHAT_API_BASE}/webhook/send?key={self.webhook_key}"
                payload = {"msgtype": "markdown", "markdown": {"content": content}}

            res = self._http_post(url, payload)
            if res.get("errcode") == 0:
                logger.info("[WeCom] 通知发送成功: %s/%s", plan.get("symbol"), plan.get("event"))
                return True
            logger.error("[WeCom] 通知被拒: %s", res)
            return False
        except (URLError, HTTPError, TimeoutError) as e:
            logger.error("[WeCom] 网络异常，跳过本次通知（不影响交易）: %s", e)
            return False
        except Exception as e:  # 兜底，任何意外都不允许影响交易主流程
            logger.exception("[WeCom] 未知异常，跳过本次通知: %s", e)
            return False


# 进程内单例，避免每次调用重复构造
_notifier_instance = None
_notifier_lock = threading.Lock()


def get_notifier(config: dict) -> WeComNotifier:
    global _notifier_instance
    if _notifier_instance is None:
        with _notifier_lock:
            if _notifier_instance is None:
                _notifier_instance = WeComNotifier(config)
    return _notifier_instance


def notify_before_trade(config: dict, plan: dict, block: bool = True):
    """
    在交易执行前调用。

    :param config: 与 main.py 一致的全局配置 dict（含 notify.wecom 段）
    :param plan:   交易计划 dict，字段见 build_message
    :param block:  True=同步发送（保证「交易前」语义，超时/失败不影响交易）；
                   False=后台线程异步发送（极致低延迟，但可能略晚于下单瞬间）
    """
    notifier = get_notifier(config)
    if not notifier.enabled:
        return
    if block:
        notifier.notify(plan)
    else:
        threading.Thread(target=notifier.notify, args=(plan,), daemon=True).start()
