"""情绪 JSON 读取 + 过期处理。"""

import json
from datetime import datetime
import os
import time
from pathlib import Path
from typing import Optional


def load_sentiment(json_path: str, max_age_hours: float = 2.0, symbols: list = None) -> tuple[dict, bool]:
    """
    读取情绪 JSON 文件。
    返回: (scores_dict, is_valid)
      - scores_dict: 每个 config 中品种一张分数字典（含 IC/IM）
      - is_valid: True 表示 JSON 有效，False 表示缺失/过期
    情绪得分 = bullish - bearish ∈ [-1, 1]
    """
    # 支持 {DAY} 动态日期占位符
    json_path = json_path.replace('{DAY}', datetime.now().strftime('%Y-%m-%d'))

    # 品种集合由调用方（config['symbols']）驱动；缺省向后兼容 AU/AG/SC/IC/IM
    if symbols is None:
        symbols = ["AU", "AG", "SC", "IC", "IM"]
    
    if not json_path or not os.path.exists(json_path):
        return _empty_scores(symbols), False

    # 检查文件修改时间
    mtime = os.path.getmtime(json_path)
    age_hours = (time.time() - mtime) / 3600.0
    if age_hours > max_age_hours:
        return _empty_scores(symbols), False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return _empty_scores(symbols), False

    scores = {}
    for symbol in symbols:
        entry = data.get(symbol, {})
        bullish = float(entry.get("bullish", 0))
        bearish = float(entry.get("bearish", 0))
        total = bullish + bearish
        scores[symbol] = (bullish - bearish) / total if total > 0 else 0.0

    return scores, True


def _empty_scores(symbols: list = None) -> dict:
    if symbols is None:
        symbols = ["AU", "AG", "SC", "IC", "IM"]
    return {s: 0.0 for s in symbols}
