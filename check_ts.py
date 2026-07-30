from db.models import get_conn
from datetime import datetime
c = get_conn()
for sym in ["AU", "AG", "SC"]:
    r = c.execute("SELECT timestamp, close FROM kline_cache WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (sym,)).fetchone()
    if r:
        ts = r["timestamp"]
        try:
            age = (datetime.now() - datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
        except Exception as e:
            age = f"err: {e}"
        print(f"{sym}: ts={ts}, age={age}, close={r['close']}")
print("now:", datetime.now())
