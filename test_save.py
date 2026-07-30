from db.models import load_ewma_state, save_ewma_state, get_conn

# Test save and load
save_ewma_state(
    symbol="TEST",
    ewma_rsi=0, ewma_macd=0, ewma_bb=0, ewma_mom=0,
    ewma_tech=0, ewma_sent=0, ewma_sent_frozen=0,
    alpha=0.5, error_count=0, cool_remaining=0,
    recovery_correct_streak=0, warmup_count=20,
    pos_direction="LONG", pos_entry=100.0, pos_size=1.0,
)
state = load_ewma_state("TEST")
print("pos_direction:", repr(state.get("pos_direction")))
print("pos_entry:", state.get("pos_entry"))
print("pos_size:", state.get("pos_size"))

# Cleanup
c = get_conn()
c.execute("DELETE FROM ewma_state WHERE symbol='TEST'")
c.commit()
print("cleaned")
