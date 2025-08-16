import os, time, queue, threading, subprocess, sys, json, pathlib, random
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# --- App Setup ---
st.set_page_config(page_title="AI Forex Trading Bot — UI", layout="wide")
st.title("AI Forex Trading Bot")
st.caption("Demo & Live control panel")

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "ui_runtime.log"

# State
if "running" not in st.session_state:
    st.session_state.running = False
if "mode" not in st.session_state:
    st.session_state.mode = "Demo"
if "log_q" not in st.session_state:
    st.session_state.log_q = queue.Queue()
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time","price"])

stop_event = threading.Event()

# --- Logging helper ---
def log(msg: str):
    line = f"[{datetime.utcnow().isoformat()}] {msg}"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        st.session_state.log_q.put_nowait(line)
    except queue.Full:
        pass

# --- Demo engine (no MT5 required) ---
def demo_engine(stop: threading.Event, symbol: str, profit_target: float, stop_loss: float):
    price = 100.0
    log(f"DEMO: starting demo engine for {symbol} (TP={profit_target}, SL={stop_loss})")
    while not stop.is_set():
        # Random walk
        price += random.uniform(-0.25, 0.25)
        ts = datetime.utcnow()
        new_row = {"time": ts, "price": round(price, 4)}
        st.session_state.data.loc[len(st.session_state.data)] = new_row

        # Fake signals/logs
        if random.random() < 0.06:
            direction = random.choice(["LONG","SHORT"])
            conf = round(random.uniform(0.35, 0.9), 2)
            log(f"DEMO: signal {direction} @ {price} (confidence={conf})")

        if len(st.session_state.data) > 600:
            st.session_state.data = st.session_state.data.iloc[-600:]

        time.sleep(0.5)
    log("DEMO: engine stopped")

# --- Live engine (tries to run your bot) ---
def live_engine(stop: threading.Event):
    """
    Strategy:
      1) Try importing src.main and call a 'run' or 'main' function in a thread-safe way.
      2) If not found, fallback to running 'python -m src.main' as a subprocess and stream its stdout to logs.
    """
    log("LIVE: starting live engine")
    try:
        sys.path.insert(0, str(ROOT))
        import src.main as main  # type: ignore

        fn = None
        for cand in ("run", "main", "start"):
            if hasattr(main, cand) and callable(getattr(main, cand)):
                fn = getattr(main, cand)
                break
        if fn is not None:
            log(f"LIVE: calling src.main.{fn.__name__}()")
            try:
                fn()  # assumes your main loop blocks; stop via your own controls
            except Exception as e:
                log(f"LIVE: error in src.main.{fn.__name__}: {e}")
        else:
            # Fallback: subprocess
            cmd = [sys.executable, "-m", "src.main"]
            log(f"LIVE: launching subprocess: {' '.join(cmd)}")
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT) as p:
                while not stop.is_set():
                    line = p.stdout.readline()
                    if not line:
                        if p.poll() is not None:
                            break
                        time.sleep(0.1)
                        continue
                    log(line.strip())
                try:
                    p.terminate()
                except Exception:
                    pass
    except Exception as e:
        log(f"LIVE: failed to start — {e}")
    log("LIVE: engine stopped")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    st.session_state.mode = st.selectbox("Mode", ["Demo", "Live"], index=0)
    symbol = st.text_input("Symbol", value="EURUSD")
    profit_target = st.number_input("Profit Target (%)", value=0.20, step=0.05, format="%.2f")
    stop_loss = st.number_input("Stop Loss (%)", value=0.75, step=0.05, format="%.2f")
    colA, colB = st.columns(2)
    start = colA.button("▶ Start", use_container_width=True, disabled=st.session_state.running)
    stopb = colB.button("■ Stop", use_container_width=True, disabled=not st.session_state.running)
    st.divider()
    st.markdown("**Notes**")
    st.markdown("- Demo works anywhere.\n- Live requires Windows + MetaTrader5 installed + your bot configured.")

# --- Start/Stop handlers ---
if start and not st.session_state.running:
    LOG_FILE.write_text("")  # clear
    stop_event.clear()
    st.session_state.data = st.session_state.data.iloc[0:0]
    if st.session_state.mode == "Demo":
        t = threading.Thread(target=demo_engine, args=(stop_event, symbol, profit_target, stop_loss), daemon=True)
    else:
        t = threading.Thread(target=live_engine, args=(stop_event,), daemon=True)
    t.start()
    st.session_state.running = True
    log("UI: start clicked")

if stopb and st.session_state.running:
    stop_event.set()
    st.session_state.running = False
    log("UI: stop clicked")

# --- Layout: chart + logs ---
lc, rc = st.columns([2,1], vertical_alignment="top")
with lc:
    st.subheader("Price (Demo stream or Live logs-derived)")
    if st.session_state.data.empty:
        st.info("No data yet. Start the engine to see activity.")
    else:
        df = st.session_state.data.copy()
        df["t"] = pd.to_datetime(df["time"])
        fig = px.line(df, x="t", y="price", title=None)
        st.plotly_chart(fig, use_container_width=True)
with rc:
    st.subheader("Logs")
    # Tail logs & queue items
    log_box = st.empty()
    last = []
    def tail_file(path, n=200):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            return lines[-n:]
        except Exception:
            return []
    # refresh region
    tail = tail_file(LOG_FILE, 200)
    while True:
        try:
            line = st.session_state.log_q.get_nowait()
            tail.append(line)
        except queue.Empty:
            break
    last = tail[-300:]
    log_box.code("\n".join(last) if last else "(no logs yet)")
    st.caption(LOG_FILE.as_posix())

st.caption("© 2025 — Portfolio UI for demo purposes")
