"""
Hawkes 過程微觀突破驗證器
作者：HFT 開發工程師
說明：使用 Streamlit + Plotly 互動式分析 Hawkes 過程在逐筆成交中的突破訊號
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# ──────────────────────────────────────────────
# 頁面基本設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hawkes 過程微觀突破驗證器",
    page_icon="⚡",
    layout="wide",
)

# 自訂 CSS 樣式
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1.5rem; }
    .metric-box-breakthrough {
        background: linear-gradient(135deg, #1a3a1a, #0d2b0d);
        border: 2px solid #2ea043;
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        color: #3fb950;
        letter-spacing: 0.04em;
        box-shadow: 0 0 20px rgba(46, 160, 67, 0.4);
    }
    .metric-box-fakeout {
        background: linear-gradient(135deg, #3a1a1a, #2b0d0d);
        border: 2px solid #da3633;
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        color: #f85149;
        letter-spacing: 0.04em;
        box-shadow: 0 0 20px rgba(218, 54, 51, 0.4);
    }
    .metric-box-neutral {
        background: linear-gradient(135deg, #1c2128, #161b22);
        border: 2px solid #484f58;
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        color: #8b949e;
    }
    .param-section {
        background-color: #161b22;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 側邊欄：Hawkes 參數設定
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Hawkes 參數設定")
st.sidebar.markdown("---")

st.sidebar.markdown("**📐 基礎強度（μ）**")
mu = st.sidebar.slider(
    "μ — 背景到達率（events/ms）",
    min_value=0.001, max_value=0.05,
    value=0.01, step=0.001,
    format="%.3f",
    help="無條件事件到達率，與市場底部雜訊對應"
)

st.sidebar.markdown("**⚡ 激發跳躍值（α）**")
alpha = st.sidebar.slider(
    "α — 每筆成交的激發強度",
    min_value=0.1, max_value=0.99,
    value=0.6, step=0.01,
    format="%.2f",
    help="每筆成交對後續成交的激發幅度（需 α < β 保持穩定）"
)

st.sidebar.markdown("**🔻 衰減率（β）**")
beta = st.sidebar.slider(
    "β — 激發效應衰減速率",
    min_value=0.5, max_value=5.0,
    value=1.5, step=0.1,
    format="%.1f",
    help="激發效應隨時間衰減的速率"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📏 中線價格設定**")
midpoint_price = st.sidebar.number_input(
    "中線（Mid-point）價格",
    min_value=100.0, max_value=100000.0,
    value=100.0, step=0.5,
    format="%.2f",
    help="定義突破判斷的關鍵中線價位"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🎲 模擬控制**")
seed_val = st.sidebar.number_input("隨機種子（Seed）", min_value=0, max_value=9999, value=42, step=1)
n_ticks = st.sidebar.slider("模擬 Tick 數量", min_value=100, max_value=800, value=300, step=50)

# 再生成按鈕
if st.sidebar.button("🔄 重新生成模擬數據", use_container_width=True):
    st.session_state["regen"] = st.session_state.get("regen", 0) + 1

# ──────────────────────────────────────────────
# 假數據生成器：模擬「測試中線」情境
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_mock_tick_data(n: int, mid: float, seed: int, regen: int = 0) -> pd.DataFrame:
    """
    生成模擬逐筆成交數據（Tick Data）。
    模擬情境：價格在中線附近震盪後，發動突破。
    """
    rng = np.random.default_rng(seed + regen)

    # 時間戳：毫秒級，間距 50~500ms
    intervals = rng.integers(50, 500, size=n)
    timestamps = np.cumsum(intervals).astype(np.int64)

    # 價格路徑設計：前段在中線下方，後段突破
    prices = []
    base = mid - 1.5  # 初始價格低於中線

    # 第一段（60%）：在中線下方震盪
    seg1 = int(n * 0.60)
    for i in range(seg1):
        noise = rng.normal(0, 0.08)
        drift = 0.015 * np.sin(i / 20)  # 模擬盤整
        base = base + drift + noise
        base = np.clip(base, mid - 3.0, mid - 0.02)  # 被中線壓制
        prices.append(round(base, 2))

    # 第二段（15%）：假突破，量小
    seg2 = int(n * 0.15)
    for i in range(seg2):
        noise = rng.normal(0.02, 0.05)
        base = base + noise
        prices.append(round(base, 2))

    # 第三段（25%）：真實突破，成交量放大
    seg3 = n - seg1 - seg2
    for i in range(seg3):
        noise = rng.normal(0.06, 0.12)
        base = base + noise
        prices.append(round(base, 2))

    prices = np.array(prices)

    # 成交量：突破段明顯放大
    quantity = rng.integers(10, 100, size=n)
    # 真實突破段成交量倍增
    quantity[seg1 + seg2:] = rng.integers(200, 800, size=seg3)

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Price": prices,
        "Quantity": quantity,
    })
    return df, seg1, seg2  # 回傳分段資訊供標記使用


# ──────────────────────────────────────────────
# Hawkes 過程條件強度函數 λ(t) 滾動計算
# ──────────────────────────────────────────────
def compute_hawkes_intensity(timestamps_ms: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """
    計算每個時間點的 Hawkes 條件強度 λ(t)。

    公式：
        λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))

    此處對每個事件點 t_k 計算左極限強度（包含前所有事件激發）。

    參數：
        timestamps_ms: 毫秒級時間戳陣列
        mu: 背景強度
        alpha: 激發跳躍值
        beta: 衰減率（單位：1/ms，內部轉換）
    """
    n = len(timestamps_ms)
    t = timestamps_ms / 1000.0  # 轉換為秒
    beta_sec = beta              # β 單位：1/s
    alpha_sec = alpha

    intensities = np.zeros(n)
    # 遞迴計算激發項（O(n) 效率）
    R = 0.0  # 遞推加速項
    intensities[0] = mu

    for k in range(1, n):
        dt = t[k] - t[k - 1]
        R = np.exp(-beta_sec * dt) * (1.0 + R)
        intensities[k] = mu + alpha_sec * R

    return intensities


# ──────────────────────────────────────────────
# 主程式：載入數據 & 計算強度
# ──────────────────────────────────────────────
regen_count = st.session_state.get("regen", 0)

with st.spinner("⚡ 生成模擬數據中..."):
    df, seg1, seg2 = generate_mock_tick_data(n_ticks, midpoint_price, seed_val, regen_count)
    df["Lambda"] = compute_hawkes_intensity(df["Timestamp"].values, mu, alpha, beta)

# 計算 λ(t) 歷史分位數（用於判斷強弱）
lambda_threshold_high = np.percentile(df["Lambda"].values, 80)   # 高位閾值
lambda_threshold_low  = np.percentile(df["Lambda"].values, 35)   # 低位閾值（「趴地」定義）
lambda_max = df["Lambda"].max()

# ──────────────────────────────────────────────
# 突破偵測邏輯
# ──────────────────────────────────────────────
last_price  = df["Price"].iloc[-1]
last_lambda = df["Lambda"].iloc[-1]
price_above_mid = last_price > midpoint_price

# 突破 + 強度超過 80% 分位 → 真實流動性突破
# 突破 + 強度低於 35% 分位 → 無量回測（假突破）
if price_above_mid and last_lambda >= lambda_threshold_high:
    signal = "real"
elif price_above_mid and last_lambda <= lambda_threshold_low:
    signal = "fake"
else:
    signal = "neutral"

# ──────────────────────────────────────────────
# 頁面主標題 & 警示訊號板
# ──────────────────────────────────────────────
st.markdown("# ⚡ Hawkes 過程微觀突破驗證器")
st.markdown("*高頻交易視角：用激發強度辨別真實突破與假突破*")

col_sig, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])

with col_sig:
    if signal == "real":
        st.markdown(
            '<div class="metric-box-breakthrough">🚀 真實流動性突破 ｜ REAL BREAKOUT</div>',
            unsafe_allow_html=True
        )
    elif signal == "fake":
        st.markdown(
            '<div class="metric-box-fakeout">⚠️ 無量回測（假突破）｜ FAKEOUT</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="metric-box-neutral">🔍 觀察中 — 尚未觸發突破訊號</div>',
            unsafe_allow_html=True
        )

with col_m1:
    st.metric("最新價格", f"{last_price:.2f}", f"{last_price - midpoint_price:+.2f} vs 中線")
with col_m2:
    st.metric("最新 λ(t)", f"{last_lambda:.4f}", f"高位閾值 {lambda_threshold_high:.4f}")
with col_m3:
    st.metric("λ 強度分位", f"{(last_lambda / lambda_max * 100):.1f}%", "相對歷史最高")

st.markdown("---")

# ──────────────────────────────────────────────
# 主圖表：上下連動 Plotly 雙圖
# ──────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,          # 連動縮放關鍵
    vertical_spacing=0.06,
    row_heights=[0.60, 0.40],
    subplot_titles=["📈 逐筆成交價格走勢 & 中線", "⚡ Hawkes 條件強度 λ(t)"]
)

# 顏色定義
COLOR_UP    = "#3fb950"
COLOR_DOWN  = "#f85149"
COLOR_MID   = "#d29922"
COLOR_LAMBD = "#58a6ff"
COLOR_THRESH_H = "#e3b341"
COLOR_THRESH_L = "#8b949e"
BG_COLOR    = "#0d1117"
PAPER_COLOR = "#0d1117"
GRID_COLOR  = "#21262d"

# ── 上圖：價格走勢 ──────────────────────────────
# 底色區塊：區分各段
fig.add_vrect(
    x0=df["Timestamp"].iloc[0], x1=df["Timestamp"].iloc[seg1 - 1],
    fillcolor="rgba(88, 166, 255, 0.04)", line_width=0, row=1, col=1
)
fig.add_vrect(
    x0=df["Timestamp"].iloc[seg1], x1=df["Timestamp"].iloc[seg1 + seg2 - 1],
    fillcolor="rgba(248, 81, 73, 0.06)", line_width=0, row=1, col=1
)
fig.add_vrect(
    x0=df["Timestamp"].iloc[seg1 + seg2], x1=df["Timestamp"].iloc[-1],
    fillcolor="rgba(63, 185, 80, 0.07)", line_width=0, row=1, col=1
)

# 成交量以圓點大小呈現
fig.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["Price"],
    mode="lines+markers",
    name="成交價",
    line=dict(color="#58a6ff", width=1.5),
    marker=dict(
        size=np.clip(df["Quantity"] / 80, 2, 14),
        color=df["Price"].apply(lambda p: COLOR_UP if p > midpoint_price else COLOR_DOWN),
        opacity=0.8,
        line=dict(width=0)
    ),
    hovertemplate="時間: %{x} ms<br>價格: %{y:.2f}<extra></extra>"
), row=1, col=1)

# 中線
fig.add_hline(
    y=midpoint_price, line_color=COLOR_MID,
    line_dash="dash", line_width=2,
    annotation_text=f"中線 {midpoint_price:.2f}",
    annotation_position="top right",
    annotation_font_color=COLOR_MID,
    row=1, col=1
)

# 段落標記
for label, idx, color in [
    ("盤整", seg1 // 2, "#58a6ff"),
    ("假突破", seg1 + seg2 // 2, "#f85149"),
    ("真突破", seg1 + seg2 + (n_ticks - seg1 - seg2) // 2, "#3fb950"),
]:
    if idx < len(df):
        fig.add_annotation(
            x=df["Timestamp"].iloc[idx],
            y=df["Price"].max() * 0.995,
            text=label,
            showarrow=False,
            font=dict(color=color, size=11),
            row=1, col=1
        )

# ── 下圖：Hawkes 條件強度 λ(t) ───────────────────
# 柱狀圖：根據強度著色
bar_colors = []
for lam in df["Lambda"]:
    if lam >= lambda_threshold_high:
        bar_colors.append(COLOR_UP)
    elif lam <= lambda_threshold_low:
        bar_colors.append(COLOR_DOWN)
    else:
        bar_colors.append("#484f58")

fig.add_trace(go.Bar(
    x=df["Timestamp"], y=df["Lambda"],
    name="λ(t)",
    marker_color=bar_colors,
    opacity=0.85,
    hovertemplate="時間: %{x} ms<br>λ(t): %{y:.5f}<extra></extra>"
), row=2, col=1)

# 疊加折線（平滑趨勢）
rolling_lambda = pd.Series(df["Lambda"]).rolling(window=8, min_periods=1).mean()
fig.add_trace(go.Scatter(
    x=df["Timestamp"], y=rolling_lambda,
    mode="lines",
    name="λ 移動平均",
    line=dict(color="#e3b341", width=2),
    hoverinfo="skip"
), row=2, col=1)

# 高低閾值水平線
fig.add_hline(
    y=lambda_threshold_high, line_color=COLOR_THRESH_H,
    line_dash="dot", line_width=1.5,
    annotation_text=f"高位閾值 (P80) {lambda_threshold_high:.4f}",
    annotation_font_color=COLOR_THRESH_H,
    annotation_font_size=10,
    row=2, col=1
)
fig.add_hline(
    y=lambda_threshold_low, line_color=COLOR_THRESH_L,
    line_dash="dot", line_width=1.5,
    annotation_text=f"低位閾值 (P35) {lambda_threshold_low:.4f}",
    annotation_font_color=COLOR_THRESH_L,
    annotation_font_size=10,
    row=2, col=1
)

# ── 全域樣式 ──────────────────────────────────
fig.update_layout(
    height=680,
    showlegend=True,
    legend=dict(
        orientation="h", y=1.03, x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", size=11)
    ),
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color="#8b949e", size=11),
    margin=dict(l=60, r=20, t=60, b=40),
    hovermode="x unified",
)

# 坐標軸樣式
for row in [1, 2]:
    fig.update_xaxes(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        showspikes=True, spikecolor="#484f58", spikethickness=1,
        row=row, col=1
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        row=row, col=1
    )

fig.update_xaxes(title_text="時間（毫秒）", row=2, col=1)
fig.update_yaxes(title_text="成交價格", row=1, col=1)
fig.update_yaxes(title_text="λ(t)", row=2, col=1)

# 渲染圖表
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# ──────────────────────────────────────────────
# 底部：數據表 & 參數摘要
# ──────────────────────────────────────────────
st.markdown("---")
col_tbl, col_param = st.columns([3, 1])

with col_tbl:
    st.markdown("#### 📋 最近 20 筆 Tick 數據")
    display_df = df.tail(20)[["Timestamp", "Price", "Quantity", "Lambda"]].copy()
    display_df.columns = ["時間戳 (ms)", "成交價", "成交量", "λ(t)"]
    display_df["λ(t)"] = display_df["λ(t)"].map("{:.5f}".format)
    st.dataframe(
        display_df.style.background_gradient(subset=["λ(t)"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True
    )

with col_param:
    st.markdown("#### 📐 當前參數摘要")
    st.markdown(f"""
| 參數 | 值 |
|------|-----|
| μ（背景強度）| `{mu:.3f}` |
| α（激發跳躍）| `{alpha:.2f}` |
| β（衰減率）  | `{beta:.2f}` |
| α/β（分支比）| `{alpha/beta:.3f}` |
| 中線價格     | `{midpoint_price:.2f}` |
| λ 高位閾值   | `{lambda_threshold_high:.4f}` |
| λ 低位閾值   | `{lambda_threshold_low:.4f}` |
| 模擬 Tick 數 | `{n_ticks}` |
    """)
    # 穩定性提示
    if alpha < beta:
        st.success(f"✅ 穩定（α/β = {alpha/beta:.2f} < 1）")
    else:
        st.error(f"⚠️ 爆炸性過程（α/β = {alpha/beta:.2f} ≥ 1）")

st.markdown("""
<div style='text-align:center; color:#484f58; margin-top:20px; font-size:0.8rem;'>
Hawkes 過程微觀突破驗證器 ｜ 僅供研究用途，非投資建議
</div>
""", unsafe_allow_html=True)
