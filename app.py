import streamlit as st
import numpy as np
import time

# ==========================================
# 1. 環境參數設定
# ==========================================
st.set_page_config(page_title="Gridworld Value Iteration", layout="wide")
st.title("Gridworld：價值迭代演算法 (Value Iteration) 視覺化")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定")
GRID_SIZE = 5
gamma = st.sidebar.slider("折扣因子 (Gamma, γ)", 0.5, 0.99, 0.9, help="對未來獎勵的重視程度")
theta = st.sidebar.number_input("收斂閾值 (Theta, θ)", value=1e-4, format="%.5f")
step_reward = st.sidebar.number_input("每步獎勵 (Step Reward)", value=-1.0)
goal_reward = st.sidebar.number_input("終點獎勵 (Goal Reward)", value=10.0)

# 固定環境設定 (根據要求)
START_STATE = (0, 0)
GOAL_STATE = (4, 4)
OBSTACLES = [(1, 1), (2, 2), (3, 3)]

# 定義動作
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上, 下, 左, 右
ACTION_SYMBOLS = ["↑", "↓", "←", "→"]

# 初始化 Session State
if 'V' not in st.session_state:
    st.session_state.V = np.zeros((GRID_SIZE, GRID_SIZE))
if 'policy' not in st.session_state:
    st.session_state.policy = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# ==========================================
# 2. 核心演算法邏輯
# ==========================================
def get_next_state(r, c, action):
    """計算採取行動後的下一個狀態（考慮邊界與障礙物）"""
    dr, dc = action
    next_r, next_c = r + dr, c + dc
    
    # 檢查邊界
    if next_r < 0 or next_r >= GRID_SIZE or next_c < 0 or next_c >= GRID_SIZE:
        return r, c
    # 檢查障礙物
    if (next_r, next_c) in OBSTACLES:
        return r, c
    return next_r, next_c

def get_reward(s_prime):
    """取得進入 s_prime 狀態的獎勵"""
    if s_prime == GOAL_STATE:
        return goal_reward
    return step_reward

# ==========================================
# 3. HTML 表格渲染 (解決部署呈現問題)
# ==========================================
def render_grid_html(V, policy):
    html = """
    <style>
        .grid-table {
            border-collapse: collapse;
            margin: 20px auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #ffffff;
        }
        .grid-cell {
            width: 85px;
            height: 85px;
            border: 2px solid #333;
            text-align: center;
            vertical-align: middle;
            position: relative;
        }
        .goal { background-color: #2ECC71 !important; color: white; }
        .obstacle { background-color: #34495E !important; color: white; }
        .start { background-color: #F1C40F !important; }
        .arrow { font-size: 28px; font-weight: bold; color: #E67E22; display: block; margin-top: 5px; }
        .val-text { font-size: 11px; color: #7F8C8D; position: absolute; bottom: 2px; right: 5px; }
        .label-text { font-size: 12px; font-weight: bold; position: absolute; top: 2px; left: 5px; }
    </style>
    <table class="grid-table">
    """
    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            state = (r, c)
            cell_class = "grid-cell"
            label = ""
            
            # 判斷格位類型
            if state == GOAL_STATE:
                cell_class += " goal"
                label = "GOAL"
            elif state in OBSTACLES:
                cell_class += " obstacle"
                label = "BLOCK"
            elif state == START_STATE:
                cell_class += " start"
                label = "START"
            
            # 取得該格位的價值與政策箭頭
            v_val = f"{V[r,c]:.2f}" if state not in OBSTACLES else "N/A"
            p_symbol = policy[r][c] if (state != GOAL_STATE and state not in OBSTACLES) else ""
            
            html += f"""
            <td class="{cell_class}">
                <span class="label-text">{label}</span>
                <span class="arrow">{p_symbol}</span>
                <span class="val-text">V:{v_val}</span>
            </td>
            """
        html += "</tr>"
    html += "</table>"
    return html

# ==========================================
# 4. 主介面佈局
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ 網格世界與最佳政策")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(render_grid_html(st.session_state.V, st.session_state.policy), unsafe_allow_html=True)

with col2:
    st.subheader("🎮 執行控制")
    st.write("點擊下方按鈕開始計算收斂路徑。")
    
    if st.button("🚀 開始價值迭代", use_container_width=True):
        V = np.zeros((GRID_SIZE, GRID_SIZE))
        policy = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        
        iteration = 0
        while True:
            delta = 0
            new_V = np.copy(V)
            
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    state = (r, c)
                    # 終點與障礙物不需要更新價值
                    if state == GOAL_STATE or state in OBSTACLES:
                        continue
                    
                    action_values = []
                    for action in ACTIONS:
                        s_prime = get_next_state(r, c, action)
                        reward = get_reward(s_prime)
                        # Bellman Equation: V(s) = R + γ * V(s')
                        val = reward + gamma * V[s_prime[0], s_prime[1]]
                        action_values.append(val)
                    
                    # 找到最大價值
                    best_val = max(action_values)
                    new_V[r, c] = best_val
                    
                    # 更新當前最佳政策箭頭
                    best_idx = np.argmax(action_values)
                    policy[r][c] = ACTION_SYMBOLS[best_idx]
                    
                    delta = max(delta, abs(V[r, c] - new_V[r, c]))
            
            V = new_V
            iteration += 1
            
            # 即時渲染動畫效果
            grid_placeholder.markdown(render_grid_html(V, policy), unsafe_allow_html=True)
            time.sleep(0.1)
            
            if delta < theta or iteration > 100:
                break
        
        st.session_state.V = V
        st.session_state.policy = policy
        st.success(f"✅ 已收斂！總計迭代次數: {iteration}")

    if st.button("🔄 重置網格", use_container_width=True):
        st.session_state.V = np.zeros((GRID_SIZE, GRID_SIZE))
        st.session_state.policy = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        st.rerun()

st.markdown("---")
st.info("""
**使用說明：**
- **START (黃色)**: 起點 (0,0)
- **GOAL (綠色)**: 終點 (4,4)
- **BLOCK (深灰色)**: 障礙物 (1,1), (2,2), (3,3)
- **橘色箭頭**: 演算法推導出的最佳行動方向。
- **V 數值**: 該狀態的期望長期價值。
""")
