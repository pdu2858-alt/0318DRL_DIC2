import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time

# ==========================================
# 1. 環境參數設定
# ==========================================
st.set_page_config(page_title="Gridworld Value Iteration", layout="wide")
st.title("Gridworld：價值迭代演算法 (Value Iteration)")

# 側邊欄設定
st.sidebar.header("⚙️ 參數設定")
GRID_SIZE = 5
gamma = st.sidebar.slider("折扣因子 (Gamma, γ)", 0.5, 0.99, 0.9)
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
if 'path' not in st.session_state:
    st.session_state.path = []

# ==========================================
# 2. 核心演算法邏輯
# ==========================================
def get_next_state(r, c, action):
    dr, dc = action
    next_r, next_c = r + dr, c + dc
    if next_r < 0 or next_r >= GRID_SIZE or next_c < 0 or next_c >= GRID_SIZE:
        return r, c
    if (next_r, next_c) in OBSTACLES:
        return r, c
    return next_r, next_c

def get_reward(s_prime):
    if s_prime == GOAL_STATE:
        return goal_reward
    return step_reward

def get_optimal_path(policy):
    path = []
    curr = START_STATE
    visited = set()
    while curr != GOAL_STATE and curr not in visited:
        visited.add(curr)
        path.append(curr)
        symbol = policy[curr[0]][curr[1]]
        if symbol == "": 
            break
        try:
            action_idx = ACTION_SYMBOLS.index(symbol)
            action = ACTIONS[action_idx]
            curr = get_next_state(curr[0], curr[1], action)
        except ValueError:
            break
    if curr == GOAL_STATE:
        path.append(GOAL_STATE)
    return path

# ==========================================
# 3. HTML 渲染函數 (使用 Components 解決顯示問題)
# ==========================================
def render_grid_html(V, policy, path=[]):
    # 建立內嵌 CSS 樣式
    css = """
    <style>
        .grid-table {
            border-collapse: collapse;
            margin: 10px auto;
            background-color: #f8f9fa;
        }
        .grid-cell {
            width: 80px;
            height: 80px;
            border: 2px solid #444;
            text-align: center;
            vertical-align: middle;
            position: relative;
            font-family: Arial, sans-serif;
        }
        .goal { background-color: #28a745 !important; color: white; }
        .obstacle { background-color: #343a40 !important; color: white; }
        .start { background-color: #ffc107 !important; }
        .path { background-color: #b7e1cd !important; } /* Light green for path */
        .arrow { font-size: 30px; font-weight: bold; color: #fd7e14; display: block; }
        .val-text { font-size: 10px; color: #6c757d; position: absolute; bottom: 2px; right: 4px; }
        .label-text { font-size: 11px; font-weight: bold; position: absolute; top: 2px; left: 4px; color: #333; }
        .obstacle .label-text { color: #fff; }
        .goal .label-text { color: #fff; }
    </style>
    """
    
    table_content = '<table class="grid-table">'
    for r in range(GRID_SIZE):
        table_content += "<tr>"
        for c in range(GRID_SIZE):
            state = (r, c)
            cell_class = "grid-cell"
            label = ""
            
            if state == GOAL_STATE:
                cell_class += " goal"
                label = "GOAL"
            elif state in OBSTACLES:
                cell_class += " obstacle"
                label = "BLOCK"
            elif state == START_STATE:
                cell_class += " start"
                label = "START"
            elif state in path:
                cell_class += " path"
            
            v_val = f"{V[r,c]:.2f}" if state not in OBSTACLES else "N/A"
            p_symbol = policy[r][c] if (state != GOAL_STATE and state not in OBSTACLES) else ""
            
            table_content += f"""
            <td class="{cell_class}">
                <div class="label-text">{label}</div>
                <div class="arrow">{p_symbol}</div>
                <div class="val-text">V:{v_val}</div>
            </td>
            """
        table_content += "</tr>"
    table_content += "</table>"
    
    return css + table_content

# ==========================================
# 4. 主介面
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ 網格與最佳政策")
    # 使用 components.html 強制渲染為網頁元件
    grid_container = st.empty()
    with grid_container:
        components.html(render_grid_html(st.session_state.V, st.session_state.policy, st.session_state.path), height=500)

with col2:
    st.subheader("🎮 控制面板")
    if st.button("🚀 開始價值迭代", use_container_width=True):
        V = np.zeros((GRID_SIZE, GRID_SIZE))
        policy = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        st.session_state.path = []
        
        iteration = 0
        while True:
            delta = 0
            new_V = np.copy(V)
            
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    state = (r, c)
                    if state == GOAL_STATE or state in OBSTACLES:
                        continue
                    
                    action_values = []
                    for action in ACTIONS:
                        s_prime = get_next_state(r, c, action)
                        reward = get_reward(s_prime)
                        val = reward + gamma * V[s_prime[0], s_prime[1]]
                        action_values.append(val)
                    
                    best_val = max(action_values)
                    new_V[r, c] = best_val
                    
                    best_idx = np.argmax(action_values)
                    policy[r][c] = ACTION_SYMBOLS[best_idx]
                    
                    delta = max(delta, abs(V[r, c] - new_V[r, c]))
            
            V = new_V
            iteration += 1
            
            # 動態更新
            with grid_container:
                components.html(render_grid_html(V, policy), height=500)
            time.sleep(0.05)
            
            if delta < theta or iteration > 100:
                break
        
        st.session_state.V = V
        st.session_state.policy = policy
        st.session_state.path = get_optimal_path(policy)
        
        # 最終呈現路徑
        with grid_container:
            components.html(render_grid_html(V, policy, st.session_state.path), height=500)
            
        st.success(f"✅ 收斂完成！迭代次數: {iteration}")

    if st.button("🔄 重置", use_container_width=True):
        st.session_state.V = np.zeros((GRID_SIZE, GRID_SIZE))
        st.session_state.policy = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        st.session_state.path = []
        st.rerun()
