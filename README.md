# Gridworld 價值迭代演算法視覺化工具 (Value Iteration Visualizer)

這是一個基於 **Streamlit** 開發的互動式網頁應用程式，旨在演示強化學習中的 **價值迭代 (Value Iteration)** 演算法如何在簡單的 Gridworld 環境中運作。

## 🚀 功能亮點

1. **互動式環境設定**：
   - 透過側邊欄調整 **折扣因子 ($\gamma$)**、**收斂閾值 ($	heta$)**、**每步獎勵**及**終點獎勵**。
   - 預設 5x5 的網格環境，包含起點、終點以及自定義的障礙物 (Block)。

2. **核心演算法實現**：
   - 完整實作價值迭代公式：$V(s) \leftarrow \max_a \sum_{s', r} p(s', r|s, a) [r + \gamma V(s')]$。
   - 自動推導每個狀態的最佳政策 (Optimal Policy)，並以箭頭符號 ($\uparrow, \downarrow, \leftarrow, ightarrow$) 標示。

3. **動態視覺化過程**：
   - 點擊執行後，程式會動態更新網格中的數值與政策，讓使用者清楚觀察到價值函數如何隨著迭代次數增加而趨於穩定。
   - 使用自定義 HTML/CSS 渲染美觀的網格介面。

4. **即時反饋**：
   - 顯示算法收斂所需的總迭代次數。
   - 提供「重置環境」功能，方便重複實驗不同的參數組合。

## 🛠️ 技術棧

- **Python**: 核心邏輯。
- **Streamlit**: 網頁介面框架與互動組件。
- **NumPy**: 處理價值矩陣運算。
- **HTML/CSS**: 用於自定義網格的可視化渲染。

## 📂 程式碼結構說明 (`app.py`)

- **參數與環境設定**: 使用 `st.sidebar` 蒐集使用者輸入，並利用 `st.session_state` 跨渲染保存環境狀態（如起點、終點、障礙物座標）。
- **核心邏輯函數**:
    - `get_next_state`: 處理邊界檢查、撞牆邏輯及障礙物限制。
    - `get_reward`: 定義狀態轉移的即時獎勵。
- **渲染函數 (`render_grid`)**: 將 NumPy 矩陣轉換為帶有樣式的 HTML 表格，並動態插入政策箭頭。
- **主循環**: 實作 `while True` 迭代，計算 $\Delta = \max|V_{new} - V_{old}|$，直到小於閾值 $	heta$ 為止。

## 📖 如何執行

確保您已安裝必要套件：
```bash
pip install streamlit numpy
```

執行應用程式：
```bash
streamlit run app.py
```
