DEMO：https://0318drldic2-fvf6qjm3hqcd5snq2sj3sn.streamlit.app/

# Gridworld 價值迭代演算法視覺化工具 (Value Iteration Visualizer)

這是一個基於 **Streamlit** 開發的互動式網頁應用程式，展示強化學習中的 **價值迭代 (Value Iteration)** 演算法如何在特定的 Gridworld 環境中尋找最佳政策。

## 🎯 專案訴求與規範

1. **環境規格**：
   - **網格大小**：5x5。
   - **起點 (Start)**：位於儲存格 (0, 0)，以 **黃色** 標示。
   - **終點 (Goal)**：位於儲存格 (4, 4)，以 **綠色** 標示。
   - **障礙物 (Block)**：固定於 (1, 1), (2, 2), (3, 3)，以 **深灰色** 標示。
2. **核心任務**：
   - 使用價值迭代演算法 (Value Iteration) 推導每個狀態的最佳政策。
   - 於網格中顯示推導出的 **最佳路徑箭頭** ($\uparrow, \downarrow, \leftarrow, \rightarrow$) 與 **價值函數 $V(s)$**。

## 🚀 技術更新亮點 (v2.0)

- **強健的網頁渲染**：為了解決傳統 `st.markdown` 無法正確呈現複雜 HTML 結構（導致顯示原始碼）的問題，本專案改用 `streamlit.components.v1.html`。
- **獨立 iframe 容器**：將 HTML/CSS 網格封裝在獨立的 iframe 中，確保在 Streamlit Cloud 或各種瀏覽器部署環境下，樣式都能 100% 正確呈現，且不會被 Markdown 解析器干擾。
- **動態收斂動畫**：實作即時重繪機制，使用者可親眼觀察價值函數從 0 開始逐漸傳播並收斂至穩定狀態的過程。

## 🛠️ 技術棧

- **Python**: 演算法邏輯實作。
- **Streamlit**: 網頁介面框架。
- **Streamlit Components**: 解決複雜 HTML/CSS 渲染的關鍵組件。
- **NumPy**: 處理價值矩陣 (Value Table) 的數學運算。
- **HTML/CSS**: 自定義高度視覺化的網格 UI。

## 📂 程式碼結構說明 (`app.py`)

- **環境初始化**: 設定固定的起終點與障礙物坐標。
- **核心邏輯**:
    - `get_next_state`: 處理撞牆、邊界與障礙物判定邏輯。
    - `get_reward`: 定義進入終點獲得高額獎勵，其他步動作給予懲罰（Step Reward）。
- **視覺化渲染 (`render_grid_html`)**: 將 NumPy 矩陣與政策陣列轉換為帶有樣式的 HTML `<table>`。
- **價值迭代循環**: 根據 Bellman 方程式更新 $V(s)$，直到 $\Delta < \theta$ 為止。

## 📖 如何執行

1. **安裝依賴**：
   ```bash
   pip install streamlit numpy
   ```

2. **啟動程式**：
   ```bash
   streamlit run app.py
   ```

---
**💡 提示**：點擊「🚀 開始價值迭代」後，系統會開始計算。收斂後，橘色箭頭將指示從起點前往終點的最短路徑。
