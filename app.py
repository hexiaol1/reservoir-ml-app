import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import streamlit as st

st.set_page_config(
    page_title="回归预测系统 (含标签对数变换)",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_txt_file(uploaded_file):
    """自适应读取 TXT 文件（兼容逗号、空格、制表符等分隔符）"""
    try:
        content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        content = uploaded_file.getvalue().decode("gbk", errors="ignore")

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines:
        return None

    first_line = lines[0]
    sep = "," if "," in first_line else r"\s+"
    return pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")


# --- 侧边栏配置 ---
st.sidebar.title("⚙️ 模型与预处理配置")

model_type = st.sidebar.selectbox(
    "选择回归算法",
    ["随机森林 (Random Forest)", "支持向量回归 (SVR)", "BP 神经网络 (MLP)"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 目标标签变换设置")
target_transform = st.sidebar.selectbox(
    "标签数学变换",
    [
        "不进行变换 (None)",
        "自然对数: ln(1 + y) [适合包含0的非负参数]",
        "常用对数: log10(y) [适合绝对正数跨数量级参数]",
    ],
)

test_size = (
    st.sidebar.slider("训练集验证切分比例 (Test Size)", 10, 40, 20, step=5)
    / 100.0
)
random_state = st.sidebar.number_input(
    "随机数种子 (Random Seed)", value=42, step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("超参数调节")

if "随机森林" in model_type:
    n_estimators = st.sidebar.slider("决策树数量 (n_estimators)", 10, 500, 100, 10)
    max_depth = st.sidebar.slider("最大树深 (max_depth, 0为不限)", 0, 30, 10)
    max_depth_val = None if max_depth == 0 else max_depth
elif "支持向量" in model_type:
    c_val = st.sidebar.number_input("惩罚系数 C", value=10.0, step=1.0)
    gamma = st.sidebar.selectbox("核函数系数 (gamma)", ["scale", "auto"])
    kernel = st.sidebar.selectbox(
        "核函数 (kernel)", ["rbf", "linear", "poly"]
    )
elif "BP 神经网络" in model_type:
    hidden_layer_1 = st.sidebar.slider("第 1 隐藏层神经元", 8, 128, 64, 8)
    hidden_layer_2 = st.sidebar.slider("第 2 隐藏层神经元", 0, 64, 32, 8)
    hidden_layers = (
        (hidden_layer_1, hidden_layer_2)
        if hidden_layer_2 > 0
        else (hidden_layer_1,)
    )
    max_iter = st.sidebar.slider("最大迭代轮数 (max_iter)", 100, 2000, 500, 100)
    activation = st.sidebar.selectbox(
        "激活函数", ["relu", "tanh", "logistic"]
    )

# --- 主界面 ---
st.title("📊 多模型回归与盲井预测系统")
st.caption(
    "集成标签自适应对数变换、特征标准化、RF/SVR/BP 训练以及盲井数据逆变换导出"
)

col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader(
        "📂 上传训练集数据 (TXT/CSV)", type=["txt", "csv"]
    )
with col2:
    blind_file = st.file_uploader(
        "📂 上传盲井数据 (TXT/CSV)", type=["txt", "csv"]
    )

if train_file and blind_file:
    train_df = load_txt_file(train_file)
    blind_df = load_txt_file(blind_file)

    if train_df is None or blind_df is None:
        st.error("数据加载失败，请检查文件编码或格式。")
        st.stop()

    with st.expander("🔍 原始数据预览", expanded=False):
        c1, c2 = st.columns(2)
        c1.write("**训练集前 5 行:**")
        c1.dataframe(train_df.head())
        c2.write("**盲井数据前 5 行:**")
        c2.dataframe(blind_df.head())

    st.markdown("### 🎯 特征与目标变量映射")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("未检测到有效数值型列，请确认分隔符是否匹配。")
        st.stop()

    col_target, col_features = st.columns([1, 2])
    with col_target:
        target_col = st.selectbox("选择预测的目标标签 (Y)", numeric_cols)

    available_features = [c for c in numeric_cols if c != target_col]
    default_features = [c for c in available_features if c in blind_df.columns]

    with col_features:
        feature_cols = st.multiselect(
            "选择输入特征 (X) - 盲井必须包含",
            available_features,
            default=default_features,
        )

    missing_in_blind = [c for c in feature_cols if c not in blind_df.columns]
    if missing_in_blind:
        st.warning(f"⚠️ 盲井文件中缺少所选特征: {missing_in_blind}")
        st.stop()

    if not feature_cols:
        st.info("请至少选择一个特征列。")
        st.stop()

    # --- 训练与预测执行 ---
    if st.button("🚀 启动模型训练并预测盲井", type="primary"):
        clean_train = train_df.dropna(
            subset=feature_cols + [target_col]
        ).copy()
        clean_blind = blind_df.copy()

        y_raw = clean_train[target_col].values.astype(float)

        # 检查对数变换合法性
        if "log10" in target_transform:
            if np.any(y_raw <= 0):
                st.error("❌ 标签包含小于或等于 0 的数值，无法进行 log10 变换！建议改用 ln(1+y) 或进行数据过滤。")
                st.stop()
            y_trans = np.log10(y_raw)
            inv_trans_fn = lambda val: 10.0**val
            transform_name = "log10(y)"
        elif "ln(1 + y)" in target_transform:
            if np.any(y_raw < -1.0):
                st.error("❌ 标签包含小于 -1 的数值，无法进行 ln(1+y) 变换！")
                st.stop()
            y_trans = np.log1p(y_raw)
            inv_trans_fn = lambda val: np.expm1(val)
            transform_name = "ln(1+y)"
        else:
            y_trans = y_raw
            inv_trans_fn = lambda val: val
            transform_name = "y (原始值)"

        X = clean_train[feature_cols].values

        # 数据集切分
        X_train, X_test, y_train_trans, y_test_trans = train_test_split(
            X, y_trans, test_size=test_size, random_state=random_state
        )

        # 还原对应的原始 y 用于实际物理量纲评估
        y_train_raw = inv_trans_fn(y_train_trans)
        y_test_raw = inv_trans_fn(y_test_trans)

        # 特征标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # 模型训练（在变换后的空间进行学习）
        if "随机森林" in model_type:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth_val,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train_trans)
            y_train_trans_pred = model.predict(X_train)
            y_test_trans_pred = model.predict(X_test)
        elif "支持向量" in model_type:
            model = SVR(C=c_val, gamma=gamma, kernel=kernel)
            model.fit(X_train_scaled, y_train_trans)
            y_train_trans_pred = model.predict(X_train_scaled)
            y_test_trans_pred = model.predict(X_test_scaled)
        elif "BP 神经网络" in model_type:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
            )
            model.fit(X_train_scaled, y_train_trans)
            y_train_trans_pred = model.predict(X_train_scaled)
            y_test_trans_pred = model.predict(X_test_scaled)

        # 反变换回原始物理尺度
        y_train_raw_pred = inv_trans_fn(y_train_trans_pred)
        y_test_raw_pred = inv_trans_fn(y_test_trans_pred)

        # 评估指标
        # 1. 变换空间的指标 (模型实际优化目标)
        r2_trans = r2_score(y_test_trans, y_test_trans_pred)
        rmse_trans = np.sqrt(
            mean_squared_error(y_test_trans, y_test_trans_pred)
        )
        # 2. 原始尺度指标 (地质/物理实际误差)
        r2_raw = r2_score(y_test_raw, y_test_raw_pred)
        rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_test_raw_pred))
        mae_raw = mean_absolute_error(y_test_raw, y_test_raw_pred)

        st.markdown("### 📈 模型测试集评估指标")
        tab_raw, tab_trans = st.tabs(
            ["🌍 还原后原始尺度表现 (实际物理量)", "📐 对数变换空间表现 (模型优化目标)"]
        )

        with tab_raw:
            m1, m2, m3 = st.columns(3)
            m1.metric("还原 R²", f"{r2_raw:.4f}")
            m2.metric("还原 RMSE", f"{rmse_raw:.4f}")
            m3.metric("还原 MAE", f"{mae_raw:.4f}")

        with tab_trans:
            mt1, mt2 = st.columns(2)
            mt1.metric(f"对数空间 R² ({transform_name})", f"{r2_trans:.4f}")
            mt2.metric(f"对数空间 RMSE", f"{rmse_trans:.4f}")

        # 拟合对比图（两张联动：对数空间 vs 还原空间）
        st.markdown("### 📊 拟合效果散点图")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        # 子图 1: 变换空间
        ax1 = axes[0]
        ax1.scatter(
            y_train_trans,
            y_train_trans_pred,
            color="#2b5c8f",
            alpha=0.6,
            label="Train",
            edgecolors="k",
            s=25,
        )
        ax1.scatter(
            y_test_trans,
            y_test_trans_pred,
            color="#e07a5f",
            alpha=0.8,
            label="Test",
            edgecolors="k",
            s=30,
        )
        min_t = min(y_trans.min(), y_test_trans_pred.min())
        max_t = max(y_trans.max(), y_test_trans_pred.max())
        ax1.plot([min_t, max_t], [min_t, max_t], "r--", lw=1.5, label="1:1 Line")
        ax1.set_xlabel(f"Measured [{transform_name}]")
        ax1.set_ylabel(f"Predicted [{transform_name}]")
        ax1.set_title(f"Model Space: {transform_name}")
        ax1.legend(frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # 子图 2: 原始物理空间
        ax2 = axes[1]
        ax2.scatter(
            y_train_raw,
            y_train_raw_pred,
            color="#2b5c8f",
            alpha=0.6,
            label="Train",
            edgecolors="k",
            s=25,
        )
        ax2.scatter(
            y_test_raw,
            y_test_raw_pred,
            color="#e07a5f",
            alpha=0.8,
            label="Test",
            edgecolors="k",
            s=30,
        )
        min_r = min(y_raw.min(), y_test_raw_pred.min())
        max_r = max(y_raw.max(), y_test_raw_pred.max())
        ax2.plot([min_r, max_r], [min_r, max_r], "r--", lw=1.5, label="1:1 Line")
        ax2.set_xlabel(f"Measured (Original: {target_col})")
        ax2.set_ylabel(f"Predicted (Original: {target_col})")
        ax2.set_title("Original Physical Scale")
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle=":", alpha=0.6)

        st.pyplot(fig)

        # --- 盲井推理与反变换导出 ---
        st.markdown("### 🕳️ 盲井预测推理")
        blind_X = clean_blind[feature_cols].values

        if "随机森林" in model_type:
            blind_trans_preds = model.predict(blind_X)
        else:
            blind_X_scaled = scaler_X.transform(blind_X)
            blind_trans_preds = model.predict(blind_X_scaled)

        # 关键：反变换回物理真实单位
        blind_raw_preds = inv_trans_fn(blind_trans_preds)

        if target_transform != "不进行变换 (None)":
            clean_blind[f"Pred_{target_col}_Transformed"] = np.round(
                blind_trans_preds, 4
            )
        clean_blind[f"Pred_{target_col}_Raw"] = np.round(blind_raw_preds, 4)

        st.dataframe(clean_blind.head(10))

        # 导出文件
        csv_buffer = io.StringIO()
        clean_blind.to_csv(csv_buffer, index=False, sep="\t")

        st.download_button(
            label="📥 导出盲井预测结果 (含反变换原始值 TXT)",
            data=csv_buffer.getvalue(),
            file_name=f"blind_predicted_{target_col}.txt",
            mime="text/plain",
        )
else:
    st.info("💡 请在上方上传训练集与盲井数据的 TXT/CSV 文件开始。")
