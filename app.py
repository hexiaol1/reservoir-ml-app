import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import streamlit as st

# 页面基础配置
st.set_page_config(
    page_title="测井/地球化学多算法回归预测系统",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_txt_file(uploaded_file):
    """自适应读取 TXT 文件（兼容常见逗号、空格、制表符分隔）"""
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

    # 检测首行分隔符
    first_line = lines[0]
    if "," in first_line:
        sep = ","
    else:
        sep = r"\s+"

    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")
    return df


# --- 侧边栏：参数配置 ---
st.sidebar.title("⚙️ 模型与参数配置")

model_type = st.sidebar.selectbox(
    "选择回归算法",
    ["随机森林 (Random Forest)", "支持向量回归 (SVR)", "BP 神经网络 (MLP)"],
)

test_size = (
    st.sidebar.slider("训练集验证切分比例 (Test Size)", 10, 40, 20, step=5)
    / 100.0
)
random_state = st.sidebar.number_input(
    "随机数种子 (Random Seed)", value=42, step=1
)

# 动态超参数面板
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
st.title("📊 多模型回归与盲井参数智能预测系统")
st.caption(
    "支持训练集建模评估、盲井目标层段参数推演与一键导出（包含 RF / SVR / BP 神经网络）"
)

col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader(
        "📂 上传训练集数据 (TXT/CSV 格式)", type=["txt", "csv"]
    )
with col2:
    blind_file = st.file_uploader(
        "📂 上传盲井数据 (TXT/CSV 格式)", type=["txt", "csv"]
    )

if train_file and blind_file:
    train_df = load_txt_file(train_file)
    blind_df = load_txt_file(blind_file)

    if train_df is None or blind_df is None:
        st.error("数据加载失败，请检查文件编码或内容格式是否完整。")
        st.stop()

    with st.expander("🔍 原始数据预览", expanded=False):
        c1, c2 = st.columns(2)
        c1.write("**训练集预览 (前 5 行):**")
        c1.dataframe(train_df.head())
        c2.write("**盲井数据预览 (前 5 行):**")
        c2.dataframe(blind_df.head())

    # --- 变量映射配置 ---
    st.markdown("### 🎯 特征与目标变量映射")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("未检测到有效数值型列，请确认 TXT 文件分隔符是否匹配。")
        st.stop()

    col_target, col_features = st.columns([1, 2])
    with col_target:
        target_col = st.selectbox("选择预测的目标标签 (Y)", numeric_cols)

    available_features = [c for c in numeric_cols if c != target_col]
    # 默认匹配盲井中也存在的列
    default_features = [c for c in available_features if c in blind_df.columns]

    with col_features:
        feature_cols = st.multiselect(
            "选择输入特征 (X) - 必须存在于盲井数据中",
            available_features,
            default=default_features,
        )

    # 校验盲井是否具备所选特征
    missing_in_blind = [c for c in feature_cols if c not in blind_df.columns]
    if missing_in_blind:
        st.warning(f"⚠️ 盲井文件中缺少以下特征列: {missing_in_blind}，请重新勾选。")
        st.stop()

    if not feature_cols:
        st.info("请选择至少一个特征列进行训练。")
        st.stop()

    # --- 开始训练与预测 ---
    if st.button("🚀 启动模型训练并预测盲井", type="primary"):
        # 数据清洗：丢弃含缺失值的行
        clean_train = train_df.dropna(subset=feature_cols + [target_col])
        clean_blind = blind_df.copy()

        X = clean_train[feature_cols].values
        y = clean_train[target_col].values

        # 划分验证集
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 标准化处理 (针对 SVR 和 BP 神经网络尤为关键)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # 构建回归器
        if "随机森林" in model_type:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth_val,
                random_state=random_state,
                n_jobs=-1,
            )
            # 随机森林无需严格标准化，直接用原始或标准化皆可，此处保持统一
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        elif "支持向量" in model_type:
            model = SVR(C=c_val, gamma=gamma, kernel=kernel)
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        elif "BP 神经网络" in model_type:
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
            )
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

        # --- 模型评估指标 ---
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        st.markdown("### 📈 模型评估指标 (测试集)")
        m1, m2, m3 = st.columns(3)
        m1.metric("决定系数 (R²)", f"{r2_test:.4f}")
        m2.metric("均方根误差 (RMSE)", f"{rmse_test:.4f}")
        m3.metric("平均绝对误差 (MAE)", f"{mae_test:.4f}")

        # 散点对比可视化
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        ax.scatter(
            y_train,
            y_pred_train,
            color="steelblue",
            alpha=0.6,
            label="Train Data",
            edgecolors="k",
            s=30,
        )
        ax.scatter(
            y_test,
            y_pred_test,
            color="darkorange",
            alpha=0.8,
            label="Test Data",
            edgecolors="k",
            s=35,
        )

        min_val = min(y.min(), y_pred_test.min())
        max_val = max(y.max(), y_pred_test.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=1.5,
            label="1:1 Fit Line",
        )

        ax.set_xlabel(f"Measured {target_col}")
        ax.set_ylabel(f"Predicted {target_col}")
        ax.set_title(f"{model_type.split(' ')[0]} Fitting Performance")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

        # --- 盲井数据推理 ---
        st.markdown("### 🕳️ 盲井数据预测结果")
        blind_X = clean_blind[feature_cols].values

        if "随机森林" in model_type:
            blind_preds = model.predict(blind_X)
        else:
            blind_X_scaled = scaler_X.transform(blind_X)
            blind_preds = model.predict(blind_X_scaled)

        pred_col_name = f"Pred_{target_col}"
        clean_blind[pred_col_name] = np.round(blind_preds, 4)

        st.dataframe(clean_blind.head(10))

        # 下载结果文件
        csv_buffer = io.StringIO()
        clean_blind.to_csv(csv_buffer, index=False, sep="\t")

        st.download_button(
            label="📥 导出盲井预测结果 (制表符分隔 TXT)",
            data=csv_buffer.getvalue(),
            file_name=f"blind_well_predicted_{target_col}.txt",
            mime="text/plain",
        )
else:
    st.info("💡 请在上方同时上传训练集与盲井数据的 TXT/CSV 文件以开始分析。")
