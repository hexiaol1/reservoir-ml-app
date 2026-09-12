import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import chardet

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------------------------------------
# 页面配置与字体全局设置
# -------------------------------------------------------------
st.set_page_config(
    page_title="储层参数机器学习预测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 兼容 Linux 云端 (WenQuanYi)、Windows (SimHei/Microsoft YaHei) 和 macOS (PingFang SC) 的中文字体列表
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',
    'SimHei',
    'Microsoft YaHei',
    'PingFang SC',
    'Source Han Sans CN',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=plt.rcParams['font.sans-serif'][0])

# -------------------------------------------------------------
# TXT 智能读取器（自动处理编码与分隔符）
# -------------------------------------------------------------
def load_txt_file(uploaded_file):
    """
    智能解析 TXT/CSV 文件，自动探测编码 (UTF-8, GBK, GB18030) 与分隔符
    """
    raw_bytes = uploaded_file.getvalue()
    
    # 1. 探测编码
    detect_res = chardet.detect(raw_bytes[:10000])
    encoding = detect_res.get('encoding', 'utf-8')
    if encoding is None or encoding.lower() in ['ascii', 'windows-1252']:
        encoding = 'gb18030'  # 中文 Windows 下 txt 默认常见编码

    # 尝试解码并解析，失败时回退编码
    encodings_to_try = [encoding, 'utf-8', 'gb18030', 'gbk', 'utf-8-sig']
    df = None
    
    for enc in encodings_to_try:
        try:
            # \s+ 可同时兼容单空格、多空格、Tab 制表符；sep=None 允许 Python 引擎自动推断
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=enc,
                sep=r'\s+|,|\t',
                engine='python'
            )
            # 如果只读成了一列且有逗号，再试一次逗号分隔
            if df.shape[1] <= 1:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            break
        except Exception:
            continue
            
    return df

# -------------------------------------------------------------
# 模拟数据生成器（兜底演示）
# -------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_samples=600):
    np.random.seed(42)
    depth = np.linspace(2000, 2600, n_samples)
    gr = np.clip(40 + 60 * np.sin(depth / 50) + np.random.normal(0, 10, n_samples), 15, 180)
    rhob = np.clip(2.65 - 0.003 * gr + np.random.normal(0, 0.04, n_samples), 2.1, 2.85)
    dt = np.clip(180 + (2.7 - rhob) * 120 + np.random.normal(0, 8, n_samples), 160, 320)
    nphi = np.clip(0.05 + 0.0015 * gr + np.random.normal(0, 0.02, n_samples), 0.01, 0.45)
    rt = 10 ** (np.random.uniform(0.5, 2.5, n_samples) + 0.01 * gr / 10)
    
    toc = np.clip(0.03 * gr + 0.01 * dt - 1.2 * rhob + np.random.normal(0, 0.3, n_samples), 0.2, 8.5)
    por = np.clip(((2.65 - rhob) / 1.65) * 70 + nphi * 30 + np.random.normal(0, 0.8, n_samples), 1.0, 30.0)
    perm = np.clip((10 ** (0.18 * por - 1.5)) * np.exp(np.random.normal(0, 0.4, n_samples)), 0.001, 2500.0)
    
    return pd.DataFrame({
        '井深(m)': depth,
        '自然伽马(GR)': gr,
        '声波时差(DT)': dt,
        '补偿密度(RHOB)': rhob,
        '中子孔隙度(NPHI)': nphi,
        '深侧向电阻率(RT)': rt,
        '总有机碳TOC(%)': toc,
        '孔隙度POR(%)': por,
        '渗透率PERM(mD)': perm
    })

# -------------------------------------------------------------
# 边栏配置：数据载入与字段关联
# -------------------------------------------------------------
st.sidebar.title("🛢️ 储层参数预测工作台")

data_mode = st.sidebar.radio("选择数据来源", ["上传本地 TXT/CSV 文件", "使用示例中文测井数据"])

df = None
if data_mode == "上传本地 TXT/CSV 文件":
    file = st.sidebar.file_uploader("上传包含中文字符的 TXT 或 CSV", type=["txt", "csv"])
    if file is not None:
        try:
            df = load_txt_file(file)
            st.sidebar.success(f"成功载入文件！探测到 {df.shape[0]} 行，{df.shape[1]} 列")
        except Exception as e:
            st.sidebar.error(f"文件解析失败: {str(e)}")
    else:
        st.sidebar.info("请上传 TXT/CSV 文件。未上传时展示演示数据。")
        df = generate_synthetic_data()
else:
    df = generate_synthetic_data()

# 清除含有 NaN 的行
df = df.dropna().reset_index(drop=True)
all_columns = df.columns.tolist()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 建模任务与目标选择")

task = st.sidebar.selectbox(
    "选择预测任务",
    [
        "基于随机森林的储层 TOC 预测",
        "基于支持向量回归 (SVR) 的储层孔隙度预测",
        "基于 BP 神经网络的储层渗透率预测"
    ]
)

# 自动推测深度列
depth_guess = next((c for c in all_columns if any(k in c.lower() for k in ['depth', '深', '井深'])), all_columns[0])
depth_col = st.sidebar.selectbox("选择井深列 (用于绘制连续测井图)", all_columns, index=all_columns.index(depth_guess))

# 自动推测目标变量列
if "TOC" in task:
    target_guess = next((c for c in all_columns if 'toc' in c.lower() or '碳' in c), all_columns[-1])
elif "孔隙度" in task:
    target_guess = next((c for c in all_columns if any(k in c.lower() for k in ['por', '孔隙'])), all_columns[-1])
else:
    target_guess = next((c for c in all_columns if any(k in c.lower() for k in ['perm', '渗透'])), all_columns[-1])

target_col = st.sidebar.selectbox("选择预测目标 (Target)", all_columns, index=all_columns.index(target_guess))

# 候选特征（排除深度和目标变量）
candidate_features = [c for c in all_columns if c not in [depth_col, target_col]]
feature_cols = st.sidebar.multiselect("选择输入特征 (Features)", candidate_features, default=candidate_features)

if not feature_cols:
    st.error("请在左侧边栏至少选择一个输入特征！")
    st.stop()

# -------------------------------------------------------------
# 划分比例与模型参数
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📐 数据集划分 (Train/Val/Test)")
test_ratio = st.sidebar.slider("测试集比例 (Test Set)", 0.1, 0.3, 0.15, step=0.05)
val_ratio = st.sidebar.slider("验证集比例 (从剩余数据中抽取 Val Set)", 0.1, 0.3, 0.2, step=0.05)

st.sidebar.subheader("⚙️ 算法超参数微调")
if "随机森林" in task:
    n_estimators = st.sidebar.slider("决策树数量 (n_estimators)", 20, 300, 100, step=20)
    max_depth = st.sidebar.slider("最大树深 (max_depth)", 3, 25, 10)
elif "支持向量" in task:
    c_param = st.sidebar.slider("惩罚参数 C", 0.1, 100.0, 10.0, step=1.0)
    eps_param = st.sidebar.slider("容忍误差 ε (epsilon)", 0.01, 1.0, 0.1, step=0.01)
    kernel_param = st.sidebar.selectbox("核函数", ["rbf", "linear", "poly"])
else:
    h1 = st.sidebar.slider("隐含层 1 节点数", 8, 128, 64, step=8)
    h2 = st.sidebar.slider("隐含层 2 节点数", 4, 64, 32, step=4)
    lr_param = st.sidebar.select_slider("初始学习率", [0.0001, 0.001, 0.01, 0.1], value=0.001)
    max_iters = st.sidebar.slider("最大迭代轮数", 200, 1500, 400, step=100)

# -------------------------------------------------------------
# 数据准备与标准化
# -------------------------------------------------------------
st.title("🛢️ 储层多参数智能回归预测平台")
st.caption("已启用多字符集解码引擎与跨平台中文字体自适应渲染。")

with st.expander("📄 查看已载入的测井数据表 (前 8 行)", expanded=False):
    st.dataframe(df.head(8), use_container_width=True)

X = df[feature_cols].values
y = df[target_col].values
depth_vals = df[depth_col].values

# 渗透率非线性处理（对数变换）
is_log_perm = False
if "渗透率" in task:
    is_log_perm = True
    y = np.log10(np.clip(y, 1e-4, None))

# 严格划分：训练集 (Train)、验证集 (Val)、测试集 (Test)
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X, y, np.arange(len(X)), test_size=test_ratio, random_state=42
)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=val_ratio, random_state=42
)

# 标准化（仅在训练集上 Fit，杜绝数据泄露）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.transform(X)

# -------------------------------------------------------------
# 算法拟合与预测
# -------------------------------------------------------------
if "随机森林" in task:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_va_pred = model.predict(X_val)
    y_te_pred = model.predict(X_test)
    y_al_pred = model.predict(X)
elif "支持向量" in task:
    model = SVR(C=c_param, epsilon=eps_param, kernel=kernel_param)
    model.fit(X_train_s, y_train)
    y_tr_pred = model.predict(X_train_s)
    y_va_pred = model.predict(X_val_s)
    y_te_pred = model.predict(X_test_s)
    y_al_pred = model.predict(X_all_s)
else:
    model = MLPRegressor(
        hidden_layer_sizes=(h1, h2),
        learning_rate_init=lr_param,
        max_iter=max_iters,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_s, y_train)
    y_tr_pred = model.predict(X_train_s)
    y_va_pred = model.predict(X_val_s)
    y_te_pred = model.predict(X_test_s)
    y_al_pred = model.predict(X_all_s)

# 物理空间还原（若为对数化渗透率）
if is_log_perm:
    y_tr_true, y_tr_hat = 10**y_train, 10**y_tr_pred
    y_va_true, y_va_hat = 10**y_val, 10**y_va_pred
    y_te_true, y_te_hat = 10**y_test, 10**y_te_pred
    y_al_true, y_al_hat = 10**y, 10**y_al_pred
else:
    y_tr_true, y_tr_hat = y_train, y_tr_pred
    y_va_true, y_va_hat = y_val, y_va_pred
    y_te_true, y_te_hat = y_test, y_te_pred
    y_al_true, y_al_hat = y, y_al_pred

# -------------------------------------------------------------
# 评价指标展示
# -------------------------------------------------------------
def get_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }

m_tr = get_metrics(y_tr_true, y_tr_hat)
m_va = get_metrics(y_va_true, y_va_hat)
m_te = get_metrics(y_te_true, y_te_hat)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("训练集拟合度 R²", f"{m_tr['r2']:.3f}", delta=f"RMSE: {m_tr['rmse']:.3f}")
    st.caption(f"样本数: {len(y_train)} | MAE: {m_tr['mae']:.3f}")
with col2:
    st.metric("验证集评估 R²", f"{m_va['r2']:.3f}", delta=f"RMSE: {m_va['rmse']:.3f}")
    st.caption(f"样本数: {len(y_val)} | MAE: {m_va['mae']:.3f}")
with col3:
    st.metric("独立测试集 R² (真实泛化)", f"{m_te['r2']:.3f}", delta=f"RMSE: {m_te['rmse']:.3f}")
    st.caption(f"样本数: {len(y_test)} | MAE: {m_te['mae']:.3f}")

st.markdown("---")

# -------------------------------------------------------------
# 图件看板展示
# -------------------------------------------------------------
st.subheader("📊 成果图件看板")

tab1, tab2, tab3 = st.tabs(["实测 vs 预测散点与残差图", "模型内核分析与学习曲线", "连续单井解释测井道图件"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 散点交叉图
    axes[0].scatter(y_tr_true, y_tr_hat, color='#1f77b4', alpha=0.6, label=f'训练集 (R²={m_tr["r2"]:.2f})')
    axes[0].scatter(y_va_true, y_va_hat, color='#ff7f0e', alpha=0.6, label=f'验证集 (R²={m_va["r2"]:.2f})')
    axes[0].scatter(y_te_true, y_te_hat, color='#d62728', marker='^', alpha=0.8, label=f'测试集 (R²={m_te["r2"]:.2f})')
    
    min_val = min(np.min(y_al_true), np.min(y_al_hat))
    max_val = max(np.max(y_al_true), np.max(y_al_hat))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='1:1 理想对角线')
    
    if is_log_perm:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel(f"实测值: {target_col}", fontsize=11)
    axes[0].set_ylabel(f"预测值: {target_col}", fontsize=11)
    axes[0].set_title(f"{target_col} 实测值与预测值散点对比", fontsize=12)
    axes[0].legend()
    
    # 残差概率密度曲线
    res_tr = y_tr_true - y_tr_hat
    res_te = y_te_true - y_te_hat
    sns.kdeplot(res_tr, ax=axes[1], fill=True, color='#1f77b4', label='训练集残差')
    sns.kdeplot(res_te, ax=axes[1], fill=True, color='#d62728', label='测试集残差')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_xlabel("残差 (实测值 - 预测值)", fontsize=11)
    axes[1].set_ylabel("概率密度", fontsize=11)
    axes[1].set_title("预测残差正态概率密度分布", fontsize=12)
    axes[1].legend()
    
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：算法机理图件
    if "随机森林" in task:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_cols = [feature_cols[i] for i in sorted_idx]
        sns.barplot(x=importances[sorted_idx], y=sorted_cols, ax=axes[0], palette="crest")
        axes[0].set_title("随机森林特征重要性 (MDI 权重)", fontsize=12)
        axes[0].set_xlabel("重要性分值", fontsize=11)
    elif "支持向量" in task:
        n_sv = len(model.support_)
        axes[0].pie(
            [n_sv, len(X_train) - n_sv],
            labels=['支持向量样本', '非支持向量样本'],
            autopct='%1.1f%%',
            colors=['#e15759', '#76b7b2'],
            startangle=90
        )
        axes[0].set_title(f"SVR 支持向量所占比例 (数量: {n_sv} / {len(X_train)})", fontsize=12)
    else:
        axes[0].plot(model.loss_curve_, color='#2ca02c', lw=2)
        axes[0].set_title("BP 神经网络损失迭代曲线 (Loss Curve)", fontsize=12)
        axes[0].set_xlabel("迭代轮数 (Epochs)", fontsize=11)
        axes[0].set_ylabel("损失值 (Loss)", fontsize=11)

    # 右图：样本量学习曲线 (检验过拟合/欠拟合)
    X_curve_in = X_temp if "随机森林" in task else scaler.transform(X_temp)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_curve_in, y_temp, cv=4, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='r2'
    )
    axes[1].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#1f77b4', label='训练集得分')
    axes[1].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='#ff7f0e', label='交叉验证得分')
    axes[1].set_title("学习曲线分析 (Learning Curve)", fontsize=12)
    axes[1].set_xlabel("参与训练的样本数量", fontsize=11)
    axes[1].set_ylabel("判定系数 R²", fontsize=11)
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].legend(loc="lower right")
    
    st.pyplot(fig)
    plt.close()

with tab3:
    # 连续井筒测井剖面道展示
    num_tracks = min(3, len(feature_cols)) + 2
    fig, axes = plt.subplots(1, num_tracks, figsize=(4 * num_tracks, 8), sharey=True)
    
    # 绘制选取的特征曲线
    for i in range(num_tracks - 2):
        col_name = feature_cols[i]
        axes[i].plot(df[col_name], depth_vals, color=sns.color_palette("tab10")[i], lw=1.2)
        axes[i].set_xlabel(col_name, fontsize=10)
        axes[i].grid(True, linestyle=':')
    
    # 绘制实测值 vs 预测值对比道
    pred_track = axes[num_tracks - 2]
    pred_track.scatter(y_al_true, depth_vals, color='black', s=8, alpha=0.5, label='实测值')
    pred_track.plot(y_al_hat, depth_vals, color='red', lw=1.5, label='机器学习预测')
    pred_track.set_xlabel(f"目标: {target_col}", fontsize=10)
    if is_log_perm:
        pred_track.set_xscale('log')
    pred_track.legend(loc='upper right')
    pred_track.grid(True, linestyle=':')
    
    # 绘制绝对误差道
    err_track = axes[num_tracks - 1]
    abs_diff = np.abs(y_al_true - y_al_hat)
    err_track.fill_betweenx(depth_vals, 0, abs_diff, color='orange', alpha=0.5)
    err_track.plot(abs_diff, depth_vals, color='darkorange', lw=1)
    err_track.set_xlabel("绝对预测误差", fontsize=10)
    err_track.set_xlim(0, np.percentile(abs_diff, 95) * 2)
    err_track.grid(True, linestyle=':')

    axes[0].invert_yaxis()  # 井深增加方向向下
    axes[0].set_ylabel(f"井深 ({depth_col})", fontsize=12)
    fig.suptitle(f"储层纵向剖面测井预测解释样图 [{target_col}]", fontsize=14, y=0.98)
    
    st.pyplot(fig)
    plt.close()
