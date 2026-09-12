import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import chardet

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------------------------------------
# 页面全局配置与中文字体支持
# -------------------------------------------------------------
st.set_page_config(
    page_title="储层物性机器学习多算法预测与增强平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 兼容 Linux 云端 (WenQuanYi)、Windows (SimHei/Microsoft YaHei) 和 macOS (PingFang SC)
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
# 稳健的 TXT/CSV 编码与分隔符自适应读取器
# -------------------------------------------------------------
def load_txt_file(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    detect_res = chardet.detect(raw_bytes[:10000])
    encoding = detect_res.get('encoding', 'utf-8')
    if encoding is None or encoding.lower() in ['ascii', 'windows-1252']:
        encoding = 'gb18030'

    encodings_to_try = [encoding, 'utf-8', 'gb18030', 'gbk', 'utf-8-sig']
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=enc,
                sep=r'\s+|,|\t',
                engine='python'
            )
            if df.shape[1] <= 1:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            break
        except Exception:
            continue
    return df

@st.cache_data
def generate_synthetic_data(n_samples=260):
    np.random.seed(42)
    depth = np.linspace(2100, 2450, n_samples)
    gr = np.clip(45 + 55 * np.sin(depth / 35) + np.random.normal(0, 8, n_samples), 20, 175)
    rhob = np.clip(2.68 - 0.0032 * gr + np.random.normal(0, 0.03, n_samples), 2.15, 2.8)
    dt = np.clip(185 + (2.7 - rhob) * 110 + np.random.normal(0, 6, n_samples), 170, 310)
    nphi = np.clip(0.04 + 0.0016 * gr + np.random.normal(0, 0.015, n_samples), 0.02, 0.42)
    rt = 10 ** (np.random.uniform(0.6, 2.2, n_samples) + 0.008 * gr / 10)
    
    toc = np.clip(0.028 * gr + 0.009 * dt - 1.15 * rhob + np.random.normal(0, 0.25, n_samples), 0.2, 8.0)
    por = np.clip(((2.65 - rhob) / 1.65) * 65 + nphi * 28 + np.random.normal(0, 0.7, n_samples), 1.5, 28.0)
    perm = np.clip((10 ** (0.17 * por - 1.4)) * np.exp(np.random.normal(0, 0.35, n_samples)), 0.01, 2000.0)
    
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
# 训练集数据增强算法引擎
# -------------------------------------------------------------
def augment_dataset(X_tr, y_tr, method="KNN流形邻域插值", factor=2, noise_level=0.03):
    n_samples, n_features = X_tr.shape
    n_synthetic = int(n_samples * factor)
    np.random.seed(42)
    
    if factor <= 0 or n_synthetic == 0:
        return X_tr, y_tr, np.zeros(n_samples, dtype=bool)

    if method == "高斯扰动抖动 (Jittering)":
        std_x = np.std(X_tr, axis=0, keepdims=True)
        std_y = np.std(y_tr)
        idx = np.random.choice(n_samples, n_synthetic, replace=True)
        synth_X = X_tr[idx] + np.random.normal(0, noise_level, (n_synthetic, n_features)) * std_x
        synth_y = y_tr[idx] + np.random.normal(0, noise_level, n_synthetic) * std_y

    elif method == "连续 Mixup 线性插值":
        idx1 = np.random.choice(n_samples, n_synthetic, replace=True)
        idx2 = np.random.choice(n_samples, n_synthetic, replace=True)
        lam = np.random.beta(0.4, 0.4, size=(n_synthetic, 1))
        synth_X = lam * X_tr[idx1] + (1 - lam) * X_tr[idx2]
        synth_y = (lam.ravel() * y_tr[idx1]) + ((1 - lam.ravel()) * y_tr[idx2])

    else:  # KNN 流形插值 (保持岩石物理耦合)
        k = min(5, n_samples - 1)
        knn = NearestNeighbors(n_neighbors=k + 1).fit(X_tr)
        _, indices = knn.kneighbors(X_tr)
        base_indices = np.random.choice(n_samples, n_synthetic, replace=True)
        neighbor_col = np.random.randint(1, k + 1, size=n_synthetic)
        neighbor_indices = indices[base_indices, neighbor_col]
        diff = X_tr[neighbor_indices] - X_tr[base_indices]
        rand_weights = np.random.uniform(0.05, 0.95, size=(n_synthetic, 1))
        synth_X = X_tr[base_indices] + rand_weights * diff
        synth_y = y_tr[base_indices] + rand_weights.ravel() * (y_tr[neighbor_indices] - y_tr[base_indices])
        synth_X += np.random.normal(0, 0.01 * np.std(X_tr, axis=0), synth_X.shape)
        synth_y += np.random.normal(0, 0.01 * np.std(y_tr), synth_y.shape)

    augmented_X = np.vstack([X_tr, synth_X])
    augmented_y = np.concatenate([y_tr, synth_y])
    is_synth = np.concatenate([np.zeros(n_samples, dtype=bool), np.ones(n_synthetic, dtype=bool)])
    return augmented_X, augmented_y, is_synth

# -------------------------------------------------------------
# 边栏：解耦配置中心
# -------------------------------------------------------------
st.sidebar.title("🛢️ 储层参数预测工作台")

# 1. 数据来源
data_mode = st.sidebar.radio("数据来源", ["上传本地 TXT/CSV 文件", "使用示例测井数据"])
if data_mode == "上传本地 TXT/CSV 文件":
    uploaded = st.sidebar.file_uploader("上传测井数据 TXT/CSV", type=["txt", "csv"])
    if uploaded is not None:
        try:
            df = load_txt_file(uploaded)
            st.sidebar.success(f"成功载入：{df.shape[0]} 行 × {df.shape[1]} 列")
        except Exception as e:
            st.sidebar.error(f"解析失败: {e}")
            df = generate_synthetic_data()
    else:
        st.sidebar.info("未上传文件，展示示例数据。")
        df = generate_synthetic_data()
else:
    df = generate_synthetic_data()

df = df.dropna().reset_index(drop=True)
all_columns = df.columns.tolist()

st.sidebar.markdown("---")

# 2. 预测变量与特征选择（完全自主）
st.sidebar.subheader("🎯 1. 变量与特征配置")
depth_guess = next((c for c in all_columns if any(k in c.lower() for k in ['depth', '深', '井深'])), all_columns[0])
depth_col = st.sidebar.selectbox("井深列 (用于沿深绘图)", all_columns, index=all_columns.index(depth_guess))

target_col = st.sidebar.selectbox("预测目标列 (Target)", all_columns, index=len(all_columns)-1)
candidate_features = [c for c in all_columns if c not in [depth_col, target_col]]
feature_cols = st.sidebar.multiselect("测井输入特征 (Features)", candidate_features, default=candidate_features)

if not feature_cols:
    st.error("请在左侧边栏至少选择一个输入特征！")
    st.stop()

# 目标对数变换选项（对于渗透率或量级跨度大的参数非常有效）
auto_log_default = any(k in target_col.lower() for k in ['perm', '渗透', 'k'])
enable_log_target = st.sidebar.checkbox("启用目标变量 log10 对数变换 (建议渗透率开启)", value=auto_log_default)

st.sidebar.markdown("---")

# 3. 独立算法选择窗口（核心改进点：自由组合模型）
st.sidebar.subheader("🧠 2. 机器学习算法选择")
selected_model_name = st.sidebar.selectbox(
    "选择拟合回归算法",
    [
        "随机森林回归 (Random Forest)",
        "支持向量回归 (SVR)",
        "BP 神经网络回归 (MLP Regressor)",
        "梯度提升回归树 (Gradient Boosting)"
    ]
)

# 动态专属超参数调节面板
with st.sidebar.expander("⚙️ 调整当前算法超参数", expanded=True):
    if selected_model_name == "随机森林回归 (Random Forest)":
        rf_n_estimators = st.slider("决策树数量 (n_estimators)", 20, 400, 150, step=20)
        rf_max_depth = st.slider("最大树深 (max_depth)", 3, 30, 12)
        rf_min_split = st.slider("内部节点再划分所需最小样本数", 2, 10, 2)
    elif selected_model_name == "支持向量回归 (SVR)":
        svr_c = st.slider("惩罚系数 C", 0.1, 100.0, 15.0, step=1.0)
        svr_epsilon = st.slider("容忍误差 ε (epsilon)", 0.01, 1.0, 0.08, step=0.01)
        svr_kernel = st.selectbox("核函数 (kernel)", ["rbf", "linear", "poly"])
    elif selected_model_name == "BP 神经网络回归 (MLP Regressor)":
        mlp_h1 = st.slider("隐含层 1 神经元数", 8, 128, 64, step=8)
        mlp_h2 = st.slider("隐含层 2 神经元数", 4, 64, 32, step=4)
        mlp_lr = st.select_slider("初始学习率", [0.0001, 0.001, 0.005, 0.01, 0.05], value=0.002)
        mlp_max_iter = st.slider("最大训练轮数 (Max Iter)", 200, 1500, 500, step=100)
    elif selected_model_name == "梯度提升回归树 (Gradient Boosting)":
        gb_n_estimators = st.slider("提升树数量 (n_estimators)", 20, 400, 100, step=20)
        gb_lr = st.slider("学习率 (Learning Rate)", 0.01, 0.3, 0.05, step=0.01)
        gb_max_depth = st.slider("单树最大深度", 2, 10, 4)

st.sidebar.markdown("---")

# 4. 数据增强与数据集切分
st.sidebar.subheader("⚡ 3. 训练集增强与切分")
enable_aug = st.sidebar.checkbox("开启训练集数据增强 (提升小样本泛化)", value=True)
if enable_aug:
    aug_method = st.sidebar.selectbox("增强策略", ["KNN流形邻域插值", "连续 Mixup 线性插值", "高斯扰动抖动 (Jittering)"])
    aug_factor = st.sidebar.slider("增强倍数 (扩充合成点倍数)", 1, 6, 2)
    noise_ratio = st.sidebar.slider("高斯扰动系数", 0.01, 0.10, 0.03, step=0.01) if "高斯" in aug_method else 0.03
else:
    aug_method = "未开启"
    aug_factor = 0
    noise_ratio = 0.0

test_ratio = st.sidebar.slider("独立测试集比例 (Test)", 0.1, 0.3, 0.15, step=0.05)
val_ratio = st.sidebar.slider("验证集比例 (Val，从剩余数据切分)", 0.1, 0.3, 0.2, step=0.05)

# -------------------------------------------------------------
# 数据准备、划分与数据增强（严防验证与测试集泄露）
# -------------------------------------------------------------
X_raw = df[feature_cols].values
y_raw = df[target_col].values.astype(float)
depth_vals = df[depth_col].values

# 对数变换处理
if enable_log_target:
    y_raw = np.log10(np.clip(y_raw, 1e-4, None))

# 步骤 1：划分出绝对独立的测试集
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X_raw, y_raw, np.arange(len(X_raw)), test_size=test_ratio, random_state=42
)
# 步骤 2：从剩余数据中划分真实训练集与真实验证集
X_train_orig, X_val, y_train_orig, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=val_ratio, random_state=42
)

# 步骤 3：数据增强（只对真实训练集施加，保证测试集和验证集 100% 为原始地质样本）
if enable_aug:
    X_train_final, y_train_final, is_synth = augment_dataset(
        X_train_orig, y_train_orig, method=aug_method, factor=aug_factor, noise_level=noise_ratio
    )
else:
    X_train_final, y_train_final = X_train_orig, y_train_orig
    is_synth = np.zeros(len(X_train_orig), dtype=bool)

# 步骤 4：标准化（在增强后的训练集上拟合变换器）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_final)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.transform(X_raw)

# -------------------------------------------------------------
# 模型实例化与拟合
# -------------------------------------------------------------
if selected_model_name == "随机森林回归 (Random Forest)":
    model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_split,
        random_state=42,
        n_jobs=-1
    )
    # 树模型使用原始尺度的 X，保留物理意义
    model.fit(X_train_final, y_train_final)
    y_tr_pred = model.predict(X_train_final)
    y_va_pred = model.predict(X_val)
    y_te_pred = model.predict(X_test)
    y_al_pred = model.predict(X_raw)

elif selected_model_name == "支持向量回归 (SVR)":
    model = SVR(C=svr_c, epsilon=svr_epsilon, kernel=svr_kernel)
    model.fit(X_train_s, y_train_final)
    y_tr_pred = model.predict(X_train_s)
    y_va_pred = model.predict(X_val_s)
    y_te_pred = model.predict(X_test_s)
    y_al_pred = model.predict(X_all_s)

elif selected_model_name == "BP 神经网络回归 (MLP Regressor)":
    model = MLPRegressor(
        hidden_layer_sizes=(mlp_h1, mlp_h2),
        learning_rate_init=mlp_lr,
        max_iter=mlp_max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_s, y_train_final)
    y_tr_pred = model.predict(X_train_s)
    y_va_pred = model.predict(X_val_s)
    y_te_pred = model.predict(X_test_s)
    y_al_pred = model.predict(X_all_s)

elif selected_model_name == "梯度提升回归树 (Gradient Boosting)":
    model = GradientBoostingRegressor(
        n_estimators=gb_n_estimators,
        learning_rate=gb_lr,
        max_depth=gb_max_depth,
        random_state=42
    )
    model.fit(X_train_final, y_train_final)
    y_tr_pred = model.predict(X_train_final)
    y_va_pred = model.predict(X_val)
    y_te_pred = model.predict(X_test)
    y_al_pred = model.predict(X_raw)

# 逆对数变换还原到真实地质单位
if enable_log_target:
    y_tr_eval, y_tr_hat = 10**y_train_final, 10**y_tr_pred
    y_va_eval, y_va_hat = 10**y_val, 10**y_va_pred
    y_te_eval, y_te_hat = 10**y_test, 10**y_te_pred
    y_al_eval, y_al_hat = 10**y_raw, 10**y_al_pred
else:
    y_tr_eval, y_tr_hat = y_train_final, y_tr_pred
    y_va_eval, y_va_hat = y_val, y_va_pred
    y_te_eval, y_te_hat = y_test, y_te_pred
    y_al_eval, y_al_hat = y_raw, y_al_pred

# -------------------------------------------------------------
# 界面呈现与下载模块
# -------------------------------------------------------------
st.title("🛢️ 储层物性机器学习多算法预测与增强平台")
st.markdown(f"当前任务：使用 **{selected_model_name}** 预测目标 **{target_col}**")

# 核心状态看板
m_c1, m_c2, m_c3, m_c4 = st.columns(4)
m_c1.metric("原始数据总量", len(df))
m_c2.metric("训练集原始点数", len(X_train_orig))
m_c3.metric("增强后训练集规模", len(X_train_final), delta=f"+{np.sum(is_synth)} 合成点")
m_c4.metric("独立盲测测试集", len(y_test))

# 数据导出保存
with st.expander("💾 导出与保存增强后的数据集 (TXT/CSV)", expanded=False):
    df_aug_train = pd.DataFrame(X_train_final, columns=feature_cols)
    df_aug_train[target_col] = y_tr_eval
    df_aug_train['样本属性'] = np.where(is_synth, '增强合成样本', '原始采集样本')
    
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        txt_buf = io.StringIO()
        df_aug_train.to_csv(txt_buf, sep='\t', index=False, encoding='utf-8')
        st.download_button(
            label="📥 下载增强训练集 (Tab制表符 TXT 格式)",
            data=txt_buf.getvalue().encode('utf-8-sig'),
            file_name=f"augmented_train_{target_col}.txt",
            mime="text/plain"
        )
    with c_dl2:
        df_full = df.copy()
        df_full['样本属性'] = '原始数据'
        df_syn = pd.DataFrame(X_train_final[is_synth], columns=feature_cols)
        df_syn[target_col] = (10**y_train_final[is_synth]) if enable_log_target else y_train_final[is_synth]
        df_syn['样本属性'] = '增强合成样本'
        df_merged = pd.concat([df_full, df_syn], ignore_index=True)
        
        csv_buf = io.StringIO()
        df_merged.to_csv(csv_buf, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载全量合并数据集 (CSV 格式)",
            data=csv_buf.getvalue().encode('utf-8-sig'),
            file_name=f"full_augmented_{target_col}.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# 评价指标
# -------------------------------------------------------------
def get_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }

m_tr = get_metrics(y_tr_eval, y_tr_hat)
m_va = get_metrics(y_va_eval, y_va_hat)
m_te = get_metrics(y_te_eval, y_te_hat)

st.markdown("### 📈 算法精度表现看板")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    col_m1.metric("训练集 R²", f"{m_tr['r2']:.3f}", delta=f"RMSE: {m_tr['rmse']:.3f}")
    st.caption(f"样本量: {len(y_train_final)} | MAE: {m_tr['mae']:.3f}")
with col_m2:
    col_m2.metric("验证集 R² (真实未增强样点)", f"{m_va['r2']:.3f}", delta=f"RMSE: {m_va['rmse']:.3f}")
    st.caption(f"样本量: {len(y_val)} | MAE: {m_va['mae']:.3f}")
with col_m3:
    col_m3.metric("测试集 R² (盲测泛化度)", f"{m_te['r2']:.3f}", delta=f"RMSE: {m_te['rmse']:.3f}")
    st.caption(f"样本量: {len(y_test)} | MAE: {m_te['mae']:.3f}")

st.markdown("---")

# -------------------------------------------------------------
# 多维度图件看板呈现
# -------------------------------------------------------------
st.subheader("🖼️ 预测过程与成果图件看板")

tab1, tab2, tab3 = st.tabs([
    "1. 散点交叉拟合与残差核密度",
    "2. 算法内生机理与样本学习曲线",
    "3. 连续单井测井剖面综合道图"
])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 散点图
    axes[0].scatter(y_tr_eval, y_tr_hat, color='#1f77b4', alpha=0.4, s=25, label=f'训练集 (R²={m_tr["r2"]:.2f})')
    axes[0].scatter(y_va_eval, y_va_hat, color='#ff7f0e', alpha=0.7, s=35, label=f'验证集 (R²={m_va["r2"]:.2f})')
    axes[0].scatter(y_te_eval, y_te_hat, color='#d62728', marker='^', alpha=0.9, s=45, label=f'测试集 (R²={m_te["r2"]:.2f})')
    
    min_v = min(np.min(y_al_eval), np.min(y_al_hat))
    max_v = max(np.max(y_al_eval), np.max(y_al_hat))
    axes[0].plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='1:1 理想对角线')
    if enable_log_target:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel(f"实测值: {target_col}", fontsize=11)
    axes[0].set_ylabel(f"预测值: {target_col}", fontsize=11)
    axes[0].set_title(f"{selected_model_name} 拟合散点交叉图", fontsize=12)
    axes[0].legend()
    
    # 残差核密度
    res_tr = y_tr_eval - y_tr_hat
    res_te = y_te_eval - y_te_hat
    sns.kdeplot(res_tr, ax=axes[1], fill=True, color='#1f77b4', label='训练集残差')
    sns.kdeplot(res_te, ax=axes[1], fill=True, color='#d62728', label='测试集残差')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_xlabel("预测误差残差 (实测 - 预测)", fontsize=11)
    axes[1].set_ylabel("残差概率密度", fontsize=11)
    axes[1].set_title("预测残差正态分布形态", fontsize=12)
    axes[1].legend()
    
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：算法机理图件
    if hasattr(model, "feature_importances_"):
        # 树模型展示特征重要性
        imp = model.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        sorted_cols = [feature_cols[i] for i in sorted_idx]
        sns.barplot(x=imp[sorted_idx], y=sorted_cols, ax=axes[0], palette="mako")
        axes[0].set_title(f"{selected_model_name} 特征重要性贡献排序", fontsize=12)
        axes[0].set_xlabel("重要性分值", fontsize=11)
    elif selected_model_name == "支持向量回归 (SVR)":
        n_sv = len(model.support_)
        axes[0].pie(
            [n_sv, len(X_train_final) - n_sv],
            labels=['支持向量样本', '其余非约束样本'],
            autopct='%1.1f%%',
            colors=['#e15759', '#76b7b2'],
            startangle=90
        )
        axes[0].set_title(f"SVR 支持向量样本分布占比 (数量: {n_sv})", fontsize=12)
    elif selected_model_name == "BP 神经网络回归 (MLP Regressor)":
        axes[0].plot(model.loss_curve_, color='#2ca02c', lw=2)
        axes[0].set_title("BP 神经网络损失迭代曲线 (Loss Curve)", fontsize=12)
        axes[0].set_xlabel("迭代轮数 (Epochs)", fontsize=11)
        axes[0].set_ylabel("Loss (MSE)", fontsize=11)

    # 右图：样本容量学习曲线 (过拟合/欠拟合检测)
    X_curve_in = X_temp if hasattr(model, "feature_importances_") else scaler.transform(X_temp)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_curve_in, y_temp, cv=4, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='r2'
    )
    axes[1].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#1f77b4', label='训练集得分')
    axes[1].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='#ff7f0e', label='交叉验证得分')
    axes[1].set_title("样本容量与泛化能力学习曲线", fontsize=12)
    axes[1].set_xlabel("训练样本数量", fontsize=11)
    axes[1].set_ylabel("判定系数 R²", fontsize=11)
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].legend(loc="lower right")
    
    st.pyplot(fig)
    plt.close()

with tab3:
    tracks = min(3, len(feature_cols)) + 2
    fig, axes = plt.subplots(1, tracks, figsize=(4 * tracks, 8), sharey=True)
    
    # 特征道
    for i in range(tracks - 2):
        col_name = feature_cols[i]
        axes[i].plot(df[col_name], depth_vals, color=sns.color_palette("tab10")[i], lw=1.2)
        axes[i].set_xlabel(col_name, fontsize=10)
        axes[i].grid(True, linestyle=':')
    
    # 目标预测对比道
    pred_track = axes[tracks - 2]
    pred_track.scatter(y_al_eval, depth_vals, color='black', s=8, alpha=0.5, label='实测值')
    pred_track.plot(y_al_hat, depth_vals, color='red', lw=1.5, label='模型预测')
    pred_track.set_xlabel(target_col, fontsize=10)
    if enable_log_target:
        pred_track.set_xscale('log')
    pred_track.legend(loc='upper right')
    pred_track.grid(True, linestyle=':')
    
    # 绝对误差道
    err_track = axes[tracks - 1]
    err_val = np.abs(y_al_eval - y_al_hat)
    err_track.fill_betweenx(depth_vals, 0, err_val, color='orange', alpha=0.5)
    err_track.plot(err_val, depth_vals, color='darkorange', lw=1)
    err_track.set_xlabel("绝对预测误差", fontsize=10)
    err_track.grid(True, linestyle=':')

    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"井深 ({depth_col})", fontsize=12)
    fig.suptitle(f"储层纵向剖面连续测井解释图 [{target_col}] - 算法: {selected_model_name}", fontsize=14, y=0.98)
    
    st.pyplot(fig)
    plt.close()
