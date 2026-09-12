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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------------------------------------
# 页面配置与跨平台中文字体
# -------------------------------------------------------------
st.set_page_config(
    page_title="储层参数机器学习预测与数据增强系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# 文件读取引擎
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
def generate_synthetic_data(n_samples=250):  # 默认较少点，突出增强必要性
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
# 地质测井专业数据增强算法
# -------------------------------------------------------------
def augment_dataset(X_tr, y_tr, method="KNN流形邻域插值", factor=2, noise_level=0.03):
    """
    X_tr: 原始训练特征矩阵
    y_tr: 原始训练目标向量
    factor: 扩充倍数 (例如 factor=2 则新增 2*N 个合成点)
    """
    n_samples, n_features = X_tr.shape
    n_synthetic = int(n_samples * factor)
    np.random.seed(42)
    
    if factor <= 0 or n_synthetic == 0:
        return X_tr, y_tr, np.zeros(n_samples, dtype=bool)

    if method == "高斯扰动抖动 (Jittering)":
        # 计算每个特征列的标准差
        std_x = np.std(X_tr, axis=0, keepdims=True)
        std_y = np.std(y_tr)
        
        # 随机抽取母本
        idx = np.random.choice(n_samples, n_synthetic, replace=True)
        x_base = X_tr[idx]
        y_base = y_tr[idx]
        
        synth_X = x_base + np.random.normal(0, noise_level, (n_synthetic, n_features)) * std_x
        synth_y = y_base + np.random.normal(0, noise_level, n_synthetic) * std_y

    elif method == "连续 Mixup 线性插值":
        # 两两随机配对加权融合
        idx1 = np.random.choice(n_samples, n_synthetic, replace=True)
        idx2 = np.random.choice(n_samples, n_synthetic, replace=True)
        
        lam = np.random.beta(0.4, 0.4, size=(n_synthetic, 1))
        synth_X = lam * X_tr[idx1] + (1 - lam) * X_tr[idx2]
        synth_y = (lam.ravel() * y_tr[idx1]) + ((1 - lam.ravel()) * y_tr[idx2])

    else:  # KNN 流形邻域插值 (类似回归任务的 SMOTE-R)
        k = min(5, n_samples - 1)
        knn = NearestNeighbors(n_neighbors=k + 1).fit(X_tr)
        distances, indices = knn.kneighbors(X_tr)
        
        base_indices = np.random.choice(n_samples, n_synthetic, replace=True)
        # 随机选择近邻中的一个（排除自身 index 0）
        neighbor_col = np.random.randint(1, k + 1, size=n_synthetic)
        neighbor_indices = indices[base_indices, neighbor_col]
        
        diff = X_tr[neighbor_indices] - X_tr[base_indices]
        rand_weights = np.random.uniform(0.05, 0.95, size=(n_synthetic, 1))
        
        synth_X = X_tr[base_indices] + rand_weights * diff
        synth_y = y_tr[base_indices] + rand_weights.ravel() * (y_tr[neighbor_indices] - y_tr[base_indices])
        
        # 添加极小量高斯噪声防止多重共线性
        synth_X += np.random.normal(0, 0.01 * np.std(X_tr, axis=0), synth_X.shape)
        synth_y += np.random.normal(0, 0.01 * np.std(y_tr), synth_y.shape)

    # 拼接合成数据
    augmented_X = np.vstack([X_tr, synth_X])
    augmented_y = np.concatenate([y_tr, synth_y])
    
    # 标记是否为生成样本 (False=原始, True=增强生成)
    is_synth = np.concatenate([np.zeros(n_samples, dtype=bool), np.ones(n_synthetic, dtype=bool)])
    
    return augmented_X, augmented_y, is_synth

# -------------------------------------------------------------
# 边栏配置
# -------------------------------------------------------------
st.sidebar.title("🛢️ 储层参数预测与增强系统")

data_source = st.sidebar.radio("数据来源", ["上传本地 TXT/CSV 文件", "使用示例测井数据"])
if data_source == "上传本地 TXT/CSV 文件":
    file = st.sidebar.file_uploader("上传测井数据 TXT/CSV", type=["txt", "csv"])
    if file is not None:
        try:
            df = load_txt_file(file)
            st.sidebar.success(f"成功读取：{df.shape[0]} 行 × {df.shape[1]} 列")
        except Exception as e:
            st.sidebar.error(f"解析错误: {e}")
            df = generate_synthetic_data()
    else:
        st.sidebar.info("未上传文件，已自动加载示例数据。")
        df = generate_synthetic_data()
else:
    df = generate_synthetic_data()

df = df.dropna().reset_index(drop=True)
all_columns = df.columns.tolist()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 建模任务与变量指定")
task = st.sidebar.selectbox(
    "预测任务",
    [
        "基于随机森林的储层 TOC 预测",
        "基于支持向量回归 (SVR) 的储层孔隙度预测",
        "基于 BP 神经网络的储层渗透率预测"
    ]
)

depth_guess = next((c for c in all_columns if any(k in c.lower() for k in ['depth', '深', '井深'])), all_columns[0])
depth_col = st.sidebar.selectbox("井深列", all_columns, index=all_columns.index(depth_guess))

if "TOC" in task:
    target_guess = next((c for c in all_columns if 'toc' in c.lower() or '碳' in c), all_columns[-1])
elif "孔隙度" in task:
    target_guess = next((c for c in all_columns if any(k in c.lower() for k in ['por', '孔隙'])), all_columns[-1])
else:
    target_guess = next((c for c in all_columns if any(k in c.lower() for k in ['perm', '渗透'])), all_columns[-1])

target_col = st.sidebar.selectbox("预测目标列", all_columns, index=all_columns.index(target_guess))
features = [c for c in all_columns if c not in [depth_col, target_col]]
feature_cols = st.sidebar.multiselect("特征测井列", features, default=features)

if not feature_cols:
    st.error("请选择至少一个特征列！")
    st.stop()

# -------------------------------------------------------------
# 数据增强配置
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ 训练集数据增强机制")
enable_aug = st.sidebar.checkbox("开启训练集数据增强 (提升小样本泛化)", value=True)

if enable_aug:
    aug_method = st.sidebar.selectbox(
        "增强算法模式",
        ["KNN流形邻域插值", "连续 Mixup 线性插值", "高斯扰动抖动 (Jittering)"]
    )
    aug_factor = st.sidebar.slider("增强倍数 (生成 N 倍合成训练样本)", 1, 6, 2)
    noise_ratio = st.sidebar.slider("扰动强度系数", 0.01, 0.10, 0.03, step=0.01) if "高斯" in aug_method else 0.03
else:
    aug_factor = 0
    aug_method = "未开启"
    noise_ratio = 0.0

# 数据集切分
st.sidebar.markdown("---")
st.sidebar.subheader("📐 数据集划分")
test_ratio = st.sidebar.slider("独立测试集比例 (Test)", 0.1, 0.3, 0.15, step=0.05)
val_ratio = st.sidebar.slider("验证集比例 (Val)", 0.1, 0.3, 0.2, step=0.05)

# -------------------------------------------------------------
# 数据划分与规范化流
# -------------------------------------------------------------
X_raw = df[feature_cols].values
y_raw = df[target_col].values
depth_vals = df[depth_col].values

is_log_perm = False
if "渗透率" in task:
    is_log_perm = True
    y_raw = np.log10(np.clip(y_raw, 1e-4, None))

# 1. 划分独立测试集与临时集
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X_raw, y_raw, np.arange(len(X_raw)), test_size=test_ratio, random_state=42
)
# 2. 划分真实训练集与独立验证集
X_train_orig, X_val, y_train_orig, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=val_ratio, random_state=42
)

# 3. 仅对训练集进行数据增强（避免验证与测试泄露）
if enable_aug:
    X_train_final, y_train_final, is_synth = augment_dataset(
        X_train_orig, y_train_orig, method=aug_method, factor=aug_factor, noise_level=noise_ratio
    )
else:
    X_train_final, y_train_final = X_train_orig, y_train_orig
    is_synth = np.zeros(len(X_train_orig), dtype=bool)

# 特征归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X_raw)

# -------------------------------------------------------------
# 模型训练
# -------------------------------------------------------------
if "随机森林" in task:
    # 树模型针对小样本增强后适当调深 max_depth
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_final, y_train_final)
    y_tr_pred = model.predict(X_train_final)
    y_va_pred = model.predict(X_val)
    y_te_pred = model.predict(X_test)
    y_al_pred = model.predict(X_raw)
elif "支持向量" in task:
    model = SVR(C=15.0, epsilon=0.08, kernel='rbf')
    model.fit(X_train_scaled, y_train_final)
    y_tr_pred = model.predict(X_train_scaled)
    y_va_pred = model.predict(X_val_scaled)
    y_te_pred = model.predict(X_test_scaled)
    y_al_pred = model.predict(X_all_scaled)
else:
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        learning_rate_init=0.002,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train_final)
    y_tr_pred = model.predict(X_train_scaled)
    y_va_pred = model.predict(X_val_scaled)
    y_te_pred = model.predict(X_test_scaled)
    y_al_pred = model.predict(X_all_scaled)

# 还原渗透率尺度
if is_log_perm:
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
# 页面内容与下载模块
# -------------------------------------------------------------
st.title("🛢️ 储层物性机器学习预测与数据增强工作台")

# 数据增强成果快速看板
c1, c2, c3, c4 = st.columns(4)
c1.metric("原始总样本点数", len(df))
c2.metric("训练集原始大小", len(X_train_orig))
c3.metric("增强后训练集大小", len(X_train_final), delta=f"+{np.sum(is_synth)} 点 ({aug_method if enable_aug else '未启用'})")
c4.metric("严格独立测试集", len(y_test))

# 数据导出保存选项卡
with st.expander("💾 保存与导出增强后的数据集文件 (支持 TXT / CSV 格式)", expanded=True):
    # 构建增强训练集 DataFrame
    df_aug_train = pd.DataFrame(X_train_final, columns=feature_cols)
    if is_log_perm:
        df_aug_train[target_col] = 10**y_train_final
    else:
        df_aug_train[target_col] = y_train_final
    df_aug_train['样本类型'] = np.where(is_synth, '增强合成点', '原始采样点')
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.write("**选项 1：导出增强训练集 (仅含训练用合成与真实样本)**")
        txt_train_buffer = io.StringIO()
        df_aug_train.to_csv(txt_train_buffer, sep='\t', index=False, encoding='utf-8')
        st.download_button(
            label="📥 下载增强训练集 (Tab 分隔 TXT 文件)",
            data=txt_train_buffer.getvalue().encode('utf-8-sig'),
            file_name="augmented_train_data.txt",
            mime="text/plain"
        )
    with col_dl2:
        st.write("**选项 2：导出全量增强集 (包含测试/验证原点与合成点完整表)**")
        # 组装完整表
        df_full = df.copy()
        df_full['样本集归属'] = '原始数据'
        df_syn_part = pd.DataFrame(X_train_final[is_synth], columns=feature_cols)
        if is_log_perm:
            df_syn_part[target_col] = 10**y_train_final[is_synth]
        else:
            df_syn_part[target_col] = y_train_final[is_synth]
        df_syn_part['样本集归属'] = '增强合成点'
        df_full_augmented = pd.concat([df_full, df_syn_part], ignore_index=True)
        
        csv_buffer = io.StringIO()
        df_full_augmented.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载全量增强合并数据集 (CSV 格式)",
            data=csv_buffer.getvalue().encode('utf-8-sig'),
            file_name="full_augmented_reservoir_data.csv",
            mime="text/csv"
        )
    st.caption("提示：导出的文件自动采用 UTF-8 BOM 编码，Excel、Python 及各类地质专业软件均可直接无乱码打开。")

# -------------------------------------------------------------
# 精度评估指标
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

st.markdown("### 📈 泛化与验证精度表现")
mc1, mc2, mc3 = st.columns(3)
mc1.metric("训练集拟合度 R²", f"{m_tr['r2']:.3f}", delta=f"RMSE: {m_tr['rmse']:.3f}")
mc2.metric("验证集 R² (真实原始点)", f"{m_va['r2']:.3f}", delta=f"RMSE: {m_va['rmse']:.3f}")
mc3.metric("独立测试集 R² (未参与增强)", f"{m_te['r2']:.3f}", delta=f"RMSE: {m_te['rmse']:.3f}")

# -------------------------------------------------------------
# 成果图件看板
# -------------------------------------------------------------
st.markdown("---")
st.subheader("🖼️ 过程与成果图件看板")

tab1, tab2, tab3 = st.tabs([
    "数据增强特征空间分布检查",
    "预测精度散点与残差分析",
    "连续测井道预测大样图"
])

with tab1:
    st.markdown("#### 🔬 增强样本合理性检查：原始样本 vs 合成样本分布")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 选取前两个重要特征进行投影分布对比
    f1 = feature_cols[0]
    f2 = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    idx_f1 = feature_cols.index(f1)
    idx_f2 = feature_cols.index(f2)
    
    # 特征空间散点对比
    axes[0].scatter(X_train_final[~is_synth, idx_f1], X_train_final[~is_synth, idx_f2], 
                    color='#1f77b4', s=45, alpha=0.8, label='原始真实点')
    if np.any(is_synth):
        axes[0].scatter(X_train_final[is_synth, idx_f1], X_train_final[is_synth, idx_f2], 
                        color='#e15759', s=35, marker='x', alpha=0.7, label='增强合成点')
    axes[0].set_xlabel(f"{f1}", fontsize=11)
    axes[0].set_ylabel(f"{f2}", fontsize=11)
    axes[0].set_title(f"特征空间点分布形态对比 ({f1} vs {f2})", fontsize=12)
    axes[0].legend()
    
    # 目标变量分布对比（验证增强后的目标值分布是否畸变）
    sns.kdeplot(y_train_orig if not is_log_perm else 10**y_train_orig, 
                ax=axes[1], color='#1f77b4', fill=True, label='原始目标分布')
    if np.any(is_synth):
        sns.kdeplot(y_train_final[is_synth] if not is_log_perm else 10**y_train_final[is_synth], 
                    ax=axes[1], color='#e15759', fill=True, label='合成目标分布')
    axes[1].set_xlabel(f"{target_col}", fontsize=11)
    axes[1].set_ylabel("密度 (Density)", fontsize=11)
    axes[1].set_title(f"{target_col} 概率密度平滑性检验", fontsize=12)
    axes[1].legend()
    
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 散点交叉图 (实测 vs 预测)
    axes[0].scatter(y_tr_eval, y_tr_hat, color='#1f77b4', alpha=0.5, s=25, label=f'训练集 (R²={m_tr["r2"]:.2f})')
    axes[0].scatter(y_va_eval, y_va_hat, color='#ff7f0e', alpha=0.7, s=35, label=f'验证集 (R²={m_va["r2"]:.2f})')
    axes[0].scatter(y_te_eval, y_te_hat, color='#d62728', marker='^', alpha=0.9, s=45, label=f'测试集 (R²={m_te["r2"]:.2f})')
    
    min_v = min(np.min(y_al_eval), np.min(y_al_hat))
    max_v = max(np.max(y_al_eval), np.max(y_al_hat))
    axes[0].plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='1:1 理想基准线')
    if is_log_perm:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel(f"实测值 {target_col}", fontsize=11)
    axes[0].set_ylabel(f"预测值 {target_col}", fontsize=11)
    axes[0].set_title("实测值 vs 预测值交叉拟合图", fontsize=12)
    axes[0].legend()
    
    # 残差核密度图
    res_train = y_tr_eval - y_tr_hat
    res_test = y_te_eval - y_te_hat
    sns.kdeplot(res_train, ax=axes[1], fill=True, color='#1f77b4', label='训练残差')
    sns.kdeplot(res_test, ax=axes[1], fill=True, color='#d62728', label='独立测试残差')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_xlabel("预测误差残差 (实测 - 预测)", fontsize=11)
    axes[1].set_ylabel("残差密度", fontsize=11)
    axes[1].set_title("预测残差正态分布与偏差检查", fontsize=12)
    axes[1].legend()
    
    st.pyplot(fig)
    plt.close()

with tab3:
    tracks = min(3, len(feature_cols)) + 2
    fig, axes = plt.subplots(1, tracks, figsize=(4 * tracks, 8), sharey=True)
    
    for i in range(tracks - 2):
        col_name = feature_cols[i]
        axes[i].plot(df[col_name], depth_vals, color=sns.color_palette("tab10")[i], lw=1.2)
        axes[i].set_xlabel(col_name, fontsize=10)
        axes[i].grid(True, linestyle=':')
    
    pred_ax = axes[tracks - 2]
    pred_ax.scatter(y_al_eval, depth_vals, color='black', s=8, alpha=0.5, label='实测原始点')
    pred_ax.plot(y_al_hat, depth_vals, color='red', lw=1.5, label='增强后模型预测')
    pred_ax.set_xlabel(target_col, fontsize=10)
    if is_log_perm:
        pred_ax.set_xscale('log')
    pred_ax.legend(loc='upper right')
    pred_ax.grid(True, linestyle=':')
    
    err_ax = axes[tracks - 1]
    err_val = np.abs(y_al_eval - y_al_hat)
    err_ax.fill_betweenx(depth_vals, 0, err_val, color='orange', alpha=0.5)
    err_ax.plot(err_val, depth_vals, color='darkorange', lw=1)
    err_ax.set_xlabel("绝对误差道", fontsize=10)
    err_ax.grid(True, linestyle=':')
    
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"井深 ({depth_col})", fontsize=12)
    fig.suptitle(f"储层连续单井测井解释与预测对比大样图 [{target_col}]", fontsize=14, y=0.98)
    
    st.pyplot(fig)
    plt.close()
