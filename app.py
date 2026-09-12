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

# 1. 基础页面设置
st.set_page_config(
    page_title="储层参数机器学习平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 字体配置（不依赖外部包，自动探测系统可用中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 2. 文件读取
def load_txt_file(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    enc_detect = chardet.detect(raw_bytes[:10000])
    encoding = enc_detect.get('encoding', 'utf-8') or 'gb18030'
    
    for enc in [encoding, 'utf-8', 'gb18030', 'gbk', 'utf-8-sig']:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=r'\s+|,|\t', engine='python')
            if df.shape[1] <= 1:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            return df
        except Exception:
            continue
    return None

def generate_synthetic_data(n_samples=200):
    np.random.seed(42)
    depth = np.linspace(2100, 2400, n_samples)
    gr = np.clip(45 + 50 * np.sin(depth / 40) + np.random.normal(0, 6, n_samples), 20, 180)
    rhob = np.clip(2.68 - 0.003 * gr + np.random.normal(0, 0.03, n_samples), 2.15, 2.8)
    dt = np.clip(185 + (2.7 - rhob) * 110 + np.random.normal(0, 5, n_samples), 170, 310)
    nphi = np.clip(0.04 + 0.0016 * gr + np.random.normal(0, 0.015, n_samples), 0.02, 0.42)
    rt = 10 ** (np.random.uniform(0.6, 2.2, n_samples) + 0.008 * gr / 10)
    
    toc = np.clip(0.028 * gr + 0.009 * dt - 1.15 * rhob + np.random.normal(0, 0.25, n_samples), 0.2, 8.0)
    por = np.clip(((2.65 - rhob) / 1.65) * 65 + nphi * 28 + np.random.normal(0, 0.7, n_samples), 1.5, 28.0)
    perm = np.clip((10 ** (0.17 * por - 1.4)) * np.exp(np.random.normal(0, 0.35, n_samples)), 0.01, 2000.0)
    
    return pd.DataFrame({
        '井深(m)': depth,
        '自然伽马_GR(API)': gr,
        '声波时差_DT(us/ft)': dt,
        '补偿密度_RHOB(g/cm3)': rhob,
        '补偿中子_NPHI(v/v)': nphi,
        '深侧向电阻率_RT(ohm.m)': rt,
        '总有机碳_TOC(%)': toc,
        '孔隙度_POR(%)': por,
        '渗透率_PERM(mD)': perm
    })

# 3. 数据增强引擎 (单线程纯 numpy 运算，杜绝容器死锁)
def augment_dataset(X_tr, y_tr, method="KNN流形邻域插值", factor=2, noise_level=0.03):
    X_tr = np.ascontiguousarray(X_tr, dtype=np.float64)
    y_tr = np.ascontiguousarray(y_tr, dtype=np.float64)
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
    else:
        k = min(5, max(1, n_samples - 1))
        knn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X_tr)
        _, indices = knn.kneighbors(X_tr)
        base_idx = np.random.choice(n_samples, n_synthetic, replace=True)
        neighbor_col = np.random.randint(1, k + 1, size=n_synthetic)
        neighbor_idx = indices[base_idx, neighbor_col]
        diff = X_tr[neighbor_idx] - X_tr[base_idx]
        weights = np.random.uniform(0.05, 0.95, size=(n_synthetic, 1))
        synth_X = X_tr[base_idx] + weights * diff
        synth_y = y_tr[base_idx] + weights.ravel() * (y_tr[neighbor_idx] - y_tr[base_idx])
        synth_X += np.random.normal(0, 0.01 * np.std(X_tr, axis=0), synth_X.shape)
        synth_y += np.random.normal(0, 0.01 * np.std(y_tr), synth_y.shape)

    return np.vstack([X_tr, synth_X]), np.concatenate([y_tr, synth_y]), np.concatenate([np.zeros(n_samples, dtype=bool), np.ones(n_synthetic, dtype=bool)])

# 4. 侧边栏与数据流
st.sidebar.title("🛢️ 储层参数预测工作台")
data_mode = st.sidebar.radio("数据来源", ["上传本地 TXT/CSV 文件", "使用示例测井数据"])

raw_df = None
if data_mode == "上传本地 TXT/CSV 文件":
    up_file = st.sidebar.file_uploader("上传测井数据 TXT/CSV", type=["txt", "csv"])
    if up_file is not None:
        raw_df = load_txt_file(up_file)
        if raw_df is None:
            st.sidebar.error("文件解码失败，自动切换为示例数据。")
            raw_df = generate_synthetic_data()
        else:
            st.sidebar.success(f"载入成功：{raw_df.shape[0]} 行 × {raw_df.shape[1]} 列")
    else:
        raw_df = generate_synthetic_data()
else:
    raw_df = generate_synthetic_data()

# 拷贝数据，避免原地篡改
df = raw_df.copy()
all_cols = df.columns.tolist()

st.sidebar.subheader("🎯 1. 变量选择")
depth_guess = next((c for c in all_cols if any(k in str(c).lower() for k in ['depth', '深', '井深'])), all_cols[0])
depth_col = st.sidebar.selectbox("井深列", all_cols, index=all_cols.index(depth_guess))

# 自动筛选数值列，剔除井名等非数值列
numeric_cols = []
for c in all_cols:
    if c == depth_col:
        continue
    series = pd.to_numeric(df[c], errors='coerce')
    if series.notna().sum() > len(df) * 0.5:
        numeric_cols.append(c)
        df[c] = series

if not numeric_cols:
    st.error("文件中未找到有效的数值测井曲线！")
    st.stop()

target_col = st.sidebar.selectbox("预测目标列 (Target)", numeric_cols, index=len(numeric_cols)-1)
feature_candidates = [c for c in numeric_cols if c != target_col]
feature_cols = st.sidebar.multiselect("输入特征列 (Features)", feature_candidates, default=feature_candidates)

if not feature_cols:
    st.warning("请在左侧至少勾选一个特征列！")
    st.stop()

# 清洗脏数据
clean_mask = df[feature_cols + [target_col, depth_col]].notna().all(axis=1)
df_clean = df[clean_mask].copy()

if len(df_clean) < 10:
    st.error("有效数值样本过少（不足 10 行），无法建模！")
    st.stop()

enable_log_target = st.sidebar.checkbox("启用目标变量 log10 对数变换", value=('perm' in target_col.lower() or '渗透' in target_col))

# 模型选择
st.sidebar.subheader("🧠 2. 回归模型选择")
selected_model_name = st.sidebar.selectbox(
    "选择拟合回归模型",
    ["随机森林回归 (Random Forest)", "支持向量回归 (SVR)", "BP 神经网络回归 (MLP)", "梯度提升树 (Gradient Boosting)"]
)

# 数据增强设置
st.sidebar.subheader("⚡ 3. 增强与划分")
enable_aug = st.sidebar.checkbox("开启训练集数据增强", value=True)
aug_method = st.sidebar.selectbox("增强策略", ["KNN流形邻域插值", "连续 Mixup 线性插值", "高斯扰动抖动 (Jittering)"]) if enable_aug else "关闭"
aug_factor = st.sidebar.slider("增强倍数", 1, 5, 2) if enable_aug else 0

test_ratio = st.sidebar.slider("独立测试集比例", 0.1, 0.3, 0.15, step=0.05)
val_ratio = st.sidebar.slider("验证集比例", 0.1, 0.3, 0.2, step=0.05)

# 5. 核心计算
X_raw = df_clean[feature_cols].to_numpy(dtype=np.float64)
y_raw = df_clean[target_col].to_numpy(dtype=np.float64)
depth_vals = df_clean[depth_col].to_numpy(dtype=np.float64)

if enable_log_target:
    y_raw = np.log10(np.clip(y_raw, 1e-4, None))

X_temp, X_test, y_temp, y_test = train_test_split(X_raw, y_raw, test_size=test_ratio, random_state=42)
X_train_orig, X_val, y_train_orig, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

if enable_aug:
    X_train, y_train, is_synth = augment_dataset(X_train_orig, y_train_orig, method=aug_method, factor=aug_factor)
else:
    X_train, y_train, is_synth = X_train_orig, y_train_orig, np.zeros(len(X_train_orig), dtype=bool)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.transform(X_raw)

# 实例化模型（必须固定 n_jobs=1，防止云端多进程锁死）
if "随机森林" in selected_model_name:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    y_tr_pred, y_va_pred, y_te_pred, y_al_pred = model.predict(X_train), model.predict(X_val), model.predict(X_test), model.predict(X_raw)
elif "支持向量" in selected_model_name:
    model = SVR(C=10.0, epsilon=0.1)
    model.fit(X_train_s, y_train)
    y_tr_pred, y_va_pred, y_te_pred, y_al_pred = model.predict(X_train_s), model.predict(X_val_s), model.predict(X_test_s), model.predict(X_all_s)
elif "BP" in selected_model_name:
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42, early_stopping=True)
    model.fit(X_train_s, y_train)
    y_tr_pred, y_va_pred, y_te_pred, y_al_pred = model.predict(X_train_s), model.predict(X_val_s), model.predict(X_test_s), model.predict(X_all_s)
else:
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_tr_pred, y_va_pred, y_te_pred, y_al_pred = model.predict(X_train), model.predict(X_val), model.predict(X_test), model.predict(X_raw)

# 还原尺度
if enable_log_target:
    y_tr_true, y_tr_hat = 10**y_train, 10**y_tr_pred
    y_va_true, y_va_hat = 10**y_val, 10**y_va_pred
    y_te_true, y_te_hat = 10**y_test, 10**y_te_pred
    y_al_true, y_al_hat = 10**y_raw, 10**y_al_pred
else:
    y_tr_true, y_tr_hat = y_train, y_tr_pred
    y_va_true, y_va_hat = y_val, y_va_pred
    y_te_true, y_te_hat = y_test, y_te_pred
    y_al_true, y_al_hat = y_raw, y_al_pred

# 6. UI 展示
st.title("🛢️ 储层物性机器学习预测与增强平台")
st.markdown(f"当前模型：**{selected_model_name}** ｜ 预测目标：**{target_col}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("总样点数", len(df_clean))
c2.metric("训练集原样点", len(X_train_orig))
c3.metric("增强后训练规模", len(X_train), delta=f"+{np.sum(is_synth)}")
c4.metric("独立盲测集", len(y_test))

# 导出功能
with st.expander("💾 保存增强后的数据集 (TXT/CSV)", expanded=False):
    df_aug = pd.DataFrame(X_train, columns=feature_cols)
    df_aug[target_col] = y_tr_true
    df_aug['样本类型'] = np.where(is_synth, '增强合成点', '原始点')
    txt_io = io.StringIO()
    df_aug.to_csv(txt_io, sep='\t', index=False)
    st.download_button("📥 下载增强训练集 (TXT)", txt_io.getvalue().encode('utf-8-sig'), file_name="augmented_data.txt")

# 指标
def calc_metrics(yt, yp):
    return r2_score(yt, yp), np.sqrt(mean_squared_error(yt, yp)), mean_absolute_error(yt, yp)

r2_tr, rmse_tr, mae_tr = calc_metrics(y_tr_true, y_tr_hat)
r2_va, rmse_va, mae_va = calc_metrics(y_va_true, y_va_hat)
r2_te, rmse_te, mae_te = calc_metrics(y_te_true, y_te_hat)

st.markdown("### 📈 精度指标评估")
m1, m2, m3 = st.columns(3)
m1.metric("训练集 R²", f"{r2_tr:.3f}", delta=f"RMSE: {rmse_tr:.3f}")
m2.metric("验证集 R²", f"{r2_va:.3f}", delta=f"RMSE: {rmse_va:.3f}")
m3.metric("独立测试集 R² (真实泛化)", f"{r2_te:.3f}", delta=f"RMSE: {rmse_te:.3f}")

# 图件展示
st.markdown("---")
tab1, tab2 = st.tabs(["散点交叉对比与残差图", "连续测井预测大样图"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_tr_true, y_tr_hat, alpha=0.4, label=f'Train (R²={r2_tr:.2f})')
    axes[0].scatter(y_va_true, y_va_hat, alpha=0.7, label=f'Val (R²={r2_va:.2f})')
    axes[0].scatter(y_te_true, y_te_hat, marker='^', alpha=0.9, label=f'Test (R²={r2_te:.2f})')
    mv_min, mv_max = min(np.min(y_al_true), np.min(y_al_hat)), max(np.max(y_al_true), np.max(y_al_hat))
    axes[0].plot([mv_min, mv_max], [mv_min, mv_max], 'k--', lw=1.5)
    if enable_log_target:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel(f"Measured: {target_col}")
    axes[0].set_ylabel(f"Predicted: {target_col}")
    axes[0].legend()
    
    sns.kdeplot(y_tr_true - y_tr_hat, ax=axes[1], fill=True, label='Train Residuals')
    sns.kdeplot(y_te_true - y_te_hat, ax=axes[1], fill=True, label='Test Residuals')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].legend()
    st.pyplot(fig)
    plt.close()

with tab2:
    tracks = min(3, len(feature_cols)) + 2
    fig, axes = plt.subplots(1, tracks, figsize=(3.5 * tracks, 7), sharey=True)
    for i in range(tracks - 2):
        axes[i].plot(df_clean[feature_cols[i]], depth_vals, lw=1.2)
        axes[i].set_xlabel(feature_cols[i])
        axes[i].grid(True, linestyle=':')
    
    axes[tracks - 2].scatter(y_al_true, depth_vals, color='black', s=8, alpha=0.5, label='Measured')
    axes[tracks - 2].plot(y_al_hat, depth_vals, color='red', lw=1.5, label='Predicted')
    axes[tracks - 2].set_xlabel(target_col)
    if enable_log_target:
        axes[tracks - 2].set_xscale('log')
    axes[tracks - 2].legend(loc='upper right')
    
    diff = np.abs(y_al_true - y_al_hat)
    axes[tracks - 1].fill_betweenx(depth_vals, 0, diff, color='orange', alpha=0.5)
    axes[tracks - 1].set_xlabel("Abs Error")
    
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"Depth ({depth_col})")
    st.pyplot(fig)
    plt.close()
