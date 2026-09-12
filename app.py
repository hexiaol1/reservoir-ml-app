import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import chardet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. 基础页面设置
st.set_page_config(
    page_title="储层物性机器学习双文件盲井预测平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 2. 稳健编码解析器
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
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

# 3. 示例数据生成器（训练集 + 独立盲井）
def generate_demo_datasets():
    np.random.seed(42)
    # 训练集：2口开发井模拟
    depth_tr = np.linspace(2100, 2450, 240)
    gr_tr = np.clip(45 + 55 * np.sin(depth_tr / 35) + np.random.normal(0, 7, 240), 20, 180)
    rhob_tr = np.clip(2.68 - 0.003 * gr_tr + np.random.normal(0, 0.03, 240), 2.15, 2.8)
    dt_tr = np.clip(185 + (2.7 - rhob_tr) * 110 + np.random.normal(0, 5, 240), 170, 310)
    nphi_tr = np.clip(0.04 + 0.0016 * gr_tr + np.random.normal(0, 0.015, 240), 0.02, 0.42)
    rt_tr = 10 ** (np.random.uniform(0.6, 2.2, 240) + 0.008 * gr_tr / 10)
    
    toc_tr = np.clip(0.028 * gr_tr + 0.009 * dt_tr - 1.15 * rhob_tr + np.random.normal(0, 0.25, 240), 0.2, 8.0)
    por_tr = np.clip(((2.65 - rhob_tr) / 1.65) * 65 + nphi_tr * 28 + np.random.normal(0, 0.7, 240), 1.5, 28.0)
    perm_tr = np.clip((10 ** (0.17 * por_tr - 1.4)) * np.exp(np.random.normal(0, 0.35, 240)), 0.01, 2000.0)

    df_train = pd.DataFrame({
        '井深(m)': depth_tr, '自然伽马_GR': gr_tr, '声波时差_DT': dt_tr,
        '补偿密度_RHOB': rhob_tr, '补偿中子_NPHI': nphi_tr, '深侧向电阻率_RT': rt_tr,
        '总有机碳_TOC(%)': toc_tr, '孔隙度_POR(%)': por_tr, '渗透率_PERM(mD)': perm_tr
    })

    # 独立盲井：1口勘探盲井（带一定地层非均质扰动）
    np.random.seed(1024)
    depth_blind = np.linspace(2500, 2680, 150)
    gr_b = np.clip(50 + 60 * np.cos(depth_blind / 30) + np.random.normal(0, 8, 150), 20, 185)
    rhob_b = np.clip(2.67 - 0.003 * gr_b + np.random.normal(0, 0.035, 150), 2.15, 2.8)
    dt_b = np.clip(182 + (2.7 - rhob_b) * 115 + np.random.normal(0, 6, 150), 170, 310)
    nphi_b = np.clip(0.045 + 0.0015 * gr_b + np.random.normal(0, 0.02, 150), 0.02, 0.45)
    rt_b = 10 ** (np.random.uniform(0.5, 2.3, 150) + 0.008 * gr_b / 10)

    toc_b = np.clip(0.028 * gr_b + 0.009 * dt_b - 1.15 * rhob_b + np.random.normal(0, 0.28, 150), 0.2, 8.0)
    por_b = np.clip(((2.65 - rhob_b) / 1.65) * 65 + nphi_b * 28 + np.random.normal(0, 0.75, 150), 1.5, 28.0)
    perm_b = np.clip((10 ** (0.17 * por_b - 1.4)) * np.exp(np.random.normal(0, 0.38, 150)), 0.01, 2000.0)

    df_blind = pd.DataFrame({
        '井深(m)': depth_blind, '自然伽马_GR': gr_b, '声波时差_DT': dt_b,
        '补偿密度_RHOB': rhob_b, '补偿中子_NPHI': nphi_b, '深侧向电阻率_RT': rt_b,
        '总有机碳_TOC(%)': toc_b, '孔隙度_POR(%)': por_b, '渗透率_PERM(mD)': perm_b
    })
    return df_train, df_blind

# 4. 数据增强引擎 (仅对训练集作用，单线程防死锁)
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

# -------------------------------------------------------------
# 5. 侧边栏双文件上传与参数绑定
# -------------------------------------------------------------
st.sidebar.title("🛢️ 储层物性盲井跨井验证工作台")

data_mode = st.sidebar.radio("数据模式", ["上传本地训练集与盲井文件", "使用系统示例基准双文件"])

if data_mode == "上传本地训练集与盲井文件":
    st.sidebar.markdown("#### 📂 上传文件")
    train_file = st.sidebar.file_uploader("1. 上传训练集数据 (TXT/CSV)", type=["txt", "csv"], key="train_file")
    blind_file = st.sidebar.file_uploader("2. 上传独立盲井数据 (TXT/CSV)", type=["txt", "csv"], key="blind_file")
    
    df_train_raw = load_file(train_file)
    df_blind_raw = load_file(blind_file)

    if df_train_raw is None or df_blind_raw is None:
        st.sidebar.info("请上传两个完整文件。未上传前已自动载入演示基准数据。")
        df_train_raw, df_blind_raw = generate_demo_datasets()
    else:
        st.sidebar.success(f"训练集: {df_train_raw.shape[0]}行 | 盲井: {df_blind_raw.shape[0]}行")
else:
    df_train_raw, df_blind_raw = generate_demo_datasets()

# 拷贝数据防止缓存篡改
df_tr = df_train_raw.copy()
df_bl = df_blind_raw.copy()

# 共同列分析与纯数值过滤
common_cols = [c for c in df_tr.columns if c in df_bl.columns]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 1. 变量与特征映射")

# 井深列
depth_guess = next((c for c in common_cols if any(k in str(c).lower() for k in ['depth', '深', '井深'])), common_cols[0])
depth_col = st.sidebar.selectbox("井深列 (用于单井剖面绘图)", common_cols, index=common_cols.index(depth_guess))

# 自动筛选数值列（剔除井名等字符串）
numeric_features = []
for c in common_cols:
    if c == depth_col:
        continue
    s_tr = pd.to_numeric(df_tr[c], errors='coerce')
    s_bl = pd.to_numeric(df_bl[c], errors='coerce')
    if s_tr.notna().sum() > len(df_tr) * 0.5 and s_bl.notna().sum() > len(df_bl) * 0.5:
        numeric_features.append(c)
        df_tr[c] = s_tr
        df_bl[c] = s_bl

if not numeric_features:
    st.error("两个文件中未检测到共同的数值型测井特征！")
    st.stop()

# 目标列选择
target_col = st.sidebar.selectbox("预测物性目标列 (Target)", numeric_features, index=len(numeric_features)-1)

# 输入特征列选择
feat_candidates = [c for c in numeric_features if c != target_col]
feature_cols = st.sidebar.multiselect("模型输入测井特征 (Features)", feat_candidates, default=feat_candidates)

if not feature_cols:
    st.warning("请至少选择一个输入特征！")
    st.stop()

# 检查盲井是否含有实测真实标签
blind_has_ground_truth = df_bl[target_col].notna().sum() > (len(df_bl) * 0.3)

# 数据清洗
df_tr_clean = df_tr[feature_cols + [target_col, depth_col]].dropna().copy()
df_bl_clean = df_bl[feature_cols + ([target_col] if blind_has_ground_truth else []) + [depth_col]].dropna().copy()

enable_log_target = st.sidebar.checkbox("启用目标变量 log10 对数变换 (渗透率建议勾选)", value=('perm' in target_col.lower() or '渗透' in target_col))

st.sidebar.markdown("---")
# 算法选择
st.sidebar.subheader("🧠 2. 算法选择")
selected_model_name = st.sidebar.selectbox(
    "拟合回归算法",
    ["随机森林回归 (Random Forest)", "支持向量回归 (SVR)", "BP 神经网络回归 (MLP)", "梯度提升回归树 (Gradient Boosting)"]
)

# 增强与验证集划分
st.sidebar.subheader("⚡ 3. 训练集增强设置")
enable_aug = st.sidebar.checkbox("对训练集开启数据增强", value=True)
aug_method = st.sidebar.selectbox("增强算法", ["KNN流形邻域插值", "连续 Mixup 线性插值", "高斯扰动抖动 (Jittering)"]) if enable_aug else "关闭"
aug_factor = st.sidebar.slider("增强倍数 (生成 N 倍合成样本)", 1, 5, 2) if enable_aug else 0
val_ratio = st.sidebar.slider("训练集内部切分验证集比例 (Val)", 0.1, 0.3, 0.2, step=0.05)

# -------------------------------------------------------------
# 6. 计算矩阵提取与模型训练
# -------------------------------------------------------------
X_tr_raw = df_tr_clean[feature_cols].to_numpy(dtype=np.float64)
y_tr_raw = df_tr_clean[target_col].to_numpy(dtype=np.float64)

X_bl_raw = df_bl_clean[feature_cols].to_numpy(dtype=np.float64)
depth_bl = df_bl_clean[depth_col].to_numpy(dtype=np.float64)

if enable_log_target:
    y_tr_raw = np.log10(np.clip(y_tr_raw, 1e-4, None))

# 仅从训练集中切分验证集
X_train_orig, X_val, y_train_orig, y_val = train_test_split(X_tr_raw, y_tr_raw, test_size=val_ratio, random_state=42)

# 对训练集执行数据增强（盲井和验证集完全不参与增强，保证绝对干净）
if enable_aug:
    X_train, y_train, is_synth = augment_dataset(X_train_orig, y_train_orig, method=aug_method, factor=aug_factor)
else:
    X_train, y_train, is_synth = X_train_orig, y_train_orig, np.zeros(len(X_train_orig), dtype=bool)

# 特征标准化（使用训练集参数，Transform 验证集与盲井）
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_bl_s = scaler.transform(X_bl_raw)

# 拟合模型
if "随机森林" in selected_model_name:
    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_va_pred = model.predict(X_val)
    y_bl_pred = model.predict(X_bl_raw)
elif "支持向量" in selected_model_name:
    model = SVR(C=12.0, epsilon=0.08)
    model.fit(X_train_s, y_train)
    y_tr_pred = model.predict(X_train_s)
    y_va_pred = model.predict(X_val_s)
    y_bl_pred = model.predict(X_bl_s)
elif "BP" in selected_model_name:
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=450, random_state=42, early_stopping=True)
    model.fit(X_train_s, y_train)
    y_tr_pred = model.predict(X_train_s)
    y_va_pred = model.predict(X_val_s)
    y_bl_pred = model.predict(X_bl_s)
else:
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_va_pred = model.predict(X_val)
    y_bl_pred = model.predict(X_bl_raw)

# 还原尺度
if enable_log_target:
    y_tr_true, y_tr_hat = 10**y_train, 10**y_tr_pred
    y_va_true, y_va_hat = 10**y_val, 10**y_va_pred
    y_bl_hat = 10**y_bl_pred
    if blind_has_ground_truth:
        y_bl_true = df_bl_clean[target_col].to_numpy(dtype=np.float64)
else:
    y_tr_true, y_tr_hat = y_train, y_tr_pred
    y_va_true, y_va_hat = y_val, y_va_pred
    y_bl_hat = y_bl_pred
    if blind_has_ground_truth:
        y_bl_true = df_bl_clean[target_col].to_numpy(dtype=np.float64)

# -------------------------------------------------------------
# 7. 主界面数据呈现与指标
# -------------------------------------------------------------
st.title("🛢️ 储层物性机器学习盲井跨井验证工作台")
st.caption(f"当前算法：**{selected_model_name}** ｜ 预测目标：**{target_col}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("训练集原样本数", len(X_train_orig))
c2.metric("训练集增强后规模", len(X_train), delta=f"+{np.sum(is_synth)} 合成样本")
c3.metric("训练集内部验证样本", len(X_val))
c4.metric("独立盲井层段点数", len(df_bl_clean))

# 导出盲井预测成果
with st.expander("💾 导出独立盲井连续预测成果表 (TXT/CSV)", expanded=False):
    df_blind_export = df_bl_clean.copy()
    df_blind_export[f'{target_col}_模型预测值'] = np.round(y_bl_hat, 4)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        txt_out = io.StringIO()
        df_blind_export.to_csv(txt_out, sep='\t', index=False)
        st.download_button("📥 下载盲井解释成果 (Tab分隔 TXT)", txt_out.getvalue().encode('utf-8-sig'), file_name=f"blind_well_{target_col}_result.txt")
    with col_e2:
        csv_out = io.StringIO()
        df_blind_export.to_csv(csv_out, index=False)
        st.download_button("📥 下载盲井解释成果 (CSV 格式)", csv_out.getvalue().encode('utf-8-sig'), file_name=f"blind_well_{target_col}_result.csv")

def calc_metrics(yt, yp):
    return r2_score(yt, yp), np.sqrt(mean_squared_error(yt, yp)), mean_absolute_error(yt, yp)

r2_tr, rmse_tr, mae_tr = calc_metrics(y_tr_true, y_tr_hat)
r2_va, rmse_va, mae_va = calc_metrics(y_va_true, y_va_hat)

st.markdown("### 📈 模型评估看板")
if blind_has_ground_truth:
    r2_bl, rmse_bl, mae_bl = calc_metrics(y_bl_true, y_bl_hat)
    m1, m2, m3 = st.columns(3)
    m1.metric("训练集拟合度 R²", f"{r2_tr:.3f}", delta=f"RMSE: {rmse_tr:.3f}")
    m1.caption(f"样本数: {len(y_train)} | MAE: {mae_tr:.3f}")
    m2.metric("内部验证集 R²", f"{r2_va:.3f}", delta=f"RMSE: {rmse_va:.3f}")
    m2.caption(f"样本数: {len(y_val)} | MAE: {mae_va:.3f}")
    m3.metric("🎯 独立盲井真实跨井 R²", f"{r2_bl:.3f}", delta=f"RMSE: {rmse_bl:.3f}")
    m3.caption(f"盲井实测点: {len(y_bl_true)} | MAE: {mae_bl:.3f}")
else:
    m1, m2 = st.columns(2)
    m1.metric("训练集拟合度 R²", f"{r2_tr:.3f}", delta=f"RMSE: {rmse_tr:.3f}")
    m2.metric("内部验证集 R²", f"{r2_va:.3f}", delta=f"RMSE: {rmse_va:.3f}")
    st.info("提示：上传的盲井数据未检测到实测标签，已进入连续物性全井段预测推演模式。")

# -------------------------------------------------------------
# 8. 成果图件呈现
# -------------------------------------------------------------
st.markdown("---")
tab1, tab2 = st.tabs(["1. 散点交叉拟合与残差检验", "2. 独立盲井测井道综合解释大样图"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 交叉散点图
    axes[0].scatter(y_tr_true, y_tr_hat, color='#1f77b4', alpha=0.4, s=25, label=f'Train (R²={r2_tr:.2f})')
    axes[0].scatter(y_va_true, y_va_hat, color='#ff7f0e', alpha=0.7, s=35, label=f'Val (R²={r2_va:.2f})')
    if blind_has_ground_truth:
        axes[0].scatter(y_bl_true, y_bl_hat, color='#d62728', marker='^', alpha=0.9, s=45, label=f'Blind Well (R²={r2_bl:.2f})')
        min_v = min(np.min(y_tr_true), np.min(y_bl_true), np.min(y_bl_hat))
        max_v = max(np.max(y_tr_true), np.max(y_bl_true), np.max(y_bl_hat))
    else:
        min_v = min(np.min(y_tr_true), np.min(y_tr_hat))
        max_v = max(np.max(y_tr_true), np.max(y_tr_hat))
    
    axes[0].plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='1:1 Line')
    if enable_log_target:
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
    axes[0].set_xlabel(f"Measured: {target_col}")
    axes[0].set_ylabel(f"Predicted: {target_col}")
    axes[0].set_title(f"Crossplot: Measured vs Predicted", fontsize=12)
    axes[0].legend()
    
    # 残差分布
    sns.kdeplot(y_tr_true - y_tr_hat, ax=axes[1], fill=True, color='#1f77b4', label='Train Residuals')
    if blind_has_ground_truth:
        sns.kdeplot(y_bl_true - y_bl_hat, ax=axes[1], fill=True, color='#d62728', label='Blind Well Residuals')
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set_xlabel("Residuals (Measured - Predicted)")
    axes[1].set_title("Residuals Probability Density", fontsize=12)
    axes[1].legend()
    
    st.pyplot(fig)
    plt.close()

with tab2:
    # 盲井综合剖面道
    tracks = min(3, len(feature_cols)) + (2 if blind_has_ground_truth else 1)
    fig, axes = plt.subplots(1, tracks, figsize=(3.5 * tracks, 8), sharey=True)
    
    # 特征曲线道
    for i in range(min(3, len(feature_cols))):
        col_name = feature_cols[i]
        axes[i].plot(df_bl_clean[col_name], depth_bl, color=sns.color_palette("tab10")[i], lw=1.2)
        axes[i].set_xlabel(col_name)
        axes[i].grid(True, linestyle=':')
    
    # 盲井预测物性道
    pred_track = axes[min(3, len(feature_cols))]
    if blind_has_ground_truth:
        pred_track.scatter(y_bl_true, depth_bl, color='black', s=10, alpha=0.6, label='Lab Core')
    pred_track.plot(y_bl_hat, depth_bl, color='red', lw=1.5, label='ML Predicted')
    pred_track.set_xlabel(f"Predicted {target_col}")
    if enable_log_target:
        pred_track.set_xscale('log')
    pred_track.legend(loc='upper right')
    pred_track.grid(True, linestyle=':')
    
    # 误差道（若有实测标签）
    if blind_has_ground_truth:
        err_track = axes[tracks - 1]
        err_val = np.abs(y_bl_true - y_bl_hat)
        err_track.fill_betweenx(depth_bl, 0, err_val, color='orange', alpha=0.5)
        err_track.plot(err_val, depth_bl, color='darkorange', lw=1)
        err_track.set_xlabel("Abs Error")
        err_track.grid(True, linestyle=':')

    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"Depth ({depth_col})", fontsize=12)
    fig.suptitle(f"Blind Well Continuous Reservoir Property Prediction Profile [{target_col}]", fontsize=13, y=0.98)
    
    st.pyplot(fig)
    plt.close()
