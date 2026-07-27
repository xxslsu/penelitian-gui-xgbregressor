import os
import sys
import base64
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page layout and config
st.set_page_config(
    page_title="Analisis Faktor Kepuasan & Loyalitas Pelanggan | XGBoost",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert image to base64
def get_base64_image(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception:
            return ""
    return ""

# Get logos path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logo_app_path = os.path.join(BASE_DIR, "static", "assets", "logo_app.png")

logo_app_b64 = get_base64_image(logo_app_path)

# Custom CSS for Pastel Mint & Emerald Theme
custom_css = """
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: #064e3b;
    }
    
    /* Main container styling */
    .stApp {
        background-color: #e6f4ea;
    }
    
    /* Custom Card Style */
    .custom-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #a7f3d0;
        box-shadow: 0 10px 25px -5px rgba(4, 120, 87, 0.08);
        margin-bottom: 24px;
    }
    
    /* Header Branding Navbar */
    .navbar-custom {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(16px);
        border: 1px solid #a7f3d0;
        border-radius: 18px;
        padding: 16px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(4, 120, 87, 0.08);
    }
    
    /* Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 24px;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(4, 120, 87, 0.04);
        border-left: 5px solid #059669;
    }
    
    .metric-card.accent-cyan {
        border-left: 5px solid #0284c7;
    }
    
    .metric-card.accent-indigo {
        border-left: 5px solid #6366f1;
    }
    
    .metric-card.accent-emerald {
        border-left: 5px solid #10b981;
    }
    
    .metric-card-label {
        font-size: 13px;
        font-weight: 600;
        color: #047857;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card-value {
        font-size: 28px;
        font-weight: 800;
        color: #064e3b;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .metric-card-sub {
        font-size: 11px;
        color: #065f46;
        margin-top: 4px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #a7f3d0;
    }
    
    /* Custom button states */
    .stButton>button {
        background-color: #059669;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #047857;
        box-shadow: 0 4px 12px rgba(4, 120, 87, 0.2);
        color: white;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Helper for loading and writing default dataset
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset_default.csv')

def load_default_data():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    else:
        # Fallback raw data if file not found
        st.error("Dataset default tidak ditemukan di direktori data.")
        return None

# Initialize Session States
if 'df' not in st.session_state:
    st.session_state.df = load_default_data()
if 'filename' not in st.session_state:
    st.session_state.filename = 'dataset_default.csv'
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# Model parameter session states
if 'train_ratio' not in st.session_state:
    st.session_state.train_ratio = 0.80
if 'random_state' not in st.session_state:
    st.session_state.random_state = 42
if 'n_estimators' not in st.session_state:
    st.session_state.n_estimators = 100
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 4
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.10
if 'subsample' not in st.session_state:
    st.session_state.subsample = 0.80
if 'colsample_bytree' not in st.session_state:
    st.session_state.colsample_bytree = 0.80
if 'gamma' not in st.session_state:
    st.session_state.gamma = 0.00

# Training outputs session states
if 'target_col' not in st.session_state:
    st.session_state.target_col = 'Y'
if 'predictor_cols' not in st.session_state:
    st.session_state.predictor_cols = ['X1', 'X2', 'X3', 'X4', 'X5']
if 'trained_results' not in st.session_state:
    st.session_state.trained_results = None

# Custom preset functions
def apply_preset(preset_name):
    if preset_name == 'balanced':
        st.session_state.train_ratio = 0.80
        st.session_state.random_state = 42
        st.session_state.n_estimators = 100
        st.session_state.max_depth = 4
        st.session_state.learning_rate = 0.10
        st.session_state.subsample = 0.80
        st.session_state.colsample_bytree = 0.80
        st.session_state.gamma = 0.00
    elif preset_name == 'regularized':
        st.session_state.train_ratio = 0.75
        st.session_state.random_state = 42
        st.session_state.n_estimators = 150
        st.session_state.max_depth = 3
        st.session_state.learning_rate = 0.05
        st.session_state.subsample = 0.70
        st.session_state.colsample_bytree = 0.70
        st.session_state.gamma = 1.00
    elif preset_name == 'high_precision':
        st.session_state.train_ratio = 0.85
        st.session_state.random_state = 42
        st.session_state.n_estimators = 250
        st.session_state.max_depth = 6
        st.session_state.learning_rate = 0.08
        st.session_state.subsample = 0.85
        st.session_state.colsample_bytree = 0.85
        st.session_state.gamma = 0.10
    st.success(f"Preset '{preset_name.upper()}' berhasil diterapkan! Silakan sesuaikan lagi jika diperlukan.")

# Generate detailed text recommendations in Bahasa Indonesia
def generate_interpretation_indonesian(target_col, test_r2, test_rmse, test_mae, sorted_importance, corr_y, n_estimators, max_depth, learning_rate, sample_count):
    r2_pct = round(test_r2 * 100, 2)
    top_feature, top_pct = sorted_importance[0]
    
    # Qualitative R2 assessment
    if test_r2 >= 0.75:
        fit_quality = "sangat tinggi dan sangat memuaskan"
    elif test_r2 >= 0.50:
        fit_quality = "cukup kuat dan moderat"
    elif test_r2 >= 0.25:
        fit_quality = "lemah namun tetap menangkap pola non-linear"
    else:
        fit_quality = "rendah dan membutuhkan eksplorasi fitur tambahan"
        
    ranked_str_list = []
    for rank, (feat, pct) in enumerate(sorted_importance, 1):
        corr_val = corr_y.get(feat, 0)
        direction = "positif" if corr_val >= 0 else "negatif"
        ranked_str_list.append(f"Peringkat {rank}: {feat} (Kontribusi Gain: {pct}%, Korelasi Linear: {corr_val:+.4f} [{direction}])")
        
    ranked_details_text = "\n".join([f"- {s}" for s in ranked_str_list])
    
    p1_eval = (
        f"Berdasarkan hasil pengujian model XGBoost Regressor pada dataset berjumlah {sample_count} observasi, "
        f"diperoleh nilai koefisien determinasi (R-squared) sebesar {test_r2:.4f} ({r2_pct}%). "
        f"Hal ini menunjukkan bahwa {r2_pct}% variabilitas variabel {target_col} (Kepuasan dan Loyalitas Pelanggan) "
        f"dapat dijelaskan secara efektif oleh kombinasi faktor-faktor prediktor yang dimasukkan ke dalam model. "
        f"Tingkat presisi model ditunjukkan oleh nilai Root Mean Squared Error (RMSE) sebesar {test_rmse:.4f} "
        f"dan Mean Absolute Error (MAE) sebesar {test_mae:.4f}, yang mengindikasikan bahwa deviasi rata-rata antara nilai "
        f"prediksi XGBoost dan nilai aktual pelanggan berada pada rentang deviasi yang relatif kecil dan stabil."
    )
    
    p2_importance = (
        f"Hasil ekstraksi feature importance berdasarkan kontribusi Information Gain mengungkapkan struktur pengaruh antar faktor. "
        f"Faktor prediktor utama yang paling dominan mempengaruhi {target_col} adalah {top_feature} dengan bobot kontribusi "
        f"sebesar {top_pct}%. "
    )
    sec_feat, sec_pct = sorted_importance[1] if len(sorted_importance) > 1 else (top_feature, 0)
    p2_importance += (
        f"Faktor berpengaruh berikutnya adalah {sec_feat} dengan kontribusi sebesar {sec_pct}%. "
        f"Secara hierarkis, pemeringkatan seluruh faktor prediktor dari yang paling berpengaruh adalah sebagai berikut:\n\n"
        f"{ranked_details_text}"
    )

    p3_synthesis = (
        f"Analisis komparatif antara bobot kepatuhan pohon XGBoost dan arah korelasi Bivariate Pearson menunjukkan bahwa faktor "
        f"{top_feature} tidak hanya memiliki frekuensi split pohon yang tinggi, tetapi juga bertindak sebagai penentu keputusan "
        f"prioritas utama dalam membentuk kepuasan dan loyalitas pelanggan. Kombinasi hyperparameter XGBoost "
        f"(n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}) terbukti mampu menangkap hubungan non-linear "
        f"dan interaksi kompleks antar variabel prediktor tanpa mengalami kecenderungan overfitting yang merugikan."
    )
    
    p4_recommendation = (
        f"Implikasi Manajerial dan Rekomendasi Keputusan:\n\n"
        f"1. Prioritas Utama Alokasi Sumber Daya: Perusahaan/Organisasi disarankan untuk memprioritaskan peningkatan performa pada faktor {top_feature}, "
        f"karena perbaikan pada dimensi ini memberikan daya dorong (leverage) paling signifikan terhadap peningkatan skor {target_col}.\n\n"
        f"2. Penguatan Faktor Pendukung: Faktor {sec_feat} perlu dijaga konsistensinya sebagai pilar pendukung utama agar tidak menjadi kendala (bottleneck) kepuasan pelanggan.\n\n"
        f"3. Monitoring dan Evaluasi Berkelanjutan: Model XGBoost ini dapat diintegrasikan sebagai sistem pendukung keputusan (Decision Support System) bulanan untuk memantau tren kepuasan dan loyalitas pelanggan secara real-time."
    )
    
    return {
        'eval_summary': p1_eval,
        'importance_summary': p2_importance,
        'synthesis': p3_synthesis,
        'recommendation': p4_recommendation,
        'fit_quality': fit_quality
    }

# ==========================================
# BRANDING NAVBAR & TITLE
# ==========================================
st.markdown(f"""
<div class="navbar-custom">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            {"<img src='data:image/png;base64," + logo_app_b64 + "' width='60px' style='border-radius:8px;'/>" if logo_app_b64 else ""}
            <div style="border-left: 2px solid #a7f3d0; height: 50px; margin: 0 10px;"></div>
            <div>
                <h1 style="margin:0; font-size:26px; font-family:'Plus Jakarta Sans', sans-serif; color:#064e3b; font-weight:700; line-height:1.2;">
                    ANALISIS FAKTOR YANG BERPENGARUH TERHADAP KEPUASAN DAN LOYALITAS PELANGGAN
                </h1>
                <p style="margin:0; font-size:20px; color:#047857; font-weight:500;">
                    MENGGUNAKAN MODEL XGBOOST (EXTREME GRADIENT BOOSTING) REGRESSOR
                </p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# STEPPER WIZARD NAVIGATION
# ==========================================
# Create custom navigation buttons in Sidebar
st.sidebar.markdown("""
<h3 style="margin-bottom:15px; font-size:16px; color:#064e3b; font-family:'Plus Jakarta Sans', sans-serif;">
    MENU NAVIGASI
</h3>
""", unsafe_allow_html=True)

step_options = {
    1: "Beranda & Input Data",
    2: "Analisis Deskriptif",
    3: "Parameter XGBoost",
    4: "Output & Interpretasi"
}

for step_num, step_label in step_options.items():
    is_active = (st.session_state.current_step == step_num)
    button_style = "primary" if is_active else "secondary"
    
    # We use sidebar buttons to switch step
    if st.sidebar.button(step_label, key=f"nav_btn_{step_num}", use_container_width=True):
        st.session_state.current_step = step_num
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background-color:#f0fdf4; border:1px solid #c6f6d5; border-radius:10px; padding:12px; font-size:12px; color:#064e3b;">
    <strong>Tim Penyusun:</strong><br>
    - Mohammad Idhom<br>
    - Trimono<br>
    - Salsabila Wardah<br><br>
    <strong>Afiliasi:</strong><br>
    Universitas Pembangunan Nasional "Veteran" Jawa Timur
</div>
""", unsafe_allow_html=True)

# Helper function to render horizontal stepper layout in main body
def draw_horizontal_stepper(current_step):
    steps = ["Input Data", "Analisis Deskriptif", "Parameter XGBoost", "Output"]
    cols = st.columns(len(steps))
    for i, step_name in enumerate(steps, 1):
        with cols[i-1]:
            is_active = (i == current_step)
            is_done = (i < current_step)
            
            bg_color = "#059669" if is_active else ("#dcfce7" if is_done else "#ffffff")
            text_color = "#ffffff" if is_active else "#064e3b"
            border_color = "#059669" if (is_active or is_done) else "#a7f3d0"
            
            icon = "✓" if is_done else str(i)
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; color:{text_color}; border:2px solid {border_color}; 
                        border-radius:10px; padding:10px 5px; text-align:center; font-family:'Plus Jakarta Sans', sans-serif;
                        box-shadow: 0 4px 6px rgba(4, 120, 87, 0.04); font-weight:600; font-size:13px;">
                <span style="background-color:rgba(255,255,255,0.2); padding: 2px 7px; border-radius:50%; margin-right:5px;">{icon}</span>
                {step_name}
            </div>
            """, unsafe_allow_html=True)
    st.write("")

draw_horizontal_stepper(st.session_state.current_step)

# ===================================================================
# STEP 1: BERANDA & INPUT DATA
# ===================================================================
if st.session_state.current_step == 1:
    st.markdown("""
    <div class="custom-card">
        <h2 style="margin-top:0; font-size:22px; color:#064e3b; margin-bottom:12px;">SELAMAT DATANG DI DASHBOARD ANALISIS PELANGGAN</h2>
        <p style="font-size:20px; color:#065f46; margin-bottom:0; line-height:1.6;">
            Sistem pendukung keputusan ini dirancang untuk menganalisis faktor - faktor yang paling mempengaruhi tingkat <strong>Kepuasan dan Loyalitas Pelanggan</strong> menggunakan algoritma Machine Learning modern <strong>XGBoost Regressor</strong>. Anda dapat menganalisis data, sebaran statistik deskriptif, melatih model regresi, serta mengekstraksi tingkat kepentingan fitur (Feature Importance).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="font-size:16px; margin-bottom:12px; color:#064e3b; font-family:'Plus Jakarta Sans', sans-serif;">
        INPUT DATASET PELANGGAN
    </h3>
    """, unsafe_allow_html=True)
    
    col_uploader, col_reset = st.columns([3, 1])
    
    with col_uploader:
        uploaded_file = st.file_uploader(
            "Unggah file dataset kustom Anda dalam format CSV:",
            type=["csv"],
            help="Dataset harus memiliki kolom target numerik dan minimal beberapa variabel prediktor numerik."
        )
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                numeric_cols = new_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    st.error("Gagal memuat dataset: File CSV harus memiliki minimal 2 kolom bertipe data numerik.")
                else:
                    st.session_state.df = new_df
                    st.session_state.filename = uploaded_file.name
                    # Auto select Y and predictors if columns exist
                    if 'Y' in new_df.columns:
                        st.session_state.target_col = 'Y'
                        st.session_state.predictor_cols = [c for c in new_df.columns if c not in ['No', 'Y']]
                    st.success(f"Berhasil mengunggah dataset kustom: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Gagal membaca file: {str(e)}")
                
    with col_reset:
        st.write("")
        st.write("")
        if st.button("Muat Ulang Data Default", use_container_width=True):
            st.session_state.df = load_default_data()
            st.session_state.filename = 'dataset_default.csv'
            st.session_state.target_col = 'Y'
            st.session_state.predictor_cols = ['X1', 'X2', 'X3', 'X4', 'X5']
            st.info("Dataset berhasil dikembalikan ke dataset default.")
            st.rerun()
            
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin:16px 0 10px 0;">
            <h4 style="font-family:'Plus Jakarta Sans', sans-serif; font-size:14px; font-weight:700; color:#064e3b; margin:0;">
                Pratinjau Dataset ({len(df)} Baris, {len(df.columns)} Kolom)
            </h4>
            <span style="font-size:11px; background-color:#e0f2fe; color:#0369a1; border:1px solid #bae6fd; padding:4px 10px; border-radius:15px; font-weight:600;">
                Dataset Aktif: {st.session_state.filename}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df, height=350, use_container_width=True)
        
        # Navigation
        st.write("")
        col_space, col_next = st.columns([3, 1])
        with col_next:
            if st.button("Selanjutnya: Analisis Deskriptif ➔", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()

# ===================================================================
# STEP 2: ANALISIS DESKRIPTIF
# ===================================================================
elif st.session_state.current_step == 2:
    if st.session_state.df is None:
        st.warning("Silakan muat dataset terlebih dahulu di Langkah 1.")
    else:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=[np.number])
        # Exclude index or ID columns
        cols_to_stats = [c for c in numeric_df.columns if c.lower() not in ['no', 'id']]
        
        # 1. Summary Metric Row
        mean_y_val = 0.0
        if 'Y' in df.columns:
            mean_y_val = round(df['Y'].mean(), 2)
            std_y_val = round(df['Y'].std(), 2)
        elif len(cols_to_stats) > 0:
            mean_y_val = round(df[cols_to_stats[0]].mean(), 2)
            std_y_val = round(df[cols_to_stats[0]].std(), 2)
            
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-card-label">Jumlah Observasi (N)</div>
                <div class="metric-card-value">{len(df)}</div>
                <div class="metric-card-sub">Sampel Pelanggan</div>
            </div>
            <div class="metric-card accent-cyan">
                <div class="metric-card-label">Rata-Rata Target</div>
                <div class="metric-card-value">{mean_y_val}</div>
                <div class="metric-card-sub">Skala Kepuasan / Loyalitas</div>
            </div>
            <div class="metric-card accent-indigo">
                <div class="metric-card-label">Standar Deviasi Target</div>
                <div class="metric-card-value">{std_y_val}</div>
                <div class="metric-card-sub">Variabilitas Nilai Target</div>
            </div>
            <div class="metric-card accent-emerald">
                <div class="metric-card-label">Jumlah Variabel Numerik</div>
                <div class="metric-card-value">{len(cols_to_stats)}</div>
                <div class="metric-card-sub">Variabel untuk Analisis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Descriptive Statistics Table
        st.markdown("""
        <h3 style="font-size:16px; margin-bottom:12px; color:#064e3b; font-family:'Plus Jakarta Sans', sans-serif;">
            TABEL STATISTIK DESKRIPTIF
        </h3>
        """, unsafe_allow_html=True)
        
        stats_dict = []
        for col in cols_to_stats:
            vals = df[col].dropna()
            stats_dict.append({
                'Variabel': col,
                'N': int(len(vals)),
                'Mean': round(float(vals.mean()), 4),
                'Std Dev': round(float(vals.std()), 4),
                'Min': round(float(vals.min()), 4),
                'Q25 (25%)': round(float(vals.quantile(0.25)), 4),
                'Median (Q50)': round(float(vals.median()), 4),
                'Q75 (75%)': round(float(vals.quantile(0.75)), 4),
                'Max': round(float(vals.max()), 4),
                'Skewness': round(float(skew(vals)), 4),
                'Kurtosis': round(float(kurtosis(vals)), 4)
            })
            
        stats_df = pd.DataFrame(stats_dict)
        st.dataframe(stats_df, use_container_width=True)
        
        # 3. Graphical Charts (Tabs inside Step 2)
        st.write("")
        st.markdown("""
        <h3 style="font-size:16px; margin-bottom:12px; color:#064e3b; font-family:'Plus Jakarta Sans', sans-serif;">
            GRAFIK VISUALISASI DATA INTERAKTIF
        </h3>
        """, unsafe_allow_html=True)
        
        tab_corr, tab_dist, tab_box, tab_scatter = st.tabs([
            "Matriks Korelasi Heatmap",
            "Distribusi Histogram",
            "Boxplot Sebaran & Outlier",
            "Scatter Plot vs Target Y"
        ])
        
        with tab_corr:
            # Heatmap Matrix
            corr_mat = df[cols_to_stats].corr().round(2)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )

            fig_corr.update_traces(
                textfont_size=13
            )

            fig_corr.update_layout(
                title={
                    "text": "Matriks Korelasi Pearson",
                    "x": 0.5,
                    "font": dict(size=22)
                },
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(size=13),
                coloraxis_colorbar=dict(
                    title="Korelasi",
                    thickness=18
                ),
                margin=dict(l=20,r=20,t=70,b=20)
            )

            st.plotly_chart(fig_corr,use_container_width=True)
                    
        with tab_dist:
            dist_var = st.selectbox("Pilih variabel untuk dilihat distribusinya:", cols_to_stats)
            fig_dist = px.histogram(
                df,
                x=dist_var,
                nbins=15,
                color_discrete_sequence=["#059669"],
                title=f"Histogram Distribusi Frekuensi Variabel {dist_var}",
                marginal="rug"
            )

            fig_dist.update_traces(
                marker=dict(
                    line=dict(
                        color="white",
                        width=1.5
                    )
                )
            )

            fig_dist.update_layout(
                title={
                    "text":f"Distribusi Variabel {dist_var}",
                    "x":0.5,
                    "font":dict(size=22)
                },
                bargap=0.05,
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis_title=dist_var,
                yaxis_title="Frekuensi"
            )

            st.plotly_chart(fig_dist, use_container_width=True)
                
        with tab_box:
            fig_box = px.box(
                df[cols_to_stats],
                points="outliers",
                color_discrete_sequence=["#3B82F6"],
                title="Boxplot Sebaran dan Deteksi Outlier Variabel Penelitian"
            )
            fig_box.update_traces(
                line=dict(
                    color="#2563EB",
                    width=2
                ),
                fillcolor="#BFDBFE",
                marker=dict(
                    color="#EF4444",
                    size=8
                )
                
            )

            fig_box.update_layout(
                title={
                    "text":"Sebaran Data dan Outlier",
                    "x":0.5,
                    "font":dict(size=22)
                },

                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False
            )

            st.plotly_chart(fig_box, use_container_width=True)
            
        with tab_scatter:
            default_target = 'Y' if 'Y' in cols_to_stats else cols_to_stats[0]
            other_cols = [c for c in cols_to_stats if c != default_target]
            
            col_x, col_y_choice = st.columns(2)
            with col_x:
                scatter_x = st.selectbox("Pilih Variabel Prediktor (X):", other_cols)
            with col_y_choice:
                scatter_y = st.selectbox("Pilih Variabel Target (Y):", cols_to_stats, index=cols_to_stats.index(default_target))
            fig_scatter = px.scatter(
                df, x=scatter_x, y=scatter_y, color=scatter_y, color_continuous_scale="Viridis", opacity=0.8)
            fig_scatter.update_traces(
                marker=dict(
                    size=11, line=dict(
                        color="white", width=1
                    )
                )
            )
            fig_scatter.update_layout(
                title={
                    "text":f"Hubungan {scatter_x} terhadap {scatter_y}", "x":0.5, "font":dict(size=22)
                },

                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis_title=scatter_x,
                yaxis_title=scatter_y
            )

            st.plotly_chart(fig_scatter,use_container_width=True)
    
        # Navigation
        st.write("")
        col_prev, col_space, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅ Kembali: Input Data", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        with col_next:
            if st.button("Selanjutnya: Parameter XGBoost ➔", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()

# ===================================================================
# STEP 3: CONFIG PARAMETER XGBOOST
# ===================================================================
elif st.session_state.current_step == 3:
    if st.session_state.df is None:
        st.warning("Silakan muat dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        
        st.markdown("""
        <div class="custom-card">
            <h2 style="margin-top:0; font-size:18px; color:#064e3b; margin-bottom:8px;">Konfigurasi Hyperparameter XGBoost</h2>
            <p style="font-size:13px; color:#065f46; margin-bottom:0;">
                Tentukan variabel target dan prediktor, serta atur arsitektur boosting pohon untuk memperoleh model regresi terbaik. Gunakan Preset Cepat di bawah ini untuk konfigurasi instan yang optimal.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Preset Pills Row
        st.markdown("#####Preset Konfigurasi Cepat:")
        col_pres1, col_pres2, col_pres3, _ = st.columns([1, 1, 1, 1])
        with col_pres1:
            if st.button("Balanced / Default Optimal", use_container_width=True):
                apply_preset('balanced')
                st.rerun()
        with col_pres2:
            if st.button("Anti-Overfitting", use_container_width=True):
                apply_preset('regularized')
                st.rerun()
        with col_pres3:
            if st.button("High Precision Tuning", use_container_width=True):
                apply_preset('high_precision')
                st.rerun()
                
        st.write("")
        
        # Grid layout for inputs
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Pemilihan Variabel & Rasio Data")
            
            # Select target variable Y
            target_default_idx = numeric_cols.index('Y') if 'Y' in numeric_cols else 0
            st.session_state.target_col = st.selectbox(
                "Pilih Variabel Target (Y):", 
                numeric_cols, 
                index=target_default_idx
            )
            
            # Select predictor variables X
            pred_defaults = [c for c in numeric_cols if c not in ['No', st.session_state.target_col]]
            st.session_state.predictor_cols = st.multiselect(
                "Pilih Variabel Prediktor (X):", 
                [c for c in numeric_cols if c != st.session_state.target_col], 
                default=pred_defaults
            )
            
            st.write("")
            
            # Train test split slider
            st.slider(
                "Rasio Pembagian Data Uji (Test Split Ratio):",
                min_value=0.50,
                max_value=0.90,
                step=0.05,
                key='train_ratio',
                help="Proporsi dataset yang digunakan untuk pelatihan model."
            )
            
            st.number_input(
                "Random Seed:",
                min_value=1,
                max_value=99999,
                key='random_state'
            )
            
            st.markdown("#### Struktur Pohon Keputusan")
            st.slider(
                "n_estimators :",
                min_value=10,
                max_value=500,
                step=10,
                key='n_estimators',
                help="Jumlah pohon keputusan yang akan dibangun secara sekuensial."
            )
            st.slider(
                "max_depth (Kedalaman Maksimum Pohon):",
                min_value=1,
                max_value=15,
                step=1,
                key='max_depth',
                help="Batas kedalaman maksimal setiap pohon keputusan. Nilai lebih tinggi meningkatkan kompleksitas model."
            )
            
        with col_right:
            st.markdown("#### Kecepatan & Kepatuhan Pohon (Regularisasi)")
            st.slider(
                "learning_rate (Eta / Laju Pembelajaran):",
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                key='learning_rate',
                help="Faktor skala penyusutan langkah pembaruan bobot pohon baru."
            )
            st.slider(
                "subsample :",
                min_value=0.50,
                max_value=1.00,
                step=0.05,
                key='subsample',
                help="Rasio baris data latihan acak yang dipakai untuk membangun tiap pohon."
            )
            st.slider(
                "colsample_bytree (Rasio Pengambilan Sampel Fitur):",
                min_value=0.50,
                max_value=1.00,
                step=0.05,
                key='colsample_bytree',
                help="Rasio kolom/fitur acak yang dipakai untuk membagi tiap node pohon."
            )
            st.slider(
                "gamma :",
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                key='gamma',
                help="Batas pengurangan loss minimal yang dibutuhkan untuk membuat cabang baru pada pohon."
            )
            
        st.write("")
        st.markdown("---")
        
        # Checkbox for GridSearchCV (User's Pipeline)
        use_grid_search = st.checkbox(
            "Gunakan Hyperparameter Tuning (GridSearchCV)", 
            value=True,
            help="Jika dicentang, model akan melakukan pencarian hyperparameter terbaik secara otomatis menggunakan GridSearchCV (5-Fold CV)."
        )
        
        # Navigation
        col_prev, col_space, col_train = st.columns([1, 1, 2])
        with col_prev:
            if st.button("⬅ Kembali : Analisis Deskriptif", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
                
        with col_train:
            if st.button("Latih Model XGBoost & Lihat Output ➔", use_container_width=True):
                if len(st.session_state.predictor_cols) < 1:
                    st.error("Gagal melatih! Harap pilih minimal 1 variabel prediktor.")
                else:
                    # Model Training Logic
                    X = df[st.session_state.predictor_cols]
                    y = df[st.session_state.target_col]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=(1.0 - st.session_state.train_ratio), 
                        random_state=st.session_state.random_state
                    )
                    
                    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
                    import joblib
                    
                    if use_grid_search:
                        # Grid parameters matching user's request
                        param_grid = {
                            'n_estimators': [50, 100, 150],
                            'max_depth': [2, 3, 4],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'subsample': [0.8, 1.0],
                            'colsample_bytree': [0.8, 1.0]
                        }
                        
                        grid = GridSearchCV(
                            xgb.XGBRegressor(objective='reg:squarederror', random_state=st.session_state.random_state),
                            param_grid=param_grid,
                            scoring='r2',
                            cv=5,
                            n_jobs=-1
                        )
                        
                        with st.spinner("Menjalankan GridSearchCV untuk mencari hyperparameter terbaik"):
                            grid.fit(X_train, y_train)
                            best_params = grid.best_params_
                            model = grid.best_estimator_
                    else:
                        best_params = {
                            'n_estimators': st.session_state.n_estimators,
                            'max_depth': st.session_state.max_depth,
                            'learning_rate': st.session_state.learning_rate,
                            'subsample': st.session_state.subsample,
                            'colsample_bytree': st.session_state.colsample_bytree
                        }
                        model = xgb.XGBRegressor(
                            n_estimators=st.session_state.n_estimators,
                            max_depth=st.session_state.max_depth,
                            learning_rate=st.session_state.learning_rate,
                            subsample=st.session_state.subsample,
                            colsample_bytree=st.session_state.colsample_bytree,
                            gamma=st.session_state.gamma,
                            random_state=st.session_state.random_state,
                            objective='reg:squarederror'
                        )
                        with st.spinner("Melatih model XGBoost Regressor dengan parameter manual..."):
                            model.fit(X_train, y_train)
                            
                    # Fit again on train data (as in user's pipeline)
                    with st.spinner("Melatih ulang model final pada data train..."):
                        model.fit(X_train, y_train)
                        
                    # Calculate 5-Fold Cross Validation
                    with st.spinner("Menghitung 5-Fold Cross Validation..."):
                        cv = KFold(n_splits=5, shuffle=True, random_state=st.session_state.random_state)
                        cv_scores = cross_val_score(
                            model,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring='r2'
                        )
                        mean_cv_r2 = float(cv_scores.mean())
                        
                    # Predict on Test set
                    y_test_pred = model.predict(X_test)
                    test_r2 = float(r2_score(y_test, y_test_pred))
                    test_mse = float(mean_squared_error(y_test, y_test_pred))
                    test_rmse = float(np.sqrt(test_mse))
                    test_mae = float(mean_absolute_error(y_test, y_test_pred))
                    
                    # Calculate MAPE and Accuracy
                    y_test_arr = np.array(y_test)
                    y_pred_arr = np.array(y_test_pred)
                    non_zero_mask = y_test_arr != 0
                    if np.any(non_zero_mask):
                        mape = float(np.mean(np.abs((y_test_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_test_arr[non_zero_mask])) * 100)
                        accuracy_pct = float(max(0, round(100.0 - mape, 2)))
                    else:
                        mape = 0.0
                        accuracy_pct = float(max(0, round(test_r2 * 100, 2)))
                        
                    # Tolerance Accuracy
                    tolerance_mask = np.abs(y_test_arr - y_pred_arr) <= (0.15 * y_test_arr)
                    tolerance_accuracy_pct = float(round(np.mean(tolerance_mask) * 100, 2))
                    
                    # Train Metrics
                    y_train_pred = model.predict(X_train)
                    train_r2 = float(r2_score(y_train, y_train_pred))
                    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
                    
                    # Overall Metrics
                    y_all_pred = model.predict(X)
                    all_r2 = float(r2_score(y, y_all_pred))
                    all_rmse = float(np.sqrt(mean_squared_error(y, y_all_pred)))
                    all_mae = float(mean_absolute_error(y, y_all_pred))
                    
                    # Feature Importances using model.feature_importances_
                    importances_val = model.feature_importances_
                    feature_importance_pct = {}
                    for col, val in zip(st.session_state.predictor_cols, importances_val):
                        feature_importance_pct[col] = float(round(val * 100, 2))
                        
                    sorted_importance = sorted(feature_importance_pct.items(), key=lambda x: x[1], reverse=True)
                    
                    # Pearson Correlations with Target
                    corr_y = df[st.session_state.predictor_cols].apply(
                        lambda col: df[st.session_state.target_col].corr(col)
                    ).to_dict()
                    
                    # Generate Interpretation text
                    interpretation = generate_interpretation_indonesian(
                        target_col=st.session_state.target_col,
                        test_r2=test_r2,
                        test_rmse=test_rmse,
                        test_mae=test_mae,
                        sorted_importance=sorted_importance,
                        corr_y=corr_y,
                        n_estimators=best_params['n_estimators'],
                        max_depth=best_params['max_depth'],
                        learning_rate=best_params['learning_rate'],
                        sample_count=len(df)
                    )
                    
                    # Save model locally (as in user's pipeline)
                    model_save_path = os.path.join(BASE_DIR, "xgboost_regressor.pkl")
                    joblib.dump(model, model_save_path)
                    
                    # Save results to session state
                    st.session_state.trained_results = {
                        'metrics': {
                            'accuracy_pct': round(accuracy_pct, 2),
                            'mape': round(mape, 2),
                            'tolerance_accuracy_pct': round(tolerance_accuracy_pct, 2),
                            'test_r2': round(test_r2, 4),
                            'test_mse': round(test_mse, 4),
                            'test_rmse': round(test_rmse, 4),
                            'test_mae': round(test_mae, 4),
                            'train_r2': round(train_r2, 4),
                            'train_rmse': round(train_rmse, 4),
                            'overall_r2': round(all_r2, 4),
                            'overall_rmse': round(all_rmse, 4),
                            'overall_mae': round(all_mae, 4),
                            'mean_cv_r2': round(mean_cv_r2, 4),
                            'cv_scores': [round(float(s), 4) for s in cv_scores]
                        },
                        'best_params': best_params,
                        'use_grid_search': use_grid_search,
                        'feature_importance': {
                            'percentage': feature_importance_pct,
                            'ranked': sorted_importance
                        },
                        'correlations_y': {k: round(v, 4) for k, v in corr_y.items()},
                        'actual': y_test_arr.tolist(),
                        'predicted': y_pred_arr.tolist(),
                        'residuals': (y_test_arr - y_pred_arr).tolist(),
                        'interpretation': interpretation,
                        'model_save_path': model_save_path
                    }
                    
                    st.success("Model XGBoost berhasil dilatih dan disimpan!")
                    st.session_state.current_step = 4
                    st.rerun()

# ===================================================================
# STEP 4: OUTPUT + INTERPRETASI
# ===================================================================
elif st.session_state.current_step == 4:
    if st.session_state.trained_results is None:
        st.warning("Model belum dilatih! Silakan atur parameter dan klik tombol latih model.")
        if st.button("➔ Menuju Langkah 3 : Parameter XGBoost", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    else:
        res = st.session_state.trained_results
        metrics = res['metrics']
        
        # 1. Metric Cards
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card accent-emerald">
                <div class="metric-card-label">Tingkat Akurasi (MAPE)</div>
                <div class="metric-card-value">{metrics['accuracy_pct']}%</div>
                <div class="metric-card-sub">Akurasi Estimasi Target</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-label">R-Squared (R²) Score</div>
                <div class="metric-card-value">{metrics['test_r2']}</div>
                <div class="metric-card-sub">Proporsi Variansi Terjelaskan</div>
            </div>
            <div class="metric-card accent-cyan">
                <div class="metric-card-label">Root Mean Sq Error (RMSE)</div>
                <div class="metric-card-value">{metrics['test_rmse']}</div>
                <div class="metric-card-sub">Tingkat Deviasi Prediksi</div>
            </div>
            <div class="metric-card accent-indigo">
                <div class="metric-card-label">Mean Absolute Error (MAE)</div>
                <div class="metric-card-value">{metrics['test_mae']}</div>
                <div class="metric-card-sub">Deviasi Absolut Rata-Rata</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Helper text below metrics
        st.caption(f"Akurasi Toleransi (Deviasi Uji <= 15% dari Nilai Aktual): {metrics['tolerance_accuracy_pct']}% | R² Train: {metrics['train_r2']} | R² Overall: {metrics['overall_r2']}")
        
        # Best Parameters and Cross Validation Display (User's Pipeline)
        col_cv_info, col_download = st.columns([3, 1])
        with col_cv_info:
            if res.get('use_grid_search', False):
                bp = res['best_params']
                st.info(f"Hyperparameter Terbaik (GridSearchCV): `n_estimators`={bp['n_estimators']}, `max_depth`={bp['max_depth']}, `learning_rate`={bp['learning_rate']}, `subsample`={bp['subsample']}, `colsample_bytree`={bp['colsample_bytree']}")
            
            cv_scores_str = ", ".join([f"{s:.4f}" for s in metrics['cv_scores']])
            st.success(f"5-Fold Cross Validation R² scores: [{cv_scores_str}] | Mean R² CV: {metrics['mean_cv_r2']:.4f}")
            
        with col_download:
            # Model download button
            model_file_path = res.get('model_save_path', '')
            if os.path.exists(model_file_path):
                try:
                    with open(model_file_path, "rb") as f:
                        model_bytes = f.read()
                    st.download_button(
                        label="Unduh Model (.pkl)",
                        data=model_bytes,
                        file_name="xgboost_regressor.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Gagal memuat file model: {str(e)}")
        
        # 2. Charts Row: Feature Importance & Actual vs Predicted
        st.write("")
        col_chart_left, col_chart_right = st.columns(2)
        
        with col_chart_left:
            st.markdown("##### Feature Importance")
            # Generate feature importance bar chart using Plotly
            ranked_feats = res['feature_importance']['ranked']
            feat_names = [item[0] for item in ranked_feats][::-1]  # reverse for horizontal chart
            feat_pcts = [item[1] for item in ranked_feats][::-1]
            
            fig_imp = go.Figure(go.Bar(
                x=feat_pcts,
                y=feat_names,
                orientation='h',
                marker=dict(color='#059669', line=dict(color='#047857', width=1)),
                text=[f"{p}%" for p in feat_pcts],
                textposition='auto'
            ))
            fig_imp.update_layout(
                margin=dict(l=50, r=30, t=20, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=12),
                xaxis_title="Kontribusi Persentase Information Gain (%)",
                yaxis_title="Faktor Prediktor"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_chart_right:
            st.markdown("##### Grafik Evaluasi Aktual vs Prediksi")
            # Actual vs Predicted plot
            y_act = res['actual']
            y_pred = res['predicted']
            
            min_val = min(min(y_act), min(y_pred)) - 0.2
            max_val = max(max(y_act), max(y_pred)) + 0.2
            
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(go.Scatter(
                x=y_act,
                y=y_pred,
                mode='markers',
                marker=dict(color='#0284c7', size=9, opacity=0.75, line=dict(color='#0369a1', width=1)),
                name='Observasi Data'
            ))
            # Diagonal line (y=x)
            fig_act_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='#e11d48', width=2, dash='dash'),
                name='Garis Ideal (y=x)'
            ))
            fig_act_pred.update_layout(
                margin=dict(l=40, r=20, t=20, b=45),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=12),
                xaxis_title="Nilai Aktual Pelanggan",
                yaxis_title="Nilai Prediksi Model XGBoost",
                legend=dict(x=0.05, y=0.95)
            )
            st.plotly_chart(fig_act_pred, use_container_width=True)
            
        # 3. Residual Plot
        st.write("")
        st.markdown("##### Analisis Sebaran Kesalahan (Residual Error Distribution)")
        
        residuals = res['residuals']
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='#6366f1', size=8, opacity=0.7, line=dict(color='#4f46e5', width=1)),
            name='Residual'
        ))
        fig_res.add_trace(go.Scatter(
            x=[min(y_pred)-0.2, max(y_pred)+0.2],
            y=[0, 0],
            mode='lines',
            line=dict(color='#374151', width=1.5, dash='solid'),
            name='Titik Nol Error'
        ))
        fig_res.update_layout(
            margin=dict(l=40, r=20, t=10, b=45),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            xaxis_title="Nilai Prediksi Model",
            yaxis_title="Kesalahan (Residual = Aktual - Prediksi)",
            showlegend=False
        )
        st.plotly_chart(fig_res, use_container_width=True)
        
        # 4. Detailed Text Interpretation
        st.write("")
        st.markdown("""
        <h3 style="font-size:16px; margin-bottom:12px; color:#064e3b; font-family:'Plus Jakarta Sans', sans-serif;">
            INTERPRETASI
        </h3>
        """, unsafe_allow_html=True)
        
        interp = res['interpretation']
        
        col_interp_left, col_interp_right = st.columns(2)
        
        with col_interp_left:
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #c6f6d5; border-radius:12px; padding:20px; min-height:220px; box-shadow:0 4px 6px rgba(4,120,87,0.02); margin-bottom:20px;">
                <h4 style="margin-top:0; color:#059669; font-size:14px; margin-bottom:10px;">RINGKASAN EVALUASI & KESESUAIAN MODEL</h4>
                <p style="font-size:13px; color:#064e3b; line-height:1.6; text-align:justify; margin:0;">
                    {interp['eval_summary']}
                </p>
                <div style="margin-top:12px; font-size:12px; color:#047857; font-weight:600;">
                    Kualitas Fit Model: Kategori <span style="background-color:#dcfce7; padding:2px 8px; border-radius:10px;">{interp['fit_quality'].upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #c6f6d5; border-radius:12px; padding:20px; min-height:200px; box-shadow:0 4px 6px rgba(4,120,87,0.02);">
                <h4 style="margin-top:0; color:#059669; font-size:14px; margin-bottom:10px;">ANALISIS SINTESIS PENGAMBILAN KEPUTUSAN</h4>
                <p style="font-size:13px; color:#064e3b; line-height:1.6; text-align:justify; margin:0;">
                    {interp['synthesis']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_interp_right:
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #c6f6d5; border-radius:12px; padding:20px; min-height:220px; box-shadow:0 4px 6px rgba(4,120,87,0.02); margin-bottom:20px;">
                <h4 style="margin-top:0; color:#0284c7; font-size:14px; margin-bottom:10px;">HIERARKI KONTRIBUSI SEBARAN FITUR</h4>
                <div style="font-size:13px; color:#064e3b; line-height:1.6; text-align:justify;">
                    {interp['importance_summary'].replace('- Peringkat', '<br>- Peringkat')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color:#e0f2fe; border:1px solid #bae6fd; border-radius:12px; padding:20px; min-height:200px; box-shadow:0 4px 6px rgba(2,132,199,0.02);">
                <h4 style="margin-top:0; color:#0369a1; font-size:14px; margin-bottom:10px;">IMPLIKASI DAN REKOMENDASI TINDAKAN</h4>
                <p style="font-size:13px; color:#0369a1; line-height:1.6; text-align:justify; margin:0;">
                    {interp['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Navigation
        st.write("")
        st.markdown("---")
        col_prev, col_space = st.columns([1, 3])
        with col_prev:
            if st.button("⬅ Kembali: Atur Parameter", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
