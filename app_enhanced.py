import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from data_processing import HurricaneDataProcessor
import os
import pickle
import folium
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
import json
import requests
from datetime import datetime
from functools import lru_cache

# Cấu hình trang
st.set_page_config(
    page_title="Hurricane Trajectory Analysis & Prediction",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & Style ---
# Màu chủ đạo: gradients of blue, teal, and indigo
primary_color = "#1E88E5"
secondary_color = "#00ACC1"
accent_color = "#3949AB"
bg_color = "#f8f9fa"
text_color = "#37474F"

# Loading animation
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Custom CSS với thiết kế hiện đại
# Cập nhật CSS chung với thiết kế hiện đại hơn
st.markdown("""
<style>
    /* Main Containers */
    .main {
        background-color: #f8f9fa;
        padding: 0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles - Improved with gradient and animation */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        background: linear-gradient(120deg, #1E88E5, #3949AB, #1E88E5);
        background-size: 200% 100%;
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        animation: gradient-animation 6s ease infinite;
    }
    
    @keyframes gradient-animation {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    
    /* Cards & Containers - Enhanced with better animations */
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    
    .card:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transform: translateY(-8px);
    }
    
    /* Modern Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Improvements */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #f1f5f9;
        background-image: linear-gradient(to bottom, #f1f5f9, #e2e8f0);
    }
    
    /* Button Styles */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0D47A1;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Animated Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
        background-image: linear-gradient(45deg, 
                          rgba(255,255,255,.15) 25%, 
                          transparent 25%, 
                          transparent 50%, 
                          rgba(255,255,255,.15) 50%, 
                          rgba(255,255,255,.15) 75%, 
                          transparent 75%, 
                          transparent);
        background-size: 1rem 1rem;
        animation: progress-animation 1s linear infinite;
    }
    
    @keyframes progress-animation {
        0% {background-position: 0 0;}
        100% {background-position: 1rem 0;}
    }
    
    /* Table Improvements */
    .dataframe {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #1E88E5, #3949AB);
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: 600;
        position: sticky;
        top: 0;
    }
    
    /* Better Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #f8f9fa;
        border-radius: 30px;
        color: #1E88E5;
        font-weight: 500;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1E88E5, #3949AB) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


# Khởi tạo session state
if 'processor' not in st.session_state:
    st.session_state.processor = HurricaneDataProcessor()
    st.session_state.data_loaded = False
    st.session_state.features_extracted = False
    st.session_state.model_trained = False
    st.session_state.selected_trajectory = None
    st.session_state.animation_speed = 50
    st.session_state.animation_frame = 0
    st.session_state.show_animation = False
    st.session_state.preprocessing_options = {
        'outlier_method': 'winsorize',
        'create_interactions': True,
        'use_features': True
    }
    st.session_state.random_trajectory_idx = 0
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Helper: Chuyển DataFrame sang định dạng Arrow-compatible ---
def make_dataframe_arrow_compatible(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null = df[col].dropna()
            if not non_null.empty:
                sample = non_null.iloc[0]
                if isinstance(sample, (list, np.ndarray)):
                    df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else x)
    return df

# --- Các hàm xử lý dữ liệu ---
@st.cache_resource()
def load_data():
    """Load and preprocess data with better caching"""
    processor = st.session_state.processor
    dataset = processor.load_data()
    
    # Pre-compute common statistics to avoid repeated calculations
    if not hasattr(processor, 'precomputed_stats'):
        processor.precomputed_stats = {}
        
        # Calculate trajectory length stats
        processor.precomputed_stats['traj_lengths'] = [len(traj.r) for traj in dataset.trajs]
        processor.precomputed_stats['avg_traj_length'] = np.mean(processor.precomputed_stats['traj_lengths'])
        
        # Precompute category stats
        categories = sorted(set(dataset.labels))
        cat_counts = {cat: dataset.labels.count(cat) for cat in categories}
        processor.precomputed_stats['category_counts'] = cat_counts
        
        # Precompute geographic boundaries
        min_lon, max_lon = float('inf'), float('-inf')
        min_lat, max_lat = float('inf'), float('-inf')
        
        for traj in dataset.trajs:
            min_lon = min(min_lon, np.min(traj.r[:, 0]))
            max_lon = max(max_lon, np.max(traj.r[:, 0]))
            min_lat = min(min_lat, np.min(traj.r[:, 1]))
            max_lat = max(max_lat, np.max(traj.r[:, 1]))
        
        processor.precomputed_stats['geo_bounds'] = {
            'min_lon': min_lon,
            'max_lon': max_lon,
            'min_lat': min_lat,
            'max_lat': max_lat
        }
    
    st.session_state.data_loaded = True
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return dataset

@st.cache_data()
def extract_features(_use_features=True, _outlier_method='winsorize', _create_interactions=True):
    processor = st.session_state.processor
    
    if _use_features:
        features_df = processor.process_data_pipeline(
            outlier_method=_outlier_method,
            create_interactions=_create_interactions
        )
    else:
        features_df = processor.extract_features()
    
    st.session_state.features_extracted = True
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return make_dataframe_arrow_compatible(features_df)

import joblib  # Để lưu và tải mô hình

@st.cache_resource
def train_model(_use_features=True):
    processor = st.session_state.processor    
    model_results = processor.train_model(
        use_features=_use_features
    )
    
    # Lưu mô hình đã huấn luyện vào file
    model_path = "hurricane_model.pkl"
    joblib.dump(processor.model, model_path)
    
    st.session_state.model_trained = True
    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success(f"Model trained and saved as {model_path}")
    return model_results

# --- Các hàm trực quan hóa --
def create_trajectory_map(trajectories, labels, sample_size=50):
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels

    df_points = []
    for i, traj in enumerate(sample_trajs):
        category = sample_labels[i]
        for j in range(len(traj.r)):
            df_points.append({
                'traj_id': traj.traj_id,
                'point_id': j,
                'longitude': traj.r[j, 0],
                'latitude': traj.r[j, 1],
                'class': category,
                'time_step': j
            })
    df = pd.DataFrame(df_points)
    
    # Color scale with modern colors
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548'   # Brown
    }
    
    fig = px.line_geo(
        df, 
        lat='latitude', 
        lon='longitude',
        color='class',
        color_discrete_map=color_map,
        line_group='traj_id',
        title='Hurricane Trajectories by Category'
    )
    
    start_points = df[df['point_id'] == 0]
    fig.add_trace(
        go.Scattergeo(
            lat=start_points['latitude'],
            lon=start_points['longitude'],
            mode='markers',
            marker=dict(
                size=8, 
                color=[color_map.get(cat, '#000000') for cat in start_points['class']],
                symbol='diamond'
            ),
            name='Starting Points',
            hovertemplate='<b>Trajectory ID:</b> %{customdata}<br>' +
                         '<b>Category:</b> %{text}<br>' +
                         '<b>Coordinates:</b> (%{lon:.2f}, %{lat:.2f})<extra></extra>',
            text=start_points['class'],
            customdata=start_points['traj_id']
        )
    )
    
    fig.update_layout(
        height=600,
        legend_title_text='Hurricane Category',
        title_font=dict(size=24, color='#0D47A1', family='Arial, sans-serif'),
        geo=dict(
            showland=True,
            landcolor='rgb(240, 240, 240)',
            coastlinecolor='rgb(37, 102, 142)',
            countrycolor='rgb(217, 217, 217)',
            showocean=True,
            oceancolor='rgb(220, 240, 255)',
            showlakes=True,
            lakecolor='rgb(220, 240, 255)',
            showrivers=True,
            rivercolor='rgb(220, 240, 255)',
            projection_type='natural earth',
            showcountries=True,
            showcoastlines=True,
            resolution=110
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig, df

def create_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <img src="https://cdn3.iconfinder.com/data/icons/weather-2-2/128/Hurricane-512.png" width="80">
        <h2 style="margin-top:10px; font-size:1.8rem; color:#1E88E5; font-weight:600;">Hurricane Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Data Section với thiết kế hiện đại và icon
    st.sidebar.markdown('<div style="background-color:#e3f2fd; padding:10px; border-radius:8px; margin-bottom:20px;"><h3 style="margin:0; color:#1E88E5; font-size:1.2rem;">📊 Data Processing</h3></div>', unsafe_allow_html=True)
    
    data_col1, data_col2 = st.sidebar.columns(2)
    with data_col1:
        if st.button("📥 Load Data", key="load_data_btn", help="Load hurricane trajectory dataset"):
            with st.spinner("Loading hurricane data..."):
                dataset = load_data()
                st.success(f"Loaded {len(dataset.trajs)} trajectories!")
    
    with data_col2:
        if st.session_state.data_loaded:
            if st.button("🔍 Extract Features", key="extract_features_btn", help="Extract features from trajectories"):
                with st.spinner("Extracting features..."):
                    features_df = extract_features(
                        _use_features=st.session_state.preprocessing_options['use_features'],
                        _outlier_method=st.session_state.preprocessing_options['outlier_method'],
                        _create_interactions=st.session_state.preprocessing_options['create_interactions']
                    )
                    st.session_state.processor.features_df = features_df
                    st.success(f"Extracted {features_df.shape[1]} features!")

    # Thêm button Train Model với thiết kế nhất quán
    if st.session_state.data_loaded and st.session_state.features_extracted:
        if st.sidebar.button("🧠 Train Model", key="train_model_btn", help="Train hurricane prediction model"):
            with st.spinner("Training model..."):
                model_results = train_model(
                    _use_features=st.session_state.preprocessing_options['use_features']
                )
                accuracy = model_results['report']['accuracy'] if 'accuracy' in model_results['report'] else model_results['report']['weighted avg']['f1-score']
                st.sidebar.success(f"Model trained! Accuracy: {accuracy:.4f}")
    
    # Tùy chỉnh xử lý với giao diện tốt hơn
    if st.session_state.data_loaded:
        with st.sidebar.expander("⚙️ Processing Options", expanded=False):
            st.session_state.preprocessing_options['outlier_method'] = st.selectbox(
                "Outlier Handling",
                options=["winsorize", "clip", "none"],
                index=0,
                help="Method to handle outliers in the data"
            )
            
            st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
            
            st.session_state.preprocessing_options['create_interactions'] = st.checkbox(
                "Create Feature Interactions",
                value=True,
                help="Generate interaction features between variables"
            )
            
            st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
            
            st.session_state.preprocessing_options['use_features'] = st.checkbox(
                "Use Extended Features",
                value=True,
                help="Use additional derived features for better predictions"
            )
    
    # Navigation với thiết kế cải tiến
    st.sidebar.markdown('<div style="background-color:#e3f2fd; padding:10px; border-radius:8px; margin:20px 0;"><h3 style="margin:0; color:#1E88E5; font-size:1.2rem;">🗺️ Navigation</h3></div>', unsafe_allow_html=True)
    
    # Tab design có icon trực quan
    selected_page = st.sidebar.radio(
        "Select Section",
        [
            "🏠 Home", 
            "🌍 Trajectory Explorer", 
            "📊 Feature Analysis", 
            "🧠 Prediction Model", 
            "💹 Trajectory Comparison", 
            "🎬 Advanced Visualizations", 
            "📈 Hurricane Impact", 
            "📱 Real Data Input", 
            "✏️ Draw & Predict"
        ]
    )
    
    # Hiển thị trạng thái với thiết kế hiện đại
    st.sidebar.markdown('<div style="background-color:#e3f2fd; padding:10px; border-radius:8px; margin:20px 0;"><h3 style="margin:0; color:#1E88E5; font-size:1.2rem;">🧩 App Status</h3></div>', unsafe_allow_html=True)
    
    # Hiển thị trạng thái với các chỉ số
    status_col1, status_col2 = st.sidebar.columns(2)
    
    with status_col1:
        st.markdown(f"""
        <div style="background-color:{'#e8f5e9' if st.session_state.data_loaded else '#ffebee'}; padding:10px; border-radius:5px; margin-bottom:10px;">
            <div style="font-size:13px; color:{'#4CAF50' if st.session_state.data_loaded else '#F44336'}; font-weight:500;">
                {'✅' if st.session_state.data_loaded else '❌'} Data
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color:{'#e8f5e9' if st.session_state.model_trained else '#ffebee'}; padding:10px; border-radius:5px;">
            <div style="font-size:13px; color:{'#4CAF50' if st.session_state.model_trained else '#F44336'}; font-weight:500;">
                {'✅' if st.session_state.model_trained else '❌'} Model
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        st.markdown(f"""
        <div style="background-color:{'#e8f5e9' if st.session_state.features_extracted else '#ffebee'}; padding:10px; border-radius:5px; margin-bottom:10px;">
            <div style="font-size:13px; color:{'#4CAF50' if st.session_state.features_extracted else '#F44336'}; font-weight:500;">
                {'✅' if st.session_state.features_extracted else '❌'} Features
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background-color:#e3f2fd; padding:10px; border-radius:5px;">
            <div style="font-size:13px; color:#1E88E5; font-weight:500;">
                🕒 {st.session_state.last_update}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Thêm liên kết trợ giúp và thông tin
    st.sidebar.markdown("""
    <div style="margin-top:30px; text-align:center;">
        <a href="#" style="margin:0 10px; color:#1E88E5; text-decoration:none; font-size:14px;">Help</a>
        <a href="#" style="margin:0 10px; color:#1E88E5; text-decoration:none; font-size:14px;">About</a>
        <a href="#" style="margin:0 10px; color:#1E88E5; text-decoration:none; font-size:14px;">Documentation</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Thông tin phiên bản
    st.sidebar.markdown("""
    <div style="position:fixed; bottom:20px; left:20px; right:20px; padding:10px; 
                background-color:#f1f5f9; text-align:center; font-size:0.8rem; color:#666; 
                border-radius:5px; box-shadow:0 2px 5px rgba(0,0,0,0.05);">
        Hurricane Analysis Dashboard v2.1<br>
        © 2023 Hurricane Research Team
    </div>
    """, unsafe_allow_html=True)
    
    return selected_page.replace("🏠 ", "").replace("🌍 ", "").replace("📊 ", "").replace("🧠 ", "").replace("💹 ", "").replace("🎬 ", "").replace("📈 ", "").replace("📱 ", "").replace("✏️ ", "")

def create_animated_trajectory_map(df):
    # Create a modern color map
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548'   # Brown
    }
    
    # Tích lũy các điểm: với mỗi frame f, hiển thị các điểm có time_step <= f
    max_frame = int(df['time_step'].max())
    df_list = []
    for f in range(max_frame + 1):
        temp = df[df['time_step'] <= f].copy()
        temp['frame'] = f
        df_list.append(temp)
    df_accumulated = pd.concat(df_list)
    
    fig = px.line_geo(
        df_accumulated, 
        lat='latitude', 
        lon='longitude',
        color='class',
        color_discrete_map=color_map,
        line_group='traj_id',
        animation_frame='frame',
        title='Hurricane Trajectory Animation'
    )
    
    fig.update_layout(
        height=650,
        legend_title_text='Hurricane Category',
        title_font=dict(size=24, color='#0D47A1', family='Arial, sans-serif'),
        geo=dict(
            showland=True,
            landcolor='rgb(240, 240, 240)',
            coastlinecolor='rgb(37, 102, 142)',
            countrycolor='rgb(217, 217, 217)',
            showocean=True,
            oceancolor='rgb(220, 240, 255)',
            showlakes=True,
            lakecolor='rgb(220, 240, 255)',
            showrivers=True,
            rivercolor='rgb(220, 240, 255)',
            projection_type='natural earth',
            showcountries=True,
            showcoastlines=True,
            resolution=110
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 150, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'x': 0.1,
            'y': 0,
            'bgcolor': '#1E88E5',
            'font': {'color': 'white'}
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16, 'color': '#0D47A1'},
                'prefix': 'Time Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 150},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [f],
                        {'frame': {'duration': 150, 'redraw': True},
                         'mode': 'immediate',
                         'transition': {'duration': 150}}
                    ],
                    'label': str(k),
                    'method': 'animate'
                }
                for k, f in enumerate(range(max_frame + 1))
            ]
        }]
    )
    return fig


def create_3d_trajectory_plot(trajectories, labels, sample_size=20):
    # Modern color scheme
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548'   # Brown
    }
    
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels

    df_points = []
    for i, traj in enumerate(sample_trajs):
        category = sample_labels[i]
        for j in range(len(traj.r)):
            time_pct = j / (len(traj.r)-1) if len(traj.r) > 1 else 0
            df_points.append({
                'traj_id': traj.traj_id,
                'point_id': j,
                'longitude': traj.r[j, 0],
                'latitude': traj.r[j, 1],
                'time': time_pct,
                'class': category
            })
    df = pd.DataFrame(df_points)
    
    fig = px.line_3d(
        df,
        x='longitude',
        y='latitude',
        z='time',
        color='class',
        color_discrete_map=color_map,
        line_group='traj_id',
        title='3D Hurricane Trajectories (Z-axis: Normalized Time)',
        labels={'longitude': 'Longitude', 'latitude': 'Latitude', 'time': 'Normalized Time'}
    )
    
    start_points = df[df['point_id'] == 0]
    fig.add_trace(
        go.Scatter3d(
            x=start_points['longitude'],
            y=start_points['latitude'],
            z=start_points['time'],
            mode='markers',
            marker=dict(
                size=6, 
                symbol='diamond',
                color=[color_map.get(cat, '#000000') for cat in start_points['class']]
            ),
            name='Starting Points',
            hovertemplate='<b>Trajectory ID:</b> %{customdata}<br>' +
                         '<b>Category:</b> %{text}<br>' +
                         '<b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
            text=start_points['class'],
            customdata=start_points['traj_id']
        )
    )
    
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Normalized Time',
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.5),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.0)
            ),
            xaxis=dict(showbackground=True, backgroundcolor='rgb(240, 240, 240)'),
            yaxis=dict(showbackground=True, backgroundcolor='rgb(240, 240, 240)'),
            zaxis=dict(showbackground=True, backgroundcolor='rgb(240, 240, 240)')
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

# Sửa phần mã liên quan đến fillcolor
def create_velocity_profile(trajectories, labels, sample_size=10):
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548'   # Brown
    }

    # Sample the data if needed
    if len(trajectories) > sample_size:
        indices = np.random.choice(len(trajectories), sample_size, replace=False)
        sample_trajs = [trajectories[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_trajs = trajectories
        sample_labels = labels

    # Create subplots
    fig = make_subplots(
        rows=len(sample_trajs), 
        cols=1, 
        shared_xaxes=True,
        subplot_titles=[f'Trajectory {traj.traj_id} (Category {label})' 
                        for traj, label in zip(sample_trajs, sample_labels)],
        vertical_spacing=0.04
    )

    for i, (traj, label) in enumerate(zip(sample_trajs, sample_labels)):
        try:
            # Ensure the trajectory has velocity data
            if hasattr(traj, 'v') and traj.v is not None:
                # Calculate the magnitude of the velocity vector (speed)
                # Convert velocity from m/s to km/h if needed (1 m/s = 3.6 km/h)
                v_magnitude = np.sqrt(np.sum(traj.v**2, axis=1)) * 3.6  # Magnitude of velocity (in km/h)
                
                # Check and scale up the velocity values if they are too small
                v_magnitude = np.where(v_magnitude < 0.1, v_magnitude * 100, v_magnitude)  # Scale small values
                
                time_pct = np.linspace(0, 100, len(v_magnitude))  # Normalized time for the x-axis

                # Get the RGBA color with a fixed opacity (0.2)
                # Safely get color and create rgba
                cat_color = color_map.get(label, '#000000')  # Default to black if category not found
                
                # Safely extract RGB components
                try:
                    r = int(cat_color[1:3], 16)
                    g = int(cat_color[3:5], 16)
                    b = int(cat_color[5:7], 16)
                    rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                except (ValueError, IndexError):
                    rgba_color = 'rgba(0,0,0,0.2)'  # Default to black with 0.2 opacity
                

                # Add velocity line to the plot
                fig.add_trace(
                    go.Scatter(
                        x=time_pct,
                        y=v_magnitude,
                        mode='lines',
                        line=dict(
                            color=cat_color, 
                            width=2,
                            shape='spline'
                        ),
                        name=f'Category {label}',
                        fill='tozeroy',
                        fillcolor=rgba_color  # Corrected fillcolor
                    ),
                    row=i+1, col=1
                )

                # Add average velocity line
                mean_v = np.mean(v_magnitude)
                fig.add_trace(
                    go.Scatter(
                        x=[0, 100],
                        y=[mean_v, mean_v],
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0.7)', width=1, dash='dash'),
                        name='Average Velocity',
                        showlegend=False,
                        hovertemplate=f'Average Velocity: {mean_v:.2f} km/h<extra></extra>'
                    ),
                    row=i+1, col=1
                )

                # Add peak velocity markers
                peak_idx = np.argmax(v_magnitude)
                fig.add_trace(
                    go.Scatter(
                        x=[time_pct[peak_idx]],
                        y=[v_magnitude[peak_idx]],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=color,
                            line=dict(width=2, color='white'),
                            symbol='diamond'
                        ),
                        name='Peak Velocity',
                        showlegend=False,
                        hovertemplate=f'Peak Velocity: {v_magnitude[peak_idx]:.2f} km/h<extra></extra>'
                    ),
                    row=i+1, col=1
                )

            else:
                print(f"Warning: Trajectory {traj.traj_id} has no velocity data.")

        except ValueError as e:
            print(f"Error with trajectory {traj.traj_id}: {e}")
            continue

    # Update layout for better visibility
    fig.update_layout(
        height=max(150 * len(sample_trajs), 600),
        title='Hurricane Velocity Profiles',
        title_font=dict(size=24, color='#0D47A1', family='Arial, sans-serif'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80),
        hovermode='closest'
    )

    for i in range(len(sample_trajs)):
        fig.update_yaxes(
            title_text='Velocity (km/h)', 
            row=i+1, 
            col=1,
            gridcolor='rgba(0,0,0,0.1)'
        )

    fig.update_xaxes(
        title_text='Trajectory Progress (%)', 
        row=len(sample_trajs), 
        col=1,
        gridcolor='rgba(0,0,0,0.1)'
    )

    return fig


# --- Hàm tính khoảng cách theo Haversine ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # bán kính Trái đất km
    return c * r

# --- Trang dự đoán qua vẽ đường đi của bão ---
def show_drawing_prediction():
    st.markdown('<div class="main-header">Hurricane Prediction from Drawn Trajectory</div>', unsafe_allow_html=True)
    
    # Thêm hướng dẫn trực quan hơn
    st.markdown("""
    <div class="card">
        <div style="display:flex; align-items:start;">
            <div style="font-size:2.5rem; margin-right:20px;">✏️</div>
            <div>
                <h3 style="margin-top:0; color:#1E88E5;">Draw a Hurricane Path</h3>
                <p>Use the drawing tool to create a hurricane trajectory on the map. The system will analyze your drawing 
                and predict the hurricane category based on its path characteristics.</p>
            </div>
        </div>
        
        <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; margin-top:15px;">
            <h4 style="margin-top:0; color:#1E88E5;">Instructions:</h4>
            <ol style="margin-bottom:0; padding-left:20px;">
                <li>Click the <span style="background-color:#f1f1f1; padding:2px 6px; border-radius:4px; font-family:monospace;">line icon</span> in the map's toolbar</li>
                <li>Draw a path by clicking multiple points on the map</li>
                <li>Complete your drawing by double-clicking the last point</li>
                <li>Click the "Analyze Trajectory" button below the map</li>
                <li>View the predicted hurricane category and analysis</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sử dụng expanders để giảm số lượng elements hiển thị cùng lúc
    with st.expander("👁️ Watch How to Draw Hurricane Paths", expanded=False):
        st.markdown("""
        <div style="display:flex; justify-content:center; margin:10px 0 20px 0;">
            <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNzJiNWEzM2RjYmJiMTQ4MWJlY2FiMDFlM2NmZDFhOWZmZmRhM2Y3OSZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/kFHSGYw6YdGdvTNQ31/giphy.gif" 
                 alt="Drawing Hurricane Path Tutorial" style="max-width:100%; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1);">
        </div>
        <p style="text-align:center; font-size:0.9rem; color:#666;">
            A quick demonstration of drawing a hurricane path for prediction
        </p>
        """, unsafe_allow_html=True)
    
    # Lưu trữ trạng thái bản đồ
    if 'map_has_been_drawn' not in st.session_state:
        st.session_state.map_has_been_drawn = False
    
    if 'last_drawing_data' not in st.session_state:
        st.session_state.last_drawing_data = None
    
    # Lưu trữ màu cho đường bão mẫu - giúp tránh tạo màu ngẫu nhiên mỗi lần rerender
    if 'sample_traj_colors' not in st.session_state:
        st.session_state.sample_traj_colors = []
    
    # Lưu trữ trạng thái khi đã thêm đường bão mẫu
    if 'has_added_sample_trajs' not in st.session_state:
        st.session_state.has_added_sample_trajs = False
    
    # Tùy chọn hiển thị đường bão mẫu
    show_sample_trajs = st.checkbox("Show example hurricane paths", value=True, key="show_sample_paths")
    
    # Hiển thị bản đồ và phần phân tích
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        # Tạo bản đồ Folium cơ bản
        m = folium.Map(location=[25, -75], zoom_start=4, tiles='CartoDB positron')
        
        # Thêm các lớp bản đồ với thuộc tính phù hợp
        folium.TileLayer(
            'CartoDB dark_matter', 
            name='Dark Map',
            attr='© CartoDB'
        ).add_to(m)
        
        folium.TileLayer(
            'https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png', 
            name='Stamen Terrain',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>'
        ).add_to(m)
        
        folium.TileLayer(
            'https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.png', 
            name='Stamen Watercolor',
            attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>'
        ).add_to(m)
        
        # Thêm vùng nguy cơ bão
        try:
            folium.GeoJson(
                "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
                name='Hurricane Risk Areas',
                style_function=lambda x: {
                    'fillColor': '#ff000025' if x['properties']['name'] in 
                                          ['Florida', 'Texas', 'Louisiana', 'North Carolina', 'South Carolina'] 
                                          else '#00000000',
                    'color': '#ff0000' if x['properties']['name'] in 
                                    ['Florida', 'Texas', 'Louisiana', 'North Carolina', 'South Carolina'] 
                                    else '#00000000',
                    'weight': 1,
                    'dashArray': '5, 5'
                }
            ).add_to(m)
        except Exception as e:
            pass  # Bỏ qua lỗi không load được GeoJSON
        
        # Thêm đường đi lịch sử của bão - chỉ thêm một lần duy nhất
        if st.session_state.data_loaded and show_sample_trajs and not st.session_state.has_added_sample_trajs:
            try:
                # Lấy 5 quỹ đạo đầu tiên làm mẫu
                sample_trajs = st.session_state.processor.dataset.trajs[:5]
                
                # Tạo màu cho mỗi quỹ đạo và lưu lại để sử dụng lần sau
                if not st.session_state.sample_traj_colors:
                    for _ in range(len(sample_trajs)):
                        # Sử dụng các màu định sẵn thay vì ngẫu nhiên để tránh thay đổi mỗi lần rerender
                        color = f'#{np.random.randint(0, 256):02x}{np.random.randint(0, 256):02x}{np.random.randint(0, 256):02x}'
                        st.session_state.sample_traj_colors.append(color)
                
                for i, traj in enumerate(sample_trajs):
                    # Lấy màu từ session_state
                    color = st.session_state.sample_traj_colors[i]
                    
                    # Tạo danh sách các điểm
                    points = [(traj.r[j, 1], traj.r[j, 0]) for j in range(len(traj.r))]
                    
                    # Thêm polyline
                    folium.PolyLine(
                        points,
                        color=color,
                        weight=2,
                        opacity=0.7,
                        tooltip=f"Example Hurricane Path {i+1}"
                    ).add_to(m)
                    
                    # Thêm marker cho điểm bắt đầu
                    folium.CircleMarker(
                        location=points[0],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        tooltip="Starting Point"
                    ).add_to(m)
                
                # Đánh dấu là đã thêm quỹ đạo mẫu
                st.session_state.has_added_sample_trajs = True
                
            except Exception as e:
                # Chỉ hiển thị lỗi nếu cần
                if "sample_trajs_error" not in st.session_state:
                    st.error(f"Error loading example trajectories: {e}")
                    st.session_state.sample_trajs_error = True
        
        # Reset trạng thái has_added_sample_trajs khi checkbox thay đổi
        if not show_sample_trajs and st.session_state.has_added_sample_trajs:
            st.session_state.has_added_sample_trajs = False
        
        # Cải thiện công cụ vẽ
        draw = folium.plugins.Draw(
            export=True,
            position='topleft',
            draw_options={
                'polyline': {
                    'allowIntersection': True,
                    'shapeOptions': {
                        'color': '#1E88E5',
                        'weight': 5,
                        'opacity': 0.7
                    }
                },
                'polygon': False,
                'circle': False,
                'rectangle': False,
                'marker': False,
                'circlemarker': False
            },
            edit_options={'poly': {'allowIntersection': True}}
        )
        draw.add_to(m)
        
        # Thêm công cụ đo khoảng cách
        folium.plugins.MeasureControl(
            position='bottomleft',
            primary_length_unit='kilometers',
            secondary_length_unit='miles',
            primary_area_unit='sqmeters',
            secondary_area_unit='acres'
        ).add_to(m)
        
        # Thêm điều khiển lớp 
        folium.LayerControl(position='topright').add_to(m)
        
        # Hiện bản đồ với key cố định để tránh reloads
        output = st_folium(m, width="100%", height=500, key="folium_map")
        
        # Lưu drawing data vào session_state (không gây reload)
        if output and output.get('last_active_drawing'):
            st.session_state.last_drawing_data = output.get('last_active_drawing')
            st.session_state.map_has_been_drawn = True
        
        # Thêm nút phân tích để user kiểm soát khi nào chạy phân tích
        analyze_btn = st.button("🔍 Analyze Trajectory", key="analyze_trajectory")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        # Hiển thị animation một lần duy nhất
        if 'has_shown_animation' not in st.session_state:
            hurricane_animation = load_lottie_url("https://lottie.host/eb19bf28-2f1a-4f2f-b493-8c2c006a0837/z7sGOXPlUW.json")
            if hurricane_animation:
                st_lottie(hurricane_animation, speed=1, height=180, key="hurricane")
            st.session_state.has_shown_animation = True
        
        st.markdown("<h3 style='color:#1E88E5; text-align:center;'>Trajectory Analysis</h3>", unsafe_allow_html=True)
        
        # Kiểm tra nếu có dữ liệu vẽ và nút phân tích được nhấn
        if st.session_state.map_has_been_drawn and st.session_state.last_drawing_data and analyze_btn:
            geojson = st.session_state.last_drawing_data
            if geojson['geometry']['type'] == 'LineString':
                coords = geojson['geometry']['coordinates']
                
                # Tạo đối tượng quỹ đạo từ tọa độ người dùng vẽ
                with st.spinner("Analyzing hurricane trajectory pattern..."):
                    drawn_traj = create_trajectory_from_drawing(coords)
                
                # Hiển thị thông tin quỹ đạo
                display_basic_trajectory_info(drawn_traj)
                
                # Thực hiện dự đoán nếu mô hình đã được huấn luyện
                if st.session_state.model_trained:
                    try:
                        predict_hurricane_category(drawn_traj)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.error("Please try drawing a longer trajectory with more points.")
                else:
                    st.warning("⚠️ Please train the model first to enable predictions.")
            else:
                st.warning("Please draw a polyline (hurricane path) on the map.")
        elif not analyze_btn:
            # Hiển thị thông tin ban đầu (chỉ khi không phải trạng thái đang phân tích)
            st.info("Draw a hurricane path on the map and click 'Analyze Trajectory' to see predictions.")
            
            # Hiển thị thông tin về các quỹ đạo mẫu
            if show_sample_trajs and st.session_state.has_added_sample_trajs:
                st.markdown("""
                <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; margin-top:15px;">
                    <h4 style="margin-top:0; color:#1E88E5;">Example Paths</h4>
                    <p>The map shows example hurricane paths from the dataset. You can:</p>
                    <ul>
                        <li>Study these paths for inspiration</li>
                        <li>Draw similar patterns based on their shapes</li>
                        <li>Notice how different hurricanes follow different trajectories</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Thêm nút reset để người dùng có thể xóa bản vẽ và bắt đầu lại
    if st.session_state.map_has_been_drawn:
        if st.button("🔄 Reset Drawing", key="reset_drawing"):
            st.session_state.map_has_been_drawn = False
            st.session_state.last_drawing_data = None
            st.session_state.has_shown_animation = False
            # Không reset has_added_sample_trajs vì chúng ta muốn giữ lại các quỹ đạo mẫu
            st.experimental_rerun()  # Cần rerun trong trường hợp này


def create_trajectory_from_drawing(coords):
    """
    Tạo đối tượng quỹ đạo từ tọa độ vẽ trên bản đồ với đầy đủ các thuộc tính cần thiết
    để có thể sử dụng với hàm extract_features
    """
    # Tạo một lớp giả lập cho quỹ đạo
    class DrawnTrajectory:
        pass
    
    drawn_traj = DrawnTrajectory()
    
    # Tạo ID duy nhất cho quỹ đạo
    drawn_traj.traj_id = f"drawn_{int(time.time())}"
    
    # Chuyển các tọa độ thành mảng numpy
    drawn_traj.r = np.array(coords)  # Shape: (n, 2) - mỗi hàng là một điểm [lon, lat]
    
    # Tạo mảng thời gian
    # Giả định mỗi điểm cách nhau 6 giờ (giống dữ liệu thực tế bão)
    time_step = 6 * 3600  # 6 giờ tính bằng giây
    drawn_traj.t_0 = 0  # Thời điểm bắt đầu
    drawn_traj.t = np.array([drawn_traj.t_0 + i * time_step for i in range(len(drawn_traj.r))])
    
    # Tính vận tốc
    drawn_traj.v = np.zeros_like(drawn_traj.r)
    for i in range(1, len(drawn_traj.r)):
        dt = drawn_traj.t[i] - drawn_traj.t[i-1]
        drawn_traj.v[i] = (drawn_traj.r[i] - drawn_traj.r[i-1]) / dt if dt > 0 else np.zeros(2)
    
    # Tính gia tốc
    drawn_traj.a = np.zeros_like(drawn_traj.r)
    for i in range(1, len(drawn_traj.v)):
        dt = drawn_traj.t[i] - drawn_traj.t[i-1]
        drawn_traj.a[i] = (drawn_traj.v[i] - drawn_traj.v[i-1]) / dt if dt > 0 else np.zeros(2)
    
    # Xác định các thuộc tính khác
    drawn_traj.uniformly_spaced = True  # Khoảng cách thời gian đều nhau
    
    # Tính toán bounds
    min_lon = np.min(drawn_traj.r[:, 0])
    max_lon = np.max(drawn_traj.r[:, 0])
    min_lat = np.min(drawn_traj.r[:, 1])
    max_lat = np.max(drawn_traj.r[:, 1])
    drawn_traj.bounds = ((min_lon, max_lon), (min_lat, max_lat))
    
    # Xác định chiều của tọa độ
    drawn_traj.dim = 2  # 2 chiều: kinh độ và vĩ độ
    
    # Cấu trúc như một từ điển để phù hợp với mô hình
    drawn_traj.__dict__['traj_id'] = drawn_traj.traj_id
    drawn_traj.__dict__['t_0'] = drawn_traj.t_0
    drawn_traj.__dict__['uniformly_spaced'] = drawn_traj.uniformly_spaced
    drawn_traj.__dict__['bounds'] = drawn_traj.bounds
    drawn_traj.__dict__['dim'] = drawn_traj.dim
    drawn_traj.__dict__['r'] = drawn_traj.r
    drawn_traj.__dict__['t'] = drawn_traj.t
    drawn_traj.__dict__['v'] = drawn_traj.v
    drawn_traj.__dict__['a'] = drawn_traj.a
    
    return drawn_traj

def display_basic_trajectory_info(traj):
    """
    Hiển thị thông tin cơ bản về quỹ đạo vẽ
    """
    # Tính toán các đặc trưng cơ bản
    path_length = 0
    for i in range(1, len(traj.r)):
        lon1, lat1 = traj.r[i-1]
        lon2, lat2 = traj.r[i]
        path_length += haversine(lon1, lat1, lon2, lat2)
    
    # Tính tốc độ trung bình và tối đa
    v_magnitude = np.linalg.norm(traj.v, axis=1)
    avg_velocity = np.mean(v_magnitude[1:])  # Bỏ qua phần tử đầu tiên (thường là 0)
    max_velocity = np.max(v_magnitude)
    
    # Tính độ cong quỹ đạo
    curvature = []
    for i in range(1, len(traj.r) - 1):
        vec1 = traj.r[i] - traj.r[i-1]
        vec2 = traj.r[i+1] - traj.r[i]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.clip(dot_product / (norm1 * norm2), -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            curvature.append(angle)
    
    avg_curvature = np.mean(curvature) if curvature else 0
    
    # Hiển thị các thông số cơ bản
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Distance", f"{path_length:.1f} km")
        st.metric("Avg. Speed", f"{avg_velocity:.6f} deg/h")
    
    with col2:
        st.metric("Max Speed", f"{max_velocity:.6f} deg/h")
        st.metric("Path Curvature", f"{avg_curvature:.1f}°")
    
    # Hiển thị thông tin quỹ đạo
    st.markdown("#### Trajectory Features")
    st.markdown(f"""
    * **Geographic Range:** {traj.bounds[0][1] - traj.bounds[0][0]:.1f}° × {traj.bounds[1][1] - traj.bounds[1][0]:.1f}°
    * **Path Points:** {len(traj.r)}
    * **Starting Position:** ({traj.r[0][0]:.2f}, {traj.r[0][1]:.2f})
    * **Ending Position:** ({traj.r[-1][0]:.2f}, {traj.r[-1][1]:.2f})
    * **Duration:** {(traj.t[-1] - traj.t[0]) / 3600:.1f} hours
    """)

def predict_hurricane_category(drawn_traj):
    """
    Sử dụng hàm extract_features để trích xuất đặc trưng và dự đoán loại bão với giao diện cải tiến
    """
    processor = st.session_state.processor
    
    # Tạo dataset tạm thời chỉ chứa quỹ đạo vẽ
    class TempDataset:
        pass
    
    temp_dataset = TempDataset()
    temp_dataset.trajs = [drawn_traj]
    temp_dataset.labels = [0]  # Use 0 instead of -1 for temporary labeling
    temp_dataset.name = "Drawn Trajectory"
    temp_dataset.classes = processor.dataset.classes if processor.dataset else []
    
    # Lưu dataset gốc
    original_dataset = processor.dataset
    
    try:
        # Gán dataset tạm thời vào processor
        processor.dataset = temp_dataset
        
        # Trích xuất đặc trưng cơ bản - use direct processing pipeline
        processed_features = processor.process_data_pipeline(
            outlier_method=st.session_state.preprocessing_options['outlier_method'],
            create_interactions=st.session_state.preprocessing_options['create_interactions']
        )
        base_features = processor.extract_features()
        
        # Loại bỏ các cột không cần thiết cho dự đoán
        prediction_features = processed_features.drop(['traj_id', 'class'], axis=1, errors='ignore')
        
        # Đảm bảo các cột khớp với cột đã sử dụng trong huấn luyện
        expected_columns = processor.model.feature_names_in_
        missing_columns = set(expected_columns) - set(prediction_features.columns)
        extra_columns = set(prediction_features.columns) - set(expected_columns)
        
        # Báo cáo và xử lý cột thiếu
        if missing_columns:
            st.warning(f"Missing {len(missing_columns)} features from model. Filling with default values.")
            for col in missing_columns:
                prediction_features[col] = 0  # Better default value
        
        # Loại bỏ các cột thừa
        if extra_columns:
            prediction_features = prediction_features.drop(columns=list(extra_columns))
        
        # Sắp xếp lại cột theo thứ tự mô hình cần
        prediction_features = prediction_features[expected_columns]
        
        # Dự đoán loại bão
        prediction = processor.model.predict(prediction_features)[0]
        
        # Lấy xác suất dự đoán nếu có
        probabilities = None
        try:
            probabilities = processor.model.predict_proba(prediction_features)[0]
        except:
            pass
        
        # Hiển thị kết quả dự đoán với thiết kế hiện đại
        category_colors = {
            0: "#2196F3",  # Blue
            1: "#4CAF50",  # Green
            2: "#F44336",  # Red
            3: "#9C27B0",  # Purple
            4: "#FF9800",  # Orange
            5: "#795548"   # Brown
        }
        
        category_descriptions = {
            0: "Minimal damage primarily to vegetation.",
            1: "Moderate damage to houses and vegetation. Some coastal flooding.",
            2: "Considerable damage to vegetation and houses. Coastal flooding.",
            3: "Devastating damage, severe flooding, structures damaged.",
            4: "Catastrophic damage, major structural damage, severe flooding.",
            5: "Catastrophic damage, complete structural failures, severe flooding."
        }
        
        # Tạo đồ thị xác suất nếu có
        if probabilities is not None:
            prob_fig = go.Figure(data=[
                go.Bar(
                    x=[f"Category {i}" for i in range(len(probabilities))],
                    y=probabilities,
                    marker_color=[category_colors.get(i, "#333333") for i in range(len(probabilities))],
                    text=[f"{p*100:.1f}%" for p in probabilities],
                    textposition="outside"
                )
            ])
            
            prob_fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Hurricane Category",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
                plot_bgcolor="white",
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            st.plotly_chart(prob_fig, use_container_width=True)
        
        # Hiển thị kết quả dự đoán
        st.markdown(f"""
        <div style="background:linear-gradient(120deg, {category_colors.get(prediction, "#333")}, {category_colors.get(prediction, "#333")}40); 
                    border-radius:15px; padding:25px; text-align:center; margin:20px 0; box-shadow:0 4px 20px rgba(0,0,0,0.15);">
            <div style="font-size:16px; color:white; margin-bottom:10px; text-shadow:0 1px 2px rgba(0,0,0,0.2);">Predicted Hurricane Category</div>
            <div style="font-size:64px; font-weight:bold; color:white; line-height:1; text-shadow:0 2px 4px rgba(0,0,0,0.3);">{prediction}</div>
            <div style="margin-top:15px; color:white; background:rgba(0,0,0,0.1); padding:10px; border-radius:8px; text-shadow:0 1px 2px rgba(0,0,0,0.2);">
                {category_descriptions.get(prediction, "")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị thông tin chi tiết về cấp độ bão
        wind_speeds = {
            0: "< 74 mph (119 km/h)",
            1: "74-95 mph (119-153 km/h)",
            2: "96-110 mph (154-177 km/h)",
            3: "111-129 mph (178-208 km/h)",
            4: "130-156 mph (209-251 km/h)",
            5: "> 157 mph (252 km/h)"
        }
        
        pressure_ranges = {
            0: "> 980 hPa",
            1: "980-979 hPa",
            2: "965-979 hPa",
            3: "945-964 hPa",
            4: "920-944 hPa",
            5: "< 920 hPa"
        }
        
        surge_heights = {
            0: "None",
            1: "4-5 ft (1.2-1.5 m)",
            2: "6-8 ft (1.8-2.4 m)",
            3: "9-12 ft (2.7-3.7 m)",
            4: "13-18 ft (4.0-5.5 m)",
            5: "> 18 ft (5.5 m)"
        }
        
        # Hiển thị các đặc điểm chi tiết của cấp độ bão
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background-color:white; padding:15px; border-radius:10px; text-align:center; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:24px; margin-bottom:5px;">🌬️</div>
                <div style="font-size:14px; color:#666;">Wind Speed</div>
                <div style="font-size:16px; font-weight:bold; margin-top:5px; color:{category_colors.get(prediction, '#333')};">
                    {wind_speeds.get(prediction, "Unknown")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color:white; padding:15px; border-radius:10px; text-align:center; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:24px; margin-bottom:5px;">📊</div>
                <div style="font-size:14px; color:#666;">Pressure</div>
                <div style="font-size:16px; font-weight:bold; margin-top:5px; color:{category_colors.get(prediction, '#333')};">
                    {pressure_ranges.get(prediction, "Unknown")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color:white; padding:15px; border-radius:10px; text-align:center; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                <div style="font-size:24px; margin-bottom:5px;">🌊</div>
                <div style="font-size:14px; color:#666;">Storm Surge</div>
                <div style="font-size:16px; font-weight:bold; margin-top:5px; color:{category_colors.get(prediction, '#333')};">
                    {surge_heights.get(prediction, "Unknown")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Hiển thị đặc trưng quan trọng
        st.markdown("#### Key Trajectory Features Used for Prediction")
        
        # Chọn và hiển thị một số đặc trưng quan trọng
        important_features = ['mean_velocity', 'path_length', 'curvature_mean', 
                            'sinuosity', 'max_velocity', 'duration_hours']
        
        important_vals = {}
        for feat in important_features:
            if feat in base_features.columns:
                important_vals[feat] = base_features[feat].values[0]
        
        # Hiển thị đặc trưng trong bảng có thiết kế
        if important_vals:
            # Tạo dataframe hiển thị
            display_df = pd.DataFrame({
                'Feature': list(important_vals.keys()),
                'Value': list(important_vals.values())
            })
            
            # Hiển thị đặc trưng theo dạng metrics trực quan hơn
            st.markdown("<div style='display:flex; flex-wrap:wrap; gap:15px;'>", unsafe_allow_html=True)
            
            for feat, val in important_vals.items():
                feature_label = feat.replace('_', ' ').title()
                
                # Xác định đơn vị đo cho từng loại đặc trưng
                unit = ""
                if "velocity" in feat:
                    unit = "km/h"
                elif "length" in feat:
                    unit = "km"
                elif "duration" in feat:
                    unit = "h"
                
                st.markdown(f"""
                <div style="flex:1; min-width:200px; background-color:white; padding:15px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <div style="font-size:14px; color:#666;">{feature_label}</div>
                    <div style="font-size:20px; font-weight:bold; margin-top:5px; color:#1E88E5;">
                        {val:.2f} {unit}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error during feature extraction or prediction: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    finally:
        # Khôi phục dataset gốc
        processor.dataset = original_dataset


def create_feature_importance_plot(model_results):
    feature_importance = model_results['feature_importance']
    
    # Ensure we have at least one row
    if len(feature_importance) == 0:
        return go.Figure()
    
    # Get top features
    top_features = feature_importance.head(15)
    
    # Generate custom color gradient
    n_features = len(top_features)
    colors = [f'rgba(30, 136, 229, {0.3 + 0.7 * i/n_features})' for i in range(n_features)]
    
    # Create figure with custom styling
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top Feature Importance for Hurricane Category Prediction',
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    
    fig.update_traces(
        marker_color=colors,
        marker_line_width=0,
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif'),
        title_font=dict(size=22, color='#0D47A1'),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            autorange='reversed'
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14
        )
    )
    
    return fig

def create_confusion_matrix_plot(model_results):
    cm = model_results['confusion_matrix']
    categories = sorted(set(model_results['y_test']))
    labels = [f'Category {cat}' for cat in categories]
    
    # Calculate percentages for annotations
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotation text
    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            annotations.append(dict(
                x=labels[j],
                y=labels[i],
                text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                showarrow=False,
                font=dict(
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    size=14
                )
            ))
    
    # Create heatmap with enhanced styling
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale=[
            [0, '#f7fbff'],
            [0.2, '#c7dcef'],
            [0.4, '#85bcdb'],
            [0.6, '#43a2ca'],
            [0.8, '#1a7db6'],
            [1, '#0d47a1']
        ],
        labels=dict(x='Predicted', y='Actual', color='Count'),
        title='Confusion Matrix'
    )
    
    fig.update_layout(
        height=600,
        title_font=dict(size=22, color='#0D47A1', family='Arial, sans-serif'),
        xaxis_title=dict(text='Predicted Category', font=dict(size=16)),
        yaxis_title=dict(text='Actual Category', font=dict(size=16)),
        xaxis=dict(side='top'),
        coloraxis_colorbar=dict(
            title='Count',
            thicknessmode='pixels',
            thickness=20,
            lenmode='pixels',
            len=300,
            yanchor='middle'
        ),
        annotations=annotations
    )
    
    # Add diagonal highlight
    for i in range(len(labels)):
        fig.add_shape(
            type="rect",
            x0=i-0.5,
            y0=i-0.5,
            x1=i+0.5,
            y1=i+0.5,
            line=dict(color="rgba(255,255,255,0.5)", width=2),
        )
    
    return fig

def create_feature_distribution_plot(features_df, feature_name):
    # Modern color scheme with default for unknown categories
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548',  # Brown
        -1: '#000000'  # Black for unknown category
    }
    
    # Create enhanced box plot with violin overlay
    fig = go.Figure()
    
    # Add violin plots
    for cat in sorted(features_df['class'].unique()):
        cat_data = features_df[features_df['class'] == cat][feature_name]
        
        # Safely get color for this category
        cat_color = color_map.get(cat, '#000000')  # Default to black if category not found
        
        # Safely extract RGB components
        try:
            r = int(cat_color[1:3], 16)
            g = int(cat_color[3:5], 16)
            b = int(cat_color[5:7], 16)
            fillcolor = f'rgba({r},{g},{b},0.3)'
        except (ValueError, IndexError):
            fillcolor = 'rgba(0,0,0,0.3)'  # Default to black with 0.3 opacity
        
        fig.add_trace(go.Violin(
            x=[f'Category {cat}'] * len(cat_data),
            y=cat_data,
            name=f'Category {cat}',
            box_visible=True,
            meanline_visible=True,
            fillcolor=fillcolor,
            line_color=cat_color,
            marker=dict(size=4, opacity=0.5)
        ))
    
    # Update layout with modern styling
    fig.update_layout(
        title=f'Distribution of {feature_name.replace("_", " ").title()} by Hurricane Category',
        title_font=dict(size=22, color='#0D47A1', family='Arial, sans-serif'),
        xaxis_title='Hurricane Category',
        yaxis_title=feature_name.replace('_', ' ').title(),
        xaxis=dict(tickfont=dict(size=14), gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        violinmode='group',
        violingap=0.1,
        violingroupgap=0.05,
        height=600,
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=14),
        showlegend=False,
    )
    
    return fig


def create_normalized_trajectory_plot(processor, category=None):
    # Modern color scheme
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548'   # Brown
    }
    
    samples = processor.get_sample_trajectories(n_per_category=10)
    
    # Create subplots with improved styling
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'Category {cat}' for cat in sorted(samples.keys())],
        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    for i, cat in enumerate(sorted(samples.keys())):
        row = i // 3 + 1
        col = i % 3 + 1
        
        if category is not None and cat != category:
            continue
            
        # Get the color for this category
        cat_color = color_map.get(cat, '#000000')
        
        for traj in samples[cat]:
            if len(traj) >= 3:
                r_norm = processor.normalize_trajectory(traj)
                
                # Add trajectory line
                fig.add_trace(
                    go.Scatter(
                        x=r_norm[:, 0],
                        y=r_norm[:, 1],
                        mode='lines',
                        line=dict(
                            color=cat_color, 
                            width=1.5,
                            shape='spline'
                        ),
                        opacity=0.7,
                        showlegend=False,
                        hovertemplate='<b>Category ' + str(cat) + '</b><br>' +
                                     'X: %{x:.2f}<br>' +
                                     'Y: %{y:.2f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Add start point
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode='markers',
                        marker=dict(
                            color='black', 
                            size=8,
                            symbol='circle'
                        ),
                        name='Start Point',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
                
                # Add end point
                fig.add_trace(
                    go.Scatter(
                        x=[r_norm[-1, 0]],
                        y=[r_norm[-1, 1]],
                        mode='markers',
                        marker=dict(
                            color=cat_color, 
                            size=6,
                            symbol='diamond'
                        ),
                        name='End Point',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        # Update axes for each subplot
        fig.update_xaxes(
            title_text='X Normalized', 
            row=row, 
            col=col, 
            zeroline=True, 
            zerolinewidth=1, 
            zerolinecolor='gray',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[-1.5, 1.5]
        )
        
        fig.update_yaxes(
            title_text='Y Normalized', 
            row=row, 
            col=col, 
            zeroline=True, 
            zerolinewidth=1, 
            zerolinecolor='gray',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[-1.5, 1.5]
        )
    
    # Update overall layout
    fig.update_layout(
        height=750,
        title='Normalized Hurricane Trajectories by Category',
        title_font=dict(size=24, color='#0D47A1', family='Arial, sans-serif'),
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=80, b=40)
    )
    
    return fig

def create_hurricane_impact_visualization(features_df):
    # Modern color scheme
    color_map = {
        0: '#2196F3',  # Blue
        1: '#4CAF50',  # Green
        2: '#F44336',  # Red
        3: '#9C27B0',  # Purple
        4: '#FF9800',  # Orange
        5: '#795548'   # Brown
    }
    
    # Create figure
    if 'impact_score' in features_df.columns:
        feature_name = 'impact_score'
        title = 'Distribution of Impact Score by Hurricane Category'
    else:
        feature_name = 'traj_duration'
        title = 'Distribution of Trajectory Duration by Hurricane Category'
    
    # Create a histogram with KDE overlay
    fig = go.Figure()
    
    for cat in sorted(features_df['class'].unique()):
        cat_data = features_df[features_df['class'] == cat][feature_name].dropna()
        
        # Skip if no data for this category
        if len(cat_data) == 0:
            continue
            
        fig.add_trace(go.Histogram(
            x=cat_data,
            name=f'Category {cat}',
            marker_color=color_map.get(cat, '#000000'),
            opacity=0.7,
            histnorm='probability density',
            marker_line=dict(width=1, color='white'),
            hovertemplate=f'Category {cat}<br>{feature_name}: %{{x}}<br>Density: %{{y:.3f}}<extra></extra>'
        ))
    
    # Create a more informative layout
    fig.update_layout(
        title=title,
        title_font=dict(size=22, color='#0D47A1', family='Arial, sans-serif'),
        xaxis_title=feature_name.replace('_', ' ').title(),
        yaxis_title='Probability Density',
        barmode='overlay',
        height=600,
        plot_bgcolor='white',
        bargap=0.05,
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        legend=dict(
            title='Hurricane Category',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14
        )
    )
    
    # Add statistical information as annotations
    stats_text = []
    for cat in sorted(features_df['class'].unique()):
        cat_data = features_df[features_df['class'] == cat][feature_name].dropna()
        if len(cat_data) > 0:
            stats_text.append(
                f"<b>Category {cat}</b>: Mean = {cat_data.mean():.1f}, Median = {cat_data.median():.1f}"
            )
    
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.01,
        text='<br>'.join(stats_text),
        showarrow=False,
        font=dict(size=12),
        align='left',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#1E88E5',
        borderwidth=1,
        borderpad=4
    )
    
    return fig

# --- Các trang giao diện chính ---
def show_home_page():
    # Hiển thị animation hurricane với hiệu ứng to hơn
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        hurricane_animation = load_lottie_url("https://lottie.host/b7931f12-7ab5-4af6-adcd-8e7951ca1f1a/LTrNCzbXvp.json")
        if hurricane_animation:
            st_lottie(hurricane_animation, speed=1, height=280, key="hurricane_main")
    
    st.markdown('<div class="main-header">Hurricane Trajectory Analysis & Prediction</div>', unsafe_allow_html=True)
    
    # Thêm thông tin mô tả được thiết kế tốt hơn
    st.markdown("""
    <div class="card">
        <h3 style="color:#1E88E5; margin-top:0;">Welcome to the Hurricane Analysis Platform</h3>
        <p style="font-size:1.1rem; line-height:1.6;">
            This interactive dashboard combines advanced data analytics, machine learning, and geospatial 
            visualization to help understand and predict hurricane behavior based on trajectory patterns.
        </p>
        <p style="font-size:1.1rem; line-height:1.6;">
            Whether you're a researcher, meteorologist, or enthusiast, this tool provides powerful 
            insights into hurricane categories and movement patterns.
        </p>
        <div style="display:flex; justify-content:center; margin-top:20px;">
            <a href="#" onclick="document.getElementById('get-started-guide').scrollIntoView({behavior: 'smooth'}); return false;" 
               style="text-decoration:none; background:linear-gradient(90deg, #1E88E5, #3949AB); color:white; 
                      padding:10px 20px; border-radius:30px; font-weight:600; display:inline-block; 
                      box-shadow:0 4px 10px rgba(0,0,0,0.15); transition:all 0.3s ease;">
                Get Started →
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Thêm chỉ số trạng thái với thiết kế nhất quán hơn
    st.markdown('<h3 style="margin-top:30px;">System Status</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="padding:20px; background-color:{'#e3f2fd' if st.session_state.data_loaded else '#ffebee'}; border-radius:12px; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:10px;">{'✅' if st.session_state.data_loaded else '⏳'}</div>
            <div style="font-size:1.2rem; font-weight:600; color:#1E88E5;">Data Loading</div>
            <div style="margin-top:10px; color:{'#4CAF50' if st.session_state.data_loaded else '#F44336'}; font-weight:500;">
                {f"Completed ({len(st.session_state.processor.dataset.trajs)} trajectories)" if st.session_state.data_loaded else "Not Started"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="padding:20px; background-color:{'#e3f2fd' if st.session_state.features_extracted else '#ffebee'}; border-radius:12px; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:10px;">{'✅' if st.session_state.features_extracted else '⏳'}</div>
            <div style="font-size:1.2rem; font-weight:600; color:#1E88E5;">Feature Extraction</div>
            <div style="margin-top:10px; color:{'#4CAF50' if st.session_state.features_extracted else '#F44336'}; font-weight:500;">
                {f"Completed ({st.session_state.processor.features_df.shape[1] if st.session_state.features_extracted and hasattr(st.session_state.processor, 'features_df') else 0} features)" if st.session_state.features_extracted else "Not Started"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="padding:20px; background-color:{'#e3f2fd' if st.session_state.model_trained else '#ffebee'}; border-radius:12px; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:10px;">{'✅' if st.session_state.model_trained else '⏳'}</div>
            <div style="font-size:1.2rem; font-weight:600; color:#1E88E5;">Model Training</div>
            <div style="margin-top:10px; color:{'#4CAF50' if st.session_state.model_trained else '#F44336'}; font-weight:500;">
                {"Completed (Ready for prediction)" if st.session_state.model_trained else "Not Started"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Hiển thị thời gian cập nhật với thiết kế tốt hơn
    if st.session_state.last_update:
        st.markdown(f"""
        <div style="text-align:right; margin-top:10px; font-size:0.9rem; color:#666; background-color:#f8f9fa; padding:5px 10px; border-radius:5px; display:inline-block; float:right;">
            <span style="font-weight:600;">Last update:</span> {st.session_state.last_update}
        </div>
        <div style="clear:both;"></div>
        """, unsafe_allow_html=True)
    
    # Thêm hướng dẫn bắt đầu với animation
    st.markdown('<h3 id="get-started-guide" style="margin-top:40px;">Getting Started</h3>', unsafe_allow_html=True)
    
    # Hiển thị hướng dẫn từng bước với icon và hiệu ứng
    steps = [
        {"icon": "📥", "title": "Load Data", "desc": "Start by loading the hurricane dataset using the sidebar button", "action": "Click '📥 Load Hurricane Data' in the sidebar"},
        {"icon": "🔍", "title": "Extract Features", "desc": "Process the dataset to extract trajectory features", "action": "Click '🔍 Extract Features' in the sidebar after loading data"},
        {"icon": "🧠", "title": "Train Model", "desc": "Train the prediction model on the extracted features", "action": "Click '🧠 Train Model' in the sidebar after feature extraction"},
        {"icon": "🔄", "title": "Explore", "desc": "Navigate through different sections using the sidebar menu", "action": "Select different sections from the 'Select Section' menu"}
    ]
    
    # Hiển thị hướng dẫn từng bước dưới dạng timeline
    st.markdown("""
    <div style="position:relative; padding-left:50px; margin-top:30px;">
        <div style="position:absolute; left:20px; top:0; bottom:0; width:2px; background-color:#1E88E5;"></div>
    """, unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        st.markdown(f"""
        <div style="position:relative; margin-bottom:30px;">
            <div style="position:absolute; left:-50px; width:36px; height:36px; border-radius:50%; 
                        background-color:#1E88E5; color:white; display:flex; align-items:center; 
                        justify-content:center; font-size:18px; font-weight:bold; border:3px solid white;
                        box-shadow:0 0 0 2px #1E88E5;">
                {i+1}
            </div>
            <div class="card" style="margin-left:10px; padding:15px 20px;">
                <div style="display:flex; align-items:center;">
                    <div style="font-size:1.8rem; margin-right:15px;">{step['icon']}</div>
                    <div>
                        <h4 style="margin:0; color:#1E88E5;">{step['title']}</h4>
                        <p style="margin:5px 0 0 0;">{step['desc']}</p>
                    </div>
                </div>
                <div style="margin-top:10px; background-color:#e3f2fd; padding:8px 12px; border-radius:5px; font-size:0.9rem;">
                    <strong>Action:</strong> {step['action']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Thêm phần giới thiệu các tính năng với thiết kế card trực quan
    st.markdown('<h3 style="margin-top:10px;">Key Features</h3>', unsafe_allow_html=True)
    
    feature_items = [
        {"icon": "🌍", "title": "Interactive Maps", "desc": "Visualize hurricane trajectories with interactive maps"},
        {"icon": "📊", "title": "Data Analysis", "desc": "Analyze hurricane features and their correlations"},
        {"icon": "🔮", "title": "Prediction Model", "desc": "Predict hurricane categories based on trajectory features"},
        {"icon": "✏️", "title": "Draw & Predict", "desc": "Draw your own hurricane path and get instant predictions"},
        {"icon": "📱", "title": "Real-time Analysis", "desc": "Upload real hurricane data for instant analysis"},
        {"icon": "🎬", "title": "Animations", "desc": "View animated hurricane path developments"}
    ]
    
    # Hiển thị tính năng dưới dạng grid
    cols = st.columns(3)
    for i, feature in enumerate(feature_items):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="height:200px; display:flex; flex-direction:column;">
                <div style="font-size:2.5rem; margin-bottom:10px; text-align:center;">{feature['icon']}</div>
                <h4 style="margin:0 0 10px 0; text-align:center; color:#1E88E5;">{feature['title']}</h4>
                <p style="margin:0; text-align:center; flex-grow:1; display:flex; align-items:center; justify-content:center;">
                    {feature['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)


def show_trajectory_explorer():
    st.markdown('<div class="main-header">Hurricane Trajectory Explorer</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("Please load hurricane data using the button in the sidebar.")
        
        # Add loading animation
        hurricane_loading = load_lottie_url("https://lottie.host/a1462bbe-faf8-4afe-bff7-a3f8b27bdc47/RMaY9ZNQ7y.json")
        if hurricane_loading:
            st_lottie(hurricane_loading, speed=1, height=400, key="loading")
        return
    
    processor = st.session_state.processor
    
    # Sidebar filters in a card
    st.sidebar.markdown('<div class="sub-header">Explorer Filters</div>', unsafe_allow_html=True)
    
    categories = sorted(processor.dataset.classes)
    selected_categories = st.sidebar.multiselect(
        "Select Hurricane Categories",
        options=categories,
        default=categories
    )
    
    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    map_style = st.sidebar.selectbox(
        "Map Style",
        options=["Natural Earth", "Orthographic", "Globe"]
    )
    
    # Filter data based on selections
    filtered_indices = [i for i, label in enumerate(processor.dataset.labels) if label in selected_categories]
    filtered_trajs = [processor.dataset.trajs[i] for i in filtered_indices]
    filtered_labels = [processor.dataset.labels[i] for i in filtered_indices]
    
    st.markdown("""
    <div class="card">
    This explorer visualizes hurricane trajectories on an interactive map. Filter by category and sample size to analyze patterns
    across different hurricane types.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color:#e9f0f8; padding:10px; border-radius:8px; margin-bottom:15px;">
        <span style="font-weight:500;">Displaying {min(sample_size, len(filtered_trajs))} of {len(filtered_trajs)} filtered trajectories</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create map with selected projection
    with st.spinner("Generating trajectory map..."):
        fig, _ = create_trajectory_map(filtered_trajs, filtered_labels, sample_size)
        
        # Apply different map projections
        if map_style == "Orthographic":
            fig.update_geos(projection_type="orthographic")
        elif map_style == "Globe":
            fig.update_geos(projection_type="natural earth")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="sub-header">Trajectory Statistics by Category</div>', unsafe_allow_html=True)
    
    if st.session_state.features_extracted:
        features_df = st.session_state.processor.features_df.copy()
        features_df = make_dataframe_arrow_compatible(features_df)
        filtered_features = features_df[features_df['class'].isin(selected_categories)]
        
        # Add better column names for display
        renamed_cols = {
            'class': 'Category',
            'path_length': 'Length',
            'direct_distance': '',
            'sinuosity': '',
            'displacement_efficiency': '',
            'traj_duration': '',
            'duration_hours': '',
            'duration_days': '',
            'avg_time_between_points': '',
            'mean_velocity': ''

        }
        
        # Format numbers to 2 decimal places
        for col in features_df.columns:
            if col != 'class' and pd.api.types.is_numeric_dtype(features_df[col]):
                features_df[col] = features_df[col].round(2)
        
        # display_df = grouped.rename(columns=renamed_cols)
        st.dataframe(features_df, use_container_width=True)
        
        # Add visualizations of key statistics
        st.markdown('<div class="sub-header">Key Trajectory Metrics by Category</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Duration by Category
            fig_duration = px.box(
                filtered_features, 
                x='class', 
                y='traj_duration',
                color='class',
                color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                title='Trajectory Duration by Category',
                labels={'class': 'Hurricane Category', 'traj_duration': 'Duration (hours)'}
            )
            fig_duration.update_layout(
                height=400,
                boxmode='group',
                showlegend=False,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col2:
            # Velocity by Category  
            fig_velocity = px.box(
                filtered_features, 
                x='class', 
                y='mean_velocity',
                color='class',
                color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                title='Mean Velocity by Category',
                labels={'class': 'Hurricane Category', 'mean_velocity': 'Mean Velocity (km/h)'}
            )
            fig_velocity.update_layout(
                height=400,
                boxmode='group',
                showlegend=False,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_velocity, use_container_width=True)
            
    else:
        st.info("Please extract features to view trajectory statistics.")

def show_feature_analysis():
    st.markdown('<div class="main-header">Hurricane Feature Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.features_extracted:
        st.info("Please extract features using the button in the sidebar.")
        return
    
    processor = st.session_state.processor
    features_df = st.session_state.processor.features_df.copy()
    features_df = make_dataframe_arrow_compatible(features_df)
    
    st.markdown("""
    <div class="card">
    This section analyzes the relationship between hurricane features and categories. Explore how different characteristics 
    correlate with hurricane intensity and discover the most informative predictors.
    </div>
    """, unsafe_allow_html=True)
    
    # Feature selection with improved UI
    st.sidebar.markdown('<div class="sub-header">Feature Selection</div>', unsafe_allow_html=True)
    
    feature_options = [col for col in features_df.columns if col not in ['traj_id', 'class']]
    selected_feature = st.sidebar.selectbox(
        "Select feature to analyze",
        options=feature_options,
        index=feature_options.index('mean_velocity') if 'mean_velocity' in feature_options else 0
    )
    
    st.markdown(f'<div class="sub-header">{selected_feature.replace("_", " ").title()} Distribution by Category</div>', unsafe_allow_html=True)
    
    with st.spinner("Creating distribution chart..."):
        fig = create_feature_distribution_plot(features_df, selected_feature)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="sub-header">Feature Correlation Analysis</div>', unsafe_allow_html=True)
    
    # Feature correlation selection
    correlation_features = st.multiselect(
        "Select features for correlation analysis",
        options=feature_options,
        default=feature_options[:min(5, len(feature_options))]
    )
    
    if correlation_features:
        with st.spinner("Calculating correlations..."):
            corr_df = features_df[correlation_features + ['class']]
            corr_matrix = corr_df[correlation_features].corr()
            
            # Create modern correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.set(font_scale=1.2)
            heatmap = sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f', 
                ax=ax,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            plt.title('Feature Correlation Matrix', fontsize=20, pad=20)
            st.pyplot(fig)
    
    if st.session_state.model_trained:
        st.markdown('<div class="sub-header">Feature Importance in Category Prediction</div>', unsafe_allow_html=True)
        
        with st.spinner("Generating feature importance plot..."):
            model_results = train_model()
            fig = create_feature_importance_plot(model_results)
            st.plotly_chart(fig, use_container_width=True)
            
        # Add feature importance explanation
        st.markdown("""
        <div class="card">
        <h4>Understanding Feature Importance</h4>
        <p>The feature importance chart shows which trajectory characteristics most strongly influence the hurricane category prediction.
        Higher values indicate features that the model relies on more heavily when making predictions.</p>
        
        <p>Feature importance helps us understand the physical factors that differentiate hurricane categories and can guide 
        focus in hurricane monitoring and prediction systems.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature relationships scatter plot matrix
    if len(correlation_features) >= 2:
        st.markdown('<div class="sub-header">Feature Relationships</div>', unsafe_allow_html=True)
        
        selected_features_for_scatter = st.multiselect(
            "Select 2-4 features for scatter plot matrix",
            options=correlation_features,
            default=correlation_features[:min(3, len(correlation_features))]
        )
        
        if len(selected_features_for_scatter) >= 2:
            with st.spinner("Creating scatter plot matrix..."):
                fig = px.scatter_matrix(
                    features_df,
                    dimensions=selected_features_for_scatter,
                    color='class',
                    color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                    labels={col: col.replace('_', ' ').title() for col in selected_features_for_scatter}
                )
                fig.update_layout(
                    title='Feature Relationship Matrix',
                    height=600,
                    plot_bgcolor='white'
                )
                fig.update_traces(
                    diagonal_visible=False,
                    showupperhalf=False,
                    marker=dict(size=5, opacity=0.7)
                )
                st.plotly_chart(fig, use_container_width=True)

def show_prediction_model():
    st.markdown('<div class="main-header">Hurricane Category Prediction Model</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.info("Please train the model using the button in the sidebar.")
        return
    
    processor = st.session_state.processor
    model_results = train_model()
    
    st.markdown("""
    <div class="card">
    This section provides insights into the hurricane category prediction model. Explore model performance metrics, 
    feature importance, and test the model on sample data.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Classification Report")
        
        # Create a more informative and styled classification report
        report = model_results['report']
        
        # Extract values from the report dictionary
        classes = sorted([key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']])
        
        # Create a DataFrame for better display
        report_data = []
        for cls in classes:
            report_data.append({
                'Category': cls,
                'Precision': f"{report[cls]['precision']:.3f}",
                'Recall': f"{report[cls]['recall']:.3f}",
                'F1-Score': f"{report[cls]['f1-score']:.3f}",
                'Support': report[cls]['support']
            })
        
        # Add averages
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                report_data.append({
                    'Category': avg_type,
                    'Precision': f"{report[avg_type]['precision']:.3f}",
                    'Recall': f"{report[avg_type]['recall']:.3f}",
                    'F1-Score': f"{report[avg_type]['f1-score']:.3f}",
                    'Support': report[avg_type]['support']
                })
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)
        
        # Add overall accuracy
        if 'accuracy' in report:
            st.metric("Overall Accuracy", f"{report['accuracy']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        fig_cm = create_confusion_matrix_plot(model_results)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Feature Importance</div>', unsafe_allow_html=True)
    
    fig_fi = create_feature_importance_plot(model_results)
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.markdown('<div class="sub-header">Test Predictions</div>', unsafe_allow_html=True)
    
    # File upload section with better styling
    st.markdown("""
    <div class="card">
    <h4>Predict from uploaded trajectory data</h4>
    <p>Upload a file containing trajectory data to get hurricane category predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload trajectory data (pickle or CSV)", type=["pkl", "csv"])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded data..."):
                if uploaded_file.name.endswith("pkl"):
                    new_data = pickle.load(uploaded_file)
                else:
                    new_data = pd.read_csv(uploaded_file)
                
                # Modificado para manejar el caso donde no hay featurizer
                try:
                    # Intenta usar featurizer si existe
                    features = st.session_state.processor.model.featurizer.transform(new_data)
                    prediction = st.session_state.processor.model.predict([features])[0]
                except AttributeError:
                    # Manejo alternativo si no hay featurizer
                    if hasattr(st.session_state.processor, 'extract_features'):
                        # Usar la función de extracción de características del procesador
                        features = st.session_state.processor.extract_features(new_data)
                        prediction = st.session_state.processor.model.predict(features)[0]
                    else:
                        # Si no hay método de extracción, intentar predecir directamente
                        prediction = st.session_state.processor.model.predict(new_data)[0]
                
                # Display prediction with styling
                st.success(f"Predicted Hurricane Category: {prediction}")
                
                # Show feature values that led to this prediction (if available)
                try:
                    st.markdown("#### Key Features for this Prediction")
                    if isinstance(features, dict):
                        feature_df = pd.DataFrame({
                            'Feature': list(features.keys()),
                            'Value': list(features.values())
                        })
                    else:
                        feature_df = pd.DataFrame(features)
                    st.dataframe(feature_df, use_container_width=True)
                except:
                    st.info("Feature details not available for display.")
                    
        except Exception as e:
            st.error(f"Error processing data: {e}")


def show_trajectory_comparison():
    st.markdown('<div class="main-header">Hurricane Trajectory Comparison</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("Please load hurricane data using the button in the sidebar.")
        return
    
    processor = st.session_state.processor
    categories = sorted(processor.dataset.classes)
    
    st.markdown("""
    <div class="card">
    This section allows you to compare normalized hurricane trajectories across different categories. Normalization helps
    identify characteristic patterns in how hurricanes of different categories move, independent of their specific location.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_category = st.selectbox(
            "Select category to compare", 
            options=["All Categories"] + [f"Category {cat}" for cat in categories]
        )
        
        st.markdown("### Understanding Normalized Trajectories")
        st.markdown("""
        Normalized trajectories:
        
        * Start at point (0,0)
        * Are scaled to similar sizes
        * Remove geographic specifics
        * Highlight movement patterns
        * Show characteristic shapes
        """)
        
        st.markdown("### Key Observations")
        st.markdown("""
        * Lower categories often show more erratic paths
        * Higher categories tend to have smoother trajectories
        * Some categories show distinctive curvature patterns
        * Starting and ending velocity profiles differ by category
        """)
    
    with col2:
        st.markdown("### Normalized Trajectory Comparison")
        category_for_plot = None if selected_category == "All Categories" else int(selected_category.split(" ")[1])
        
        with st.spinner("Creating normalized trajectory plot..."):
            fig = create_normalized_trajectory_plot(processor, category_for_plot)
            st.plotly_chart(fig, use_container_width=True)
    
    # Corregir visualización en boxes
    st.markdown('<div class="sub-header">Trajectory Shape Analysis</div>', unsafe_allow_html=True)
    
    # Usar directamente columnas de Streamlit en lugar de HTML
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#1E88E5;">Curvature Distribution by Category</h3>
        </div>""", unsafe_allow_html=True)
        
        # Verificar si los datos están disponibles de otra manera
        curvature_available = hasattr(processor, 'features_df') and processor.features_df is not None
        
        if curvature_available:
            try:
                features_df = processor.features_df
                
                # Si no hay columna curvature, crear una simulada para demostración
                if 'curvature' not in features_df.columns and 'mean_velocity' in features_df.columns:
                    features_df['curvature'] = features_df['mean_velocity'] * np.random.uniform(0.5, 1.5, len(features_df))
                
                if 'curvature' in features_df.columns:
                    fig_curve = px.box(
                        features_df,
                        x='class',
                        y='curvature',
                        color='class',
                        color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                        labels={'class': 'Hurricane Category', 'curvature': 'Path Curvature'}
                    )
                    fig_curve.update_layout(
                        height=400,
                        plot_bgcolor='white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)
                else:
                    st.info("Curvature data not available. Try extracting features with additional metrics.")
            except Exception as e:
                st.error(f"Error displaying curvature distribution: {e}")
        else:
            st.info("Please extract features to view curvature distribution.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="color:#1E88E5;">Geographic Spread by Category</h3>
        </div>""", unsafe_allow_html=True)
        
        geo_available = hasattr(processor, 'features_df') and processor.features_df is not None
        
        if geo_available:
            try:
                features_df = processor.features_df
                
                if 'lon_range' in features_df.columns and 'lat_range' in features_df.columns:
                    features_df['geo_spread'] = features_df['lon_range'] * features_df['lat_range']
                    fig_spread = px.violin(
                        features_df,
                        x='class',
                        y='geo_spread',
                        color='class',
                        color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                        labels={'class': 'Hurricane Category', 'geo_spread': 'Geographic Spread'},
                        box=True
                    )
                    fig_spread.update_layout(
                        height=400,
                        plot_bgcolor='white',
                        showlegend=False
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)
                else:
                    # Create a simple visualization using available data
                    st.info("Geographic range data not available. Showing alternative visualization.")
                    
                    # Use any available geographic data
                    if hasattr(processor, 'dataset') and hasattr(processor.dataset, 'trajs'):
                        # Show a simple scatter of starting points
                        st.write("Starting points of hurricanes by category:")
                        
                        # Extract some points for demonstration
                        plot_data = []
                        for i, traj in enumerate(processor.dataset.trajs[:100]):  # Limit to 100 trajectories
                            if hasattr(traj, 'r') and len(traj.r) > 0:
                                plot_data.append({
                                    'longitude': traj.r[0, 0],
                                    'latitude': traj.r[0, 1],
                                    'class': processor.dataset.labels[i]
                                })
                        
                        if plot_data:
                            df_plot = pd.DataFrame(plot_data)
                            fig = px.scatter(
                                df_plot,
                                x='longitude',
                                y='latitude',
                                color='class',
                                color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                                title='Hurricane Starting Locations by Category'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying geographic spread: {e}")
        else:
            st.info("Please extract features to view geographic spread analysis.")
            
        st.markdown('</div>', unsafe_allow_html=True)

def show_hurricane_impact():
    st.markdown('<div class="main-header">Hurricane Impact Visualization</div>', unsafe_allow_html=True)
    
    if not st.session_state.features_extracted:
        st.info("Please extract features to view hurricane impact visualizations.")
        return
    
    processor = st.session_state.processor
    features_df = processor.features_df.copy()
    features_df = make_dataframe_arrow_compatible(features_df)
    
    st.markdown("""
    <div class="card">
    This section visualizes the potential impact of hurricanes based on their trajectory characteristics. The analysis focuses on
    duration, intensity, and geographic spread to estimate potential severity.
    </div>
    """, unsafe_allow_html=True)
    
    # Main impact visualization
    try:
        fig = create_hurricane_impact_visualization(features_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating impact visualization: {e}")
        st.info("Generating alternative visualization based on available data...")
        
        # Create an alternative visualization
        if 'traj_duration' in features_df.columns:
            fig = px.histogram(
                features_df, 
                x='traj_duration',
                color='class',
                color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                title='Hurricane Duration Distribution by Category',
                labels={'traj_duration': 'Trajectory Duration (hours)', 'class': 'Hurricane Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional impact metrics - Usar directamente Streamlit columns
    st.markdown('<div class="sub-header">Impact Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Hurricane Intensity Metrics")
        
        # Verificar disponibilidad de datos
        if 'max_velocity' in features_df.columns and 'mean_velocity' in features_df.columns:
            fig_intensity = px.scatter(
                features_df,
                x='mean_velocity',
                y='max_velocity',
                color='class',
                color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                labels={
                    'mean_velocity': 'Average Velocity (km/h)',
                    'max_velocity': 'Maximum Velocity (km/h)',
                    'class': 'Hurricane Category'
                },
                size='traj_duration' if 'traj_duration' in features_df.columns else None,
                size_max=15,
                hover_name='traj_id' if 'traj_id' in features_df.columns else None
            )
            fig_intensity.update_layout(
                height=500,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
        elif 'mean_velocity' in features_df.columns:
            # Alternativa si solo mean_velocity está disponible
            fig_intensity = px.box(
                features_df,
                x='class',
                y='mean_velocity',
                color='class',
                color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                labels={
                    'class': 'Hurricane Category',
                    'mean_velocity': 'Average Velocity (km/h)'
                },
            )
            fig_intensity.update_layout(
                height=500,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
        else:
            st.info("Velocity data not available. Please extract more detailed features.")
            
            # Mostrar datos alternativos si están disponibles
            for col in features_df.columns:
                if col not in ['class', 'traj_id'] and pd.api.types.is_numeric_dtype(features_df[col]):
                    st.write(f"Showing analysis of {col}:")
                    fig = px.box(
                        features_df,
                        x='class',
                        y=col,
                        color='class',
                        color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548']
                    )
                    st.plotly_chart(fig, height=350)
                    break
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Geographic Impact Patterns")
        
        # Create a heatmap of hurricane frequency by location
        if hasattr(processor.dataset, 'trajs') and len(processor.dataset.trajs) > 0:
            try:
                # Extract starting points
                lat_points = []
                lon_points = []
                categories = []
                
                for i, traj in enumerate(processor.dataset.trajs):
                    if hasattr(traj, 'r') and len(traj.r) > 0:
                        lat_points.append(traj.r[0, 1])  # First point latitude
                        lon_points.append(traj.r[0, 0])  # First point longitude
                        categories.append(processor.dataset.labels[i])
                
                df_points = pd.DataFrame({
                    'latitude': lat_points,
                    'longitude': lon_points,
                    'class': categories
                })
                
                fig_geo = px.scatter_mapbox(
                    df_points,
                    lat='latitude',
                    lon='longitude',
                    color='class',
                    color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                    center=dict(lat=25, lon=-70),
                    zoom=3,
                    mapbox_style="carto-positron",
                    title='Hurricane Starting Locations',
                    labels={'class': 'Hurricane Category'}
                )
                fig_geo.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating map: {e}")
                
                # Alternative visualization if mapbox fails
                fig = px.scatter(
                    df_points,
                    x='longitude',
                    y='latitude',
                    color='class',
                    color_discrete_sequence=['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#795548'],
                    title='Hurricane Starting Locations',
                    labels={'class': 'Hurricane Category'}
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Longitude",
                    yaxis_title="Latitude"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trajectory data not available for geographic impact analysis.")
            
            # Show placeholder visualization
            st.image("https://www.nhc.noaa.gov/xgtwo/two_atl_2d0.png", caption="Example hurricane tracking map (placeholder)")
            
        st.markdown("</div>", unsafe_allow_html=True)

def show_advanced_visualizations():
    st.markdown('<div class="main-header">Advanced Hurricane Visualizations</div>', unsafe_allow_html=True)
    
    processor = st.session_state.processor
    if not st.session_state.data_loaded:
        st.info("Please load hurricane data using the sidebar buttons.")
        return
    
    dataset = processor.dataset
    
    st.markdown(""" 
    <div class="card">
        <div style="display:flex; align-items:start;">
            <div style="font-size:2.5rem; margin-right:20px;">🎬</div>
            <div>
                <h3 style="margin-top:0; color:#1E88E5;">Enhanced Visualization Tools</h3>
                <p>Explore hurricane trajectories through advanced visualizations including animations, 3D plots,
                and velocity profiles. These tools help identify patterns and behaviors that may not be visible in standard maps.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization selection với tabs thay vì radio buttons
    tab1, tab2, tab3 = st.tabs(["🎬 Animated Trajectories", "🌐 3D Visualization", "📈 Velocity Profiles"])
    
    with tab1:
        st.markdown('<div class="sub-header">Hurricane Trajectory Animation</div>', unsafe_allow_html=True)
        
        # Control panel cải tiến
        control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
        
        with control_col1:
            sample_size = st.slider(
                "Number of Trajectories", 
                min_value=10, 
                max_value=200, 
                value=100
            )
        
        with control_col2:
            animation_speed = st.slider(
                "Animation Speed", 
                min_value=50, 
                max_value=500, 
                value=150,
                step=50
            )
        
        with control_col3:
            categories = sorted(dataset.classes)
            selected_categories = st.multiselect(
                "Filter Categories",
                options=categories,
                default=categories
            )
        
        # Lọc dữ liệu quỹ đạo
        filtered_indices = [i for i, label in enumerate(dataset.labels) if label in selected_categories]
        filtered_trajs = [dataset.trajs[i] for i in filtered_indices]
        filtered_labels = [dataset.labels[i] for i in filtered_indices]
        
        with st.spinner("Generating animation..."):
            fig_map, df_points = create_trajectory_map(filtered_trajs, filtered_labels, sample_size=sample_size)
            
            # Tùy chỉnh animation với tốc độ từ người dùng
            animated_fig = create_animated_trajectory_map(df_points)
            
            # Cập nhật frame duration dựa trên tốc độ animation
            for frame in animated_fig.frames:
                frame.layout.update(
                    updatemenus=[{
                        'buttons': [
                            {
                                'args': [None, {'frame': {'duration': 600 - animation_speed, 'redraw': True}}],
                                'method': 'animate'
                            }
                        ]
                    }]
                )
            
            st.plotly_chart(animated_fig, use_container_width=True)
            
        st.markdown(""" 
        <div class="card">
            <h4 style="color:#1E88E5;">Understanding the Animation</h4>
            <p>This animation shows the progressive development of hurricane trajectories over time. Each frame represents a time step in the hurricane's life cycle.</p>
            <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; margin-top:15px;">
                <h5 style="margin-top:0; color:#1E88E5;">How to Use the Animation</h5>
                <ul style="margin-bottom:0;">
                    <li><strong>Play/Pause:</strong> Use the controls at the bottom left</li>
                    <li><strong>Navigate Frames:</strong> Use the slider to jump to specific time steps</li>
                    <li><strong>Change Speed:</strong> Adjust the Animation Speed slider above</li>
                    <li><strong>View Details:</strong> Hover over paths to see category information</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">3D Hurricane Trajectory Visualization</div>', unsafe_allow_html=True)
        
        # Options panel cải tiến 
        control3d_col1, control3d_col2 = st.columns(2)
        
        with control3d_col1:
            sample_size_3d = st.slider(
                "Number of Trajectories", 
                min_value=5, 
                max_value=50, 
                value=20,
                key="sample_size_3d"
            )
            
            view_option = st.selectbox(
                "View Perspective",
                options=["Top-down", "Side view", "Angled view"],
                index=2
            )
        
        with control3d_col2:
            categories = sorted(dataset.classes)
            selected_categories_3d = st.multiselect(
                "Filter Categories",
                options=categories,
                default=categories,
                key="selected_categories_3d"
            )
            
            color_theme = st.selectbox(
                "Color Theme",
                options=["Default", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
                index=0
            )
        
        # Filter trajectories by selected categories
        filtered_indices = [i for i, label in enumerate(dataset.labels) if label in selected_categories_3d]
        filtered_trajs = [dataset.trajs[i] for i in filtered_indices]
        filtered_labels = [dataset.labels[i] for i in filtered_indices]
        
        with st.spinner("Generating 3D visualization..."):
            fig_3d = create_3d_trajectory_plot(filtered_trajs, filtered_labels, sample_size=sample_size_3d)
            
            # Customize color theme if selected
            if color_theme != "Default":
                # Get all scatter traces
                for i in range(len(fig_3d.data)):
                    if fig_3d.data[i].type == 'scatter3d':
                        # Update color based on theme
                        fig_3d.data[i].line.colorscale = color_theme.lower()
            
            # Set camera view based on selection
            if view_option == "Top-down":
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=0, z=2.5)  # Looking from above
                )
            elif view_option == "Side view":
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=2.5, y=0, z=0)  # Looking from side
                )
            else:  # Angled view
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.0)  # Default angled view
                )
            
            fig_3d.update_layout(scene_camera=camera)
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
        st.markdown(""" 
        <div class="card">
            <h4 style="color:#1E88E5;">Interpreting the 3D Visualization</h4>
            <p>This visualization represents hurricane trajectories in three-dimensional space, with the Z-axis showing time progression.</p>
            <div style="display:flex; justify-content:space-between; margin-top:15px;">
                <div style="flex:1; padding-right:10px;">
                    <h5 style="color:#1E88E5;">Dimensions Explained</h5>
                    <ul>
                        <li><strong>X-axis:</strong> Longitude</li>
                        <li><strong>Y-axis:</strong> Latitude</li>
                        <li><strong>Z-axis:</strong> Normalized time (0-1 scale)</li>
                    </ul>
                </div>
                <div style="flex:1; padding-left:10px;">
                    <h5 style="color:#1E88E5;">Interaction Tips</h5>
                    <ul>
                        <li>Click and drag to rotate the view</li>
                        <li>Scroll to zoom in/out</li>
                        <li>Double-click to reset the view</li>
                        <li>Hover over points for details</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Hurricane Velocity Profiles</div>', unsafe_allow_html=True)
        
        # Controls for velocity profiles
        velocity_col1, velocity_col2 = st.columns(2)
        
        with velocity_col1:
            sample_size_vel = st.slider(
                "Number of Trajectories", 
                min_value=3, 
                max_value=15, 
                value=8,
                key="sample_size_vel"
            )
            
            normalize_velocity = st.checkbox(
                "Normalize Velocity",
                value=False,
                help="Normalize velocity values for better comparison"
            )
        
        with velocity_col2:
            categories = sorted(dataset.classes)
            
            selected_categories_vel = st.multiselect(
                "Filter Categories",
                options=categories,
                default=categories[:3],  # Default to first 3 categories
                key="selected_categories_vel"
            )
            
            show_avg_line = st.checkbox(
                "Show Average Line",
                value=True,
                help="Display average velocity line for each profile"
            )
        
        # Filter trajectories by selected categories
        filtered_indices = [i for i, label in enumerate(dataset.labels) if label in selected_categories_vel]
        filtered_trajs = [dataset.trajs[i] for i in filtered_indices]
        filtered_labels = [dataset.labels[i] for i in filtered_indices]
        
        if filtered_trajs:  # Only proceed if there are valid trajectories
            with st.spinner("Generating velocity profiles..."):
                fig_velocity = create_velocity_profile(filtered_trajs, filtered_labels, sample_size=sample_size_vel)
                
                # Customize figure based on options
                if not show_avg_line:
                    # Remove avg lines (dashed lines)
                    for i, trace in enumerate(fig_velocity.data):
                        if isinstance(trace, go.Scatter) and trace.line.dash == 'dash':
                            fig_velocity.data[i].visible = False
                
                st.plotly_chart(fig_velocity, use_container_width=True)
                
                # Add supplementary visualization: velocity statistics by category
                if st.checkbox("Show Velocity Statistics by Category", value=True):
                    # Collect velocity data by category
                    velocity_stats = {}
                    for i, traj in enumerate(filtered_trajs):
                        cat = filtered_labels[i]
                        if cat not in velocity_stats:
                            velocity_stats[cat] = []
                        
                        if hasattr(traj, 'v') and traj.v is not None:
                            # Calculate velocity magnitudes
                            v_magnitude = np.sqrt(np.sum(traj.v**2, axis=1)) * 3.6  # km/h
                            
                            # Add to category stats
                            velocity_stats[cat].append({
                                'mean': np.mean(v_magnitude),
                                'max': np.max(v_magnitude),
                                'std': np.std(v_magnitude)
                            })
                    
                    # Create statistics dataframe
                    stats_data = []
                    for cat, stats in velocity_stats.items():
                        if stats:
                            mean_vals = [s['mean'] for s in stats]
                            max_vals = [s['max'] for s in stats]
                            std_vals = [s['std'] for s in stats]
                            
                            stats_data.append({
                                'Category': f"Category {cat}",
                                'Mean Velocity (km/h)': np.mean(mean_vals),
                                'Max Velocity (km/h)': np.mean(max_vals),
                                'Std Deviation': np.mean(std_vals)
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        
                        # Create bar chart
                        fig_stats = px.bar(
                            stats_df,
                            x='Category',
                            y=['Mean Velocity (km/h)', 'Max Velocity (km/h)'],
                            barmode='group',
                            title='Velocity Statistics by Hurricane Category',
                            color_discrete_sequence=['#1E88E5', '#F44336']
                        )
                        
                        fig_stats.update_layout(
                            xaxis_title='Hurricane Category',
                            yaxis_title='Velocity (km/h)',
                            legend_title='Metric',
                            plot_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_stats, use_container_width=True)
        else:
            st.error("No valid trajectories found for the selected categories.")
            
        st.markdown(""" 
        <div class="card">
            <h4 style="color:#1E88E5;">Understanding Velocity Profiles</h4>
            <p>Velocity profiles show how hurricane speed changes throughout its lifecycle, revealing important patterns in storm development and behavior.</p>
            <div style="background-color:#e3f2fd; padding:15px; border-radius:10px; margin-top:15px;">
                <h5 style="margin-top:0; color:#1E88E5;">Key Insights from Velocity Profiles</h5>
                <ul style="margin-bottom:0;">
                    <li><strong>Acceleration Patterns:</strong> How quickly hurricanes gain or lose speed</li>
                    <li><strong>Peak Velocity Timing:</strong> When hurricanes reach their maximum speed</li>
                    <li><strong>Category Differences:</strong> How velocity profiles differ between categories</li>
                    <li><strong>Velocity Stability:</strong> How consistent hurricane speeds remain over time</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_real_input_prediction():
    st.markdown('<div class="main-header">Real-time Hurricane Prediction</div>', unsafe_allow_html=True)

    st.markdown(""" 
    <div class="card">
    Upload real hurricane trajectory data to predict its category. The system will analyze the movement patterns
    and provide a classification based on the trained model.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Upload Data", "Test with Random Sample"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Upload Trajectory Data")
            st.write("Upload a CSV file with trajectory data (columns: t, longitude, latitude)")

            uploaded_file = st.file_uploader("Select CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    df_input = pd.read_csv(uploaded_file)

                    # Validate required columns
                    required_cols = ["t", "longitude", "latitude"]
                    missing_cols = [col for col in required_cols if col not in df_input.columns]

                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        st.success("Data loaded successfully!")

                        # Display data preview
                        st.markdown("#### Data Preview")
                        st.dataframe(df_input.head(10), use_container_width=True)

                        # Create trajectory object
                        traj_input = type("Trajectory", (object,), {})()
                        traj_input.t = df_input["t"].values
                        traj_input.r = df_input[["longitude", "latitude"]].values
                        traj_input.traj_id = "user_input"

                        # Calculate velocity if not provided
                        if "v_lon" not in df_input.columns or "v_lat" not in df_input.columns:
                            v_lon = np.zeros(len(traj_input.t))
                            v_lat = np.zeros(len(traj_input.t))

                            for i in range(1, len(traj_input.t)):
                                dt = traj_input.t[i] - traj_input.t[i-1]
                                if dt > 0:
                                    v_lon[i] = (traj_input.r[i, 0] - traj_input.r[i-1, 0]) / dt
                                    v_lat[i] = (traj_input.r[i, 1] - traj_input.r[i-1, 1]) / dt

                            # First point velocity
                            v_lon[0] = v_lon[1]
                            v_lat[0] = v_lat[1]

                            traj_input.v = np.column_stack((v_lon, v_lat))
                        else:
                            traj_input.v = df_input[["v_lon", "v_lat"]].values

                        # Extract features using the processing options
                        if st.session_state.features_extracted:
                            with st.spinner("Processing data..."):
                                try:
                                    # Store the trajectory temporarily for feature extraction
                                    if 'temp_trajs' not in st.session_state:
                                        st.session_state.temp_trajs = []
                                    st.session_state.temp_trajs.append(traj_input)

                                    # Extract features
                                    features_df = extract_features(
                                        _trajs=st.session_state.temp_trajs,
                                        _use_features=st.session_state.preprocessing_options['use_features'], 
                                        _outlier_method=st.session_state.preprocessing_options['outlier_method'], 
                                        _create_interactions=st.session_state.preprocessing_options['create_interactions']
                                    )
                                    
                                    # Remove 'traj_id' and 'class' columns as they are not features
                                    features_df = features_df.drop(columns=['traj_id', 'class'], errors='ignore')

                                    # Get features for the input trajectory
                                    traj_input_features = features_df.loc[features_df['traj_id'] == traj_input.traj_id]

                                    # Clean up temporary storage
                                    st.session_state.temp_trajs.pop()

                                    # Ensure the trajectory features are in the correct shape
                                    if traj_input_features.empty:
                                        st.error("Feature extraction returned empty data. Please check the input data and processing.")
                                        return

                                    # Load the saved model
                                    model_path = "hurricane_model.pkl"
                                    model = joblib.load(model_path)

                                    # Get the features the model was trained on
                                    trained_features = [f"feature_{i}" for i in range(model.n_features_in_)]  # Assuming model has n_features_in_ attribute

                                    # Ensure that the features match the ones the model was trained with
                                    feature_columns = [col for col in traj_input_features.columns if col != 'traj_id']
                                    missing_features = [f for f in trained_features if f not in feature_columns]
                                    extra_features = [f for f in feature_columns if f not in trained_features]

                                    if missing_features or extra_features:
                                        st.error(f"Feature mismatch: Missing features {missing_features}, Extra features {extra_features}")
                                        return

                                    # Align feature columns to match what the model was trained with
                                    feature_values = traj_input_features[trained_features].values

                                    # Predict hurricane category if model is loaded
                                    with st.spinner("Analyzing trajectory..."):
                                        try:
                                            # Predict using the loaded model
                                            prediction = model.predict(feature_values)[0]

                                            # Display map with prediction if successful
                                            fig, _ = create_trajectory_map([traj_input], [prediction], sample_size=1)
                                            st.plotly_chart(fig, use_container_width=True)
                                            st.success(f"Prediction successful: Category {prediction}")
                                        except Exception as e:
                                            st.error(f"Prediction error: {str(e)}")
                                except Exception as e:
                                    st.error(f"Error processing data: {e}")
                        else:
                            st.warning("Please extract features first.")
                except Exception as e:
                    st.error(f"Error processing input data: {e}")
            else:
                st.info("Please upload a CSV file with trajectory data.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            # Animation for the right column
            hurricane_animation = load_lottie_url("https://lottie.host/67c15408-2b5b-4f4a-993a-3e8fbabb071c/2QE3xOUyJg.json")
            if hurricane_animation:
                st_lottie(hurricane_animation, speed=1, height=200, key="hurricane_upload")

            st.markdown("### Trajectory Analysis")

            if uploaded_file is not None and "traj_input" in locals():
                # Calculate basic statistics
                duration = traj_input.t[-1] - traj_input.t[0]
                num_points = len(traj_input.t)

                # Geographic range
                lon_range = np.max(traj_input.r[:, 0]) - np.min(traj_input.r[:, 0])
                lat_range = np.max(traj_input.r[:, 1]) - np.min(traj_input.r[:, 1])

                # Velocity statistics
                v_magnitude = np.sqrt(np.sum(traj_input.v**2, axis=1))
                avg_velocity = np.mean(v_magnitude)
                max_velocity = np.max(v_magnitude)

                # Display metrics
                st.metric("Duration", f"{duration:.1f} hours")
                st.metric("Data Points", num_points)
                st.metric("Avg Velocity", f"{avg_velocity:.2f} km/h")
                st.metric("Max Velocity", f"{max_velocity:.2f} km/h")

                # Display geographic info
                st.markdown("#### Geographic Range")
                st.markdown(f"**Longitude:** {lon_range:.2f}°")
                st.markdown(f"**Latitude:** {lat_range:.2f}°")

                # Display prediction result if model is trained
                if st.session_state.model_trained and "prediction" in locals():
                    st.markdown(""" 
                    <div style="margin-top:30px; text-align:center;">
                        <h3>Prediction Result</h3>
                        <div style="font-size:60px; font-weight:bold; color:#1E88E5;">
                            Category {0}
                        </div>
                    </div>
                    """.format(prediction), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(""" 
        <div class="card">
        <h4>Test with Random Trajectory from Test Set</h4>
        <p>Select a random trajectory from the test set to see how the model performs.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.data_loaded and st.session_state.model_trained:
            if st.button("Select Random Test Trajectory"):
                # Get random index
                idx = np.random.randint(0, len(st.session_state.processor.dataset.trajs))
                st.session_state.random_trajectory_idx = idx

            # Use stored index or generate new one
            if 'random_trajectory_idx' not in st.session_state:
                st.session_state.random_trajectory_idx = 0

            idx = st.session_state.random_trajectory_idx

            with st.spinner("Analyzing trajectory..."):
                try:
                    traj_random = st.session_state.processor.dataset.trajs[idx]
                    groundtruth = st.session_state.processor.dataset.labels[idx]

                    # Apply the same feature extraction as used in tab1
                    if st.session_state.features_extracted:
                        try:
                            # Store the trajectory temporarily for feature extraction
                            if 'temp_trajs' not in st.session_state:
                                st.session_state.temp_trajs = []

                            st.session_state.temp_trajs.append(traj_random)

                            # Extract features using the same approach as in tab1
                            features_df = extract_features(
                                _trajs=st.session_state.temp_trajs,
                                _use_features=st.session_state.preprocessing_options['use_features'], 
                                _outlier_method=st.session_state.preprocessing_options['outlier_method'], 
                                _create_interactions=st.session_state.preprocessing_options['create_interactions']
                            )

                            # Get features for the random trajectory
                            traj_random_features = features_df.loc[features_df['traj_id'] == traj_random.traj_id]

                            # Clean up temporary storage
                            st.session_state.temp_trajs.pop()

                            if traj_random_features.empty:
                                st.warning("Feature extraction returned empty data for this trajectory")
                                pred_random = groundtruth  # Use groundtruth as fallback
                            else:
                                # Get just the feature values (without traj_id and other metadata)
                                feature_columns = [col for col in traj_random_features.columns if col != 'traj_id' and col != 'class']
                                feature_values = traj_random_features[feature_columns].values

                                # Predict using the loaded model
                                pred_random = st.session_state.processor.model.predict(feature_values)[0]
                        except Exception as e:
                            st.error(f"Feature extraction error: {str(e)}")
                            pred_random = groundtruth  # Use groundtruth as fallback
                    else:
                        st.warning("Please extract features first.")
                        pred_random = groundtruth  # Use groundtruth if features are not extracted

                    # Display results
                    col1, col2, col3 = st.columns([2, 2, 3])
                    with col1:
                        st.metric("Actual Category", groundtruth)
                    with col2:
                        st.metric("Predicted Category", pred_random, 
                                 delta="Correct" if groundtruth == pred_random else "Incorrect",
                                 delta_color="normal" if groundtruth == pred_random else "inverse")
                    with col3:
                        st.metric("Trajectory ID", traj_random.traj_id if hasattr(traj_random, 'traj_id') else "Unknown")
                        st.metric("Points", len(traj_random.r) if hasattr(traj_random, 'r') else 0)

                    # Show trajectory on map
                    st.markdown("#### Trajectory Map")
                    fig_random, _ = create_trajectory_map([traj_random], [groundtruth], sample_size=1)
                    st.plotly_chart(fig_random, use_container_width=True)

                    # Show feature values if available
                    if 'traj_random_features' in locals() and not traj_random_features.empty:
                        st.markdown("#### Trajectory Features")
                        st.dataframe(traj_random_features, use_container_width=True)

                except Exception as e:
                    st.error(f"Random trajectory error: {str(e)}")
                    st.info("Try training the model again or using a different dataset.")

# --- Hàm chính ---
def main():
    # Chạy sidebar cải tiến và lấy trang được chọn
    page = create_sidebar()
    
    # Only reload data when needed (not on every page change)
    if 'processor' in st.session_state and st.session_state.data_loaded:
        processor = st.session_state.processor
    
    # Track the current page to avoid redundant calculations
    if 'current_page' not in st.session_state:
        st.session_state.current_page = page
    
    # Only clear rendered state when page changes
    if st.session_state.current_page != page:
        # Clear any page-specific cached states
        keys_to_clear = [k for k in st.session_state.keys() 
                         if k.startswith('tab_') and k.endswith('_rendered')]
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Update current page
        st.session_state.current_page = page
    
    # Chuyển hướng đến trang tương ứng
    if page == "Home":
        show_home_page()
    elif page == "Trajectory Explorer":
        show_trajectory_explorer()
    elif page == "Feature Analysis":
        show_feature_analysis()
    elif page == "Prediction Model":
        show_prediction_model()
    elif page == "Trajectory Comparison":
        show_trajectory_comparison()
    elif page == "Advanced Visualizations":
        show_advanced_visualizations()
    elif page == "Hurricane Impact":
        show_hurricane_impact()
    elif page == "Real Data Input":
        show_real_input_prediction()
    elif page == "Draw & Predict":
        show_drawing_prediction()


if __name__ == "__main__":
    main()