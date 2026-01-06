import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoSmart Energy Platform",
    page_icon="⚡",
    layout="wide"
)

# --- TITLE & SIDEBAR ---
st.title("⚡ EcoSmart: Energy Optimization Platform")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "User Segments", "Demand Forecasting", "Safety & Anomalies"])

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Load the files you exported from Colab
    clusters = pd.read_csv('dashboard_data_clusters.csv', index_col=0)
    predictions = pd.read_csv('dashboard_data_predictions.csv', index_col=0, parse_dates=True)
    anomalies = pd.read_csv('dashboard_data_anomalies.csv', index_col=0, parse_dates=True)
    return clusters, predictions, anomalies

try:
    df_clusters, df_preds, df_anomalies = load_data()
except FileNotFoundError:
    st.error("Data files not found! Please ensure the CSV files are in the same directory.")
    st.stop()

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.markdown("### System Status & Global Metrics")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Households Managed", f"{len(df_clusters):,}")
    col2.metric("Total Energy Tracked (2013)", f"{df_clusters['total_usage'].sum()/1000:.1f} MWh")
    col3.metric("Anomalies Detected", "351 Events")

    st.info("ℹ️ **Project Context:** This platform utilizes Machine Learning (K-Means & Random Forest) to optimize energy distribution for London households.")
    
    # Cluster Distribution Pie Chart
    st.subheader("User Distribution by Segment")
    cluster_counts = df_clusters['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    fig = px.pie(cluster_counts, values='Count', names='Cluster', 
                 title='Household Segmentation', 
                 color='Cluster',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: USER SEGMENTS (CLUSTERING) ---
elif page == "User Segments":
    st.markdown("### 🔍 Customer Segmentation Analysis")
    st.write("Analysis of household consumption behaviors to identify distinct user groups.")

    # 1. Profiles Heatmap (Recreating the Seaborn plot in Plotly)
    st.subheader("Cluster DNA: Behavioral Profiles")
    
    # Calculate means for heatmap
    cluster_means = df_clusters.groupby('cluster').mean()
    
    fig_heat = px.imshow(cluster_means.T, 
                         labels=dict(x="Cluster ID", y="Feature", color="Value"),
                         x=cluster_means.index,
                         y=cluster_means.columns,
                         color_continuous_scale='RdBu_r',
                         origin='lower')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # 2. Detailed Interpretations & Recommendations
    st.subheader("🤖 AI Recommendations")
    
    selected_cluster = st.selectbox("Select a Cluster to Analyze:", sorted(df_clusters['cluster'].unique()))
    
    col_desc, col_rec = st.columns(2)
    
    with col_desc:
        st.markdown(f"**Profile for Cluster {selected_cluster}:**")
        stats = cluster_means.loc[selected_cluster]
        st.write(f"- **Avg Usage:** {stats['mean_usage']:.3f} kWh/hh")
        st.write(f"- **Peak Time:** {'Evening' if stats['mean_Evening'] > stats['mean_Morning'] else 'Morning'}")
        st.write(f"- **Volatility:** {'High' if stats['std_dev'] > 0.5 else 'Stable'}")
        
    with col_rec:
        st.markdown("**Actionable Insights:**")
        if selected_cluster == 3: # Heavy Users
            st.warning("⚠️ **High Intensity Group:** Recommended for 'Time-of-Use' tariff switch. Consider shifting EV charging and laundry to post-midnight hours.")
        elif selected_cluster == 0: # Low Users
            st.success("✅ **Efficient Group:** No major interventions needed. Eligible for 'Energy Saver' rebates.")
        else:
            st.info("ℹ️ **Standard Group:** Monitor heating settings during winter evenings.")

# --- PAGE 3: FORECASTING ---
elif page == "Demand Forecasting":
    st.markdown("### 📈 Predictive Grid Analytics")
    st.write("Forecasting demand using Random Forest Regression (Results from Phase 2).")

    # Dropdown to select cluster series
    # Filter columns that end with '_Pred'
    pred_cols = [c for c in df_preds.columns if '_Pred' in c]
    # Clean names for dropdown
    options = [c.replace('_Pred', '') for c in pred_cols]
    
    selected_series = st.selectbox("Select Cluster Load Profile:", options)
    
    # Plotting
    st.subheader(f"Forecast vs Actual: {selected_series}")
    
    # Slider to zoom in on dates
    dates = st.slider("Select Date Range:", 
                      min_value=df_preds.index.date.min(), 
                      max_value=df_preds.index.date.max(),
                      value=(df_preds.index.date.max()-pd.Timedelta(days=7), df_preds.index.date.max()))
    
    # Filter data
    mask = (df_preds.index.date >= dates[0]) & (df_preds.index.date <= dates[1])
    subset = df_preds.loc[mask]
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=subset.index, y=subset[selected_series], mode='lines', name='Actual Load', line=dict(color='blue', width=2)))
    fig_line.add_trace(go.Scatter(x=subset.index, y=subset[f"{selected_series}_Pred"], mode='lines', name='Predicted Load', line=dict(color='red', dash='dash')))
    
    fig_line.update_layout(title="Energy Demand (kWh)", xaxis_title="Time", yaxis_title="Consumption")
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Metrics
    st.markdown("#### Model Performance (Test Set)")
    # Simple dummy calculation or display real metrics if you saved them
    rmse = ((subset[selected_series] - subset[f"{selected_series}_Pred"]) ** 2).mean() ** 0.5
    st.metric("Real-time RMSE (Error)", f"{rmse:.4f}")

# --- PAGE 4: ANOMALY DETECTION ---
elif page == "Safety & Anomalies":
    st.markdown("### 🛡️ Anomaly Detection System")
    st.write("Identifying irregular consumption patterns using Isolation Forest.")
    
    st.markdown("#### Case Study: User MAC000049")
    
    # Filter only anomalies
    anomalies_only = df_anomalies[df_anomalies['is_anomaly'] == True]
    
    # Interactive Plot
    fig_anom = go.Figure()
    
    # 1. Normal Usage
    fig_anom.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies['usage'], 
                                  mode='lines', name='Normal Usage', 
                                  line=dict(color='lightgrey')))
    
    # 2. Anomalies
    fig_anom.add_trace(go.Scatter(x=anomalies_only.index, y=anomalies_only['usage'], 
                                  mode='markers', name='Anomaly Detected', 
                                  marker=dict(color='red', size=8, symbol='x')))
    
    fig_anom.update_layout(title="Anomaly Timeline (2013)", xaxis_title="Time", yaxis_title="Energy (kWh)")
    st.plotly_chart(fig_anom, use_container_width=True)
    
    st.error(f"🚨 **Alert:** {len(anomalies_only)} irregular events detected. These correlate with usage spikes > 3.0 kWh.")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 EcoSmart Project | MSc Data Science Dissertation")