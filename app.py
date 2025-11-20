import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from feature_engineering import create_features, get_model_features
from tensorflow.keras.models import load_model
# Page config
st.set_page_config(
    page_title="Cyclone Anomaly Detection",
    page_icon="",
    layout="wide"
)

# Title
st.title("Cyclone Preheater Anomaly Detection System")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("Upload your sensor data and select a detection model")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ["Isolation Forest", "Local Outlier Factor (LOF)", "LSTM Autoencoder"]
)

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Data",
    type=['csv'],
    help="Upload a CSV file with the same schema as data.csv"
)



# Detect button
detect_button = st.sidebar.button("Detect Anomalies", type="primary")

# Model info
model_info = {
    "Isolation Forest": {
        "description": "Detects extreme point anomalies and sudden spikes",
        "best_for": "Temperature spikes, sudden transitions",
        "file": "iso_model.pkl",
        "scaler": "iso_scaler.pkl"
    },
    "Local Outlier Factor (LOF)": {
        "description": "Identifies density-based outliers and gradual drift",
        "best_for": "Subtle degradation, operational drift",
        "file": "lof_model.pkl",
        "scaler": "lof_scaler.pkl"
    },
    "LSTM Autoencoder": {
        "description": "Detects temporal pattern deviations",
        "best_for": "Sequential anomalies, irregular transitions",
        "file": "lstm_model.h5",
        "scaler": "lstm_scaler.pkl"
    }
}


# Main content
if uploaded_file is None:
    st.info("Please upload a CSV file to begin")
    
    # Show example
    st.markdown("### Example Data Format")
    example_df = pd.DataFrame({
        'time': ['1/1/2017 0:00', '1/1/2017 0:05'],
        'Cyclone_Inlet_Gas_Temp': [867.63, 879.23],
        'Cyclone_Material_Temp': [910.42, 918.14],
        'Cyclone_Outlet_Gas_draft': [-189.54, -184.33],
        'Cyclone_cone_draft': [-186.04, -182.1],
        'Cyclone_Gas_Outlet_Temp': [852.13, 862.53],
        'Cyclone_Inlet_Draft': [-145.9, -149.76]
    })
    st.dataframe(example_df, use_container_width=True)
    
else:
    # Load data
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Data loaded: {len(df_raw)} records")
        
        if detect_button:
            with st.spinner("Processing data and detecting anomalies..."):
                # Create features
                df_feat = create_features(df_raw)
                
                # Model-specific processing
                if model_choice == "LSTM Autoencoder":
                    # LSTM requires sequences
                    try:
                        
                        
                        # Load LSTM components
                        model = load_model("models/lstm_model.keras", compile=False)


                        
                        with open('models/lstm_scaler.pkl', 'rb') as f:
                            scaler = pickle.load(f)
                        
                        with open('models/lstm_config.pkl', 'rb') as f:
                            config = pickle.load(f)
                        
                        with open('models/lstm_threshold.pkl', 'rb') as f:
                            threshold = pickle.load(f)
                        
                        seq_len = config['seq_len']
                        feature_cols = [
                            "Cyclone_Inlet_Gas_Temp",
                            "Cyclone_Gas_Outlet_Temp",
                            "Cyclone_Material_Temp",
                            "Cyclone_Inlet_Draft",
                            "Cyclone_Outlet_Gas_draft",
                            "Cyclone_cone_draft",
                            "temp_drop",
                            "pressure_drop",
                            "dT_dt",
                            "dP_dt"
                        ]
                        
                        df_model = df_feat[feature_cols].dropna()
                        X_scaled = scaler.transform(df_model.values)
                        
                        # Create sequences
                        def create_sequences(data, seq_len):
                            seqs = []
                            for i in range(len(data) - seq_len + 1):
                                seqs.append(data[i:i+seq_len])
                            return np.array(seqs)
                        
                        X_seq = create_sequences(X_scaled, seq_len)
                        
                        # Predict
                        X_pred = model.predict(X_seq, verbose=0)
                        reconstruction_errors = np.mean(np.abs(X_seq - X_pred), axis=(1, 2))
                        
                        # Anomalies
                        anomaly_flags = (reconstruction_errors > threshold).astype(int)
                        
                        # Align with original indices
                        seq_index = df_model.index[seq_len-1:]
                        df_results = pd.DataFrame({
                            'time': seq_index,
                            'Cyclone_Gas_Outlet_Temp': df_model.loc[seq_index, 'Cyclone_Gas_Outlet_Temp'],
                            'anomaly': anomaly_flags,
                            'reconstruction_error': reconstruction_errors
                        })
                        
                    except Exception as e:
                        st.error(f"Error loading LSTM model: {str(e)}")
                        st.stop()
                
                else:
                    # Isolation Forest or LOF
                    features = get_model_features()
                    df_model = df_feat[features].dropna()
                    
                    # Load model and scaler
                    model_file = f"models/{model_info[model_choice]['file']}"
                    scaler_file = f"models/{model_info[model_choice]['scaler']}"
                    
                    if not os.path.exists(model_file):
                        st.error(f"Model file not found: {model_file}")
                        st.info("Please train the model first by running the training script.")
                        st.stop()
                    
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    # Scale and predict
                    X_scaled = scaler.transform(df_model)
                    predictions = model.predict(X_scaled)
                    
                    # Convert to binary (1 = anomaly, 0 = normal)
                    anomaly_flags = np.where(predictions == -1, 1, 0)
                    
                    df_results = pd.DataFrame({
                        'time': df_model.index,
                        'Cyclone_Gas_Outlet_Temp': df_model['Cyclone_Gas_Outlet_Temp'],
                        'anomaly': anomaly_flags
                    })
                
                # Calculate statistics
                total_points = len(df_results)
                anomaly_count = df_results['anomaly'].sum()
                anomaly_percentage = (anomaly_count / total_points) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("## Detection Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Data Points", f"{total_points:,}")
                
                with col2:
                    st.metric("Anomalies Detected", f"{anomaly_count:,}")
                
                with col3:
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                
                # Plot with Plotly
                st.markdown("### Time Series with Anomalies")
                
                fig = go.Figure()
                
                # Normal points
                normal_data = df_results[df_results['anomaly'] == 0]
                fig.add_trace(go.Scatter(
                    x=normal_data['time'],
                    y=normal_data['Cyclone_Gas_Outlet_Temp'],
                    mode='lines',
                    name='Normal',
                    line=dict(color='lightblue', width=1),
                    opacity=0.7
                ))
                
                # Anomaly points
                anomaly_data = df_results[df_results['anomaly'] == 1]
                fig.add_trace(go.Scatter(
                    x=anomaly_data['time'],
                    y=anomaly_data['Cyclone_Gas_Outlet_Temp'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=6, symbol='circle'),
                ))
                
                fig.update_layout(
                    title=f"{model_choice} - Cyclone Gas Outlet Temperature",
                    xaxis_title="Time",
                    yaxis_title="Temperature (°C)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct schema.")