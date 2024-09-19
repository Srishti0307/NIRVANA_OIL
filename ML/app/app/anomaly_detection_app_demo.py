import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import load_model
import joblib
import pickle
from tensorflow.keras.losses import MeanSquaredError
import plotly.graph_objects as go

# Load models
custom_objects = {'mse': MeanSquaredError()}
dbscan_model = pickle.load(open(r'D:\Projects\AIS-Anamoly_detector\app\app\models\dbscan_model.pkl', 'rb'))
isolation_forest_model = joblib.load(r'D:\Projects\AIS-Anamoly_detector\app\app\models\isolation_forest_model.joblib')
autoencoder_model = load_model(r'D:\Projects\AIS-Anamoly_detector\app\app\models\autoencoder_model.h5', custom_objects=custom_objects)
with open(r'D:\Projects\AIS-Anamoly_detector\app\app\models\SOG_kalman_filter.pkl', 'rb') as f:
    kalman_filter = pickle.load(f)

# Streamlit app
st.title("AIS Data Anomaly Detection")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload AIS CSV File", type="csv")

if uploaded_file:
    # Read the uploaded CSV file
    ais_data = pd.read_csv(uploaded_file)
    
    # Extract unique MMSI numbers and display a dropdown menu for selection
    mmsi_numbers = ais_data['MMSI'].unique().tolist()
    selected_mmsi = st.selectbox('Select MMSI Number:', mmsi_numbers)
    
    if st.button('Process MMSI'):
        # Filter the data based on the selected MMSI number
        vessel_data = ais_data[ais_data['MMSI'] == selected_mmsi]
        
        if vessel_data.empty:
            st.write(f"No data found for MMSI: {selected_mmsi}")
        else:
            # Feature Engineering
            vessel_data['speed_change'] = vessel_data['SOG'].diff()
            vessel_data['acceleration'] = vessel_data['speed_change'].diff()
            vessel_data['turning_rate'] = vessel_data['Heading'].diff()
            vessel_data['sog_anomaly'] = vessel_data['SOG'] > (vessel_data['SOG'].mean() + 2 * vessel_data['SOG'].std())
            vessel_data['draught_change'] = vessel_data['Draft'].diff()

            # Prepare feature set
            features = vessel_data[['SOG', 'COG', 'LAT', 'LON']].values

            # Apply DBSCAN
            vessel_data['anomaly_dbscan'] = dbscan_model.fit_predict(features) == -1

            # Apply Isolation Forest
            vessel_data['anomaly_isolation_forest'] = isolation_forest_model.predict(features) == -1

            # Apply Autoencoder
            features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
            reconstructions = autoencoder_model.predict(features_norm)
            mse = np.mean(np.power(features_norm - reconstructions, 2), axis=1)
            vessel_data['anomaly_autoencoder'] = mse > np.percentile(mse, 95)

            # Kalman Filter application
            state_means, _ = kalman_filter.em(vessel_data['SOG'].values).filter(vessel_data['SOG'].values)
            state_means = state_means.ravel()  # Ensure state_means is 1D
            vessel_data['SOG_kalman'] = state_means
            vessel_data['anomaly_kalman'] = np.abs(vessel_data['SOG'] - state_means) > 2 * vessel_data['SOG'].std()

            # Combine anomalies from all models
            vessel_data['combined_anomaly'] = vessel_data[['anomaly_dbscan', 'anomaly_isolation_forest', 'anomaly_autoencoder', 'anomaly_kalman']].sum(axis=1) > 2

            # Display the number of detected anomalies
            st.write(f"Total anomalies detected: {vessel_data['combined_anomaly'].sum()}")
            
            # Display the resulting DataFrame with anomalies
            st.dataframe(vessel_data)

            # Plotting the graph of SOG with anomalies highlighted
            fig = go.Figure()

            # Plot normal values
            fig.add_trace(go.Scatter(
                x=vessel_data.index, y=vessel_data['SOG'],
                mode='lines',
                name='Normal SOG',
                line=dict(color='blue'),
                hoverinfo='x+y'
            ))

            # Plot anomalies
            anomaly_points = vessel_data[vessel_data['combined_anomaly'] == True]
            fig.add_trace(go.Scatter(
                x=anomaly_points.index, y=anomaly_points['SOG'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8),
                hoverinfo='x+y'
            ))

            # Update layout
            fig.update_layout(
                title="SOG Values with Anomalies Highlighted",
                xaxis_title="Index",
                yaxis_title="SOG",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="closest"
            )

            # Display the plot
            st.plotly_chart(fig)
            
            # Allow download of processed data
            csv = vessel_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download processed data with anomalies", data=csv, file_name='vessel_data_with_anomalies.csv', mime='text/csv')
