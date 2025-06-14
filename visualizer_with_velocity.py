
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import signal

st.title("Accelerometer + Gyroscope Visualization with Manual Label + Velocity")

# Upload files
acc_file = st.file_uploader("Upload Linear Acceleration CSV", type="csv")
gyro_file = st.file_uploader("Upload Gyroscope CSV", type="csv")

# Input label manually
label_input = st.text_input("Enter the label for this session (e.g., walk, metro):", "walk")

def highpass_filter(data, cutoff=0.05, fs=100.0, order=2):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)

if acc_file and gyro_file and label_input:
    acc_df = pd.read_csv(acc_file)
    gyro_df = pd.read_csv(gyro_file)

    merged_df = pd.concat([acc_df.reset_index(drop=True), gyro_df.reset_index(drop=True)], axis=1)
    merged_df.columns = ['acc_time', 'acc_x', 'acc_y', 'acc_z',
                         'gyro_time', 'gyro_x', 'gyro_y', 'gyro_z']
    merged_df['label'] = label_input

    # High-pass filter
    merged_df['acc_x_filtered'] = highpass_filter(merged_df['acc_x'])
    merged_df['acc_y_filtered'] = highpass_filter(merged_df['acc_y'])
    merged_df['acc_z_filtered'] = highpass_filter(merged_df['acc_z'])

    # Velocity estimation via integration
    dt = 1.0 / 100  # assume 100 Hz
    merged_df['vel_x'] = np.cumsum(merged_df['acc_x_filtered']) * dt
    merged_df['vel_y'] = np.cumsum(merged_df['acc_y_filtered']) * dt
    merged_df['vel_z'] = np.cumsum(merged_df['acc_z_filtered']) * dt

    # Preview
    st.write("Merged Data Preview:", merged_df.head())

    # Plot Accelerometer
    st.subheader("Accelerometer Data")
    fig_acc = px.line(merged_df, x="acc_time", y=["acc_x", "acc_y", "acc_z"],
                      labels={"value": "Acceleration (m/s²)", "acc_time": "Time (s)", "variable": "Axis"})
    st.plotly_chart(fig_acc, use_container_width=True)

    # Plot Gyroscope
    st.subheader("Gyroscope Data")
    fig_gyro = px.line(merged_df, x="gyro_time", y=["gyro_x", "gyro_y", "gyro_z"],
                       labels={"value": "Angular Velocity (rad/s)", "gyro_time": "Time (s)", "variable": "Axis"})
    st.plotly_chart(fig_gyro, use_container_width=True)

    # Plot Filtered vs Raw
    st.subheader("Filtered vs Raw Acceleration (X)")
    fig_filtered = px.line(merged_df, x="acc_time", y=["acc_x", "acc_x_filtered"],
                           labels={"value": "Acceleration (m/s²)", "acc_time": "Time", "variable": "Signal"},
                           title="Raw vs Filtered Acceleration X")
    st.plotly_chart(fig_filtered, use_container_width=True)

    # Plot Velocity
    st.subheader("Estimated Velocity X Over Time")
    fig_vel = px.line(merged_df, x="acc_time", y="vel_x",
                      labels={"vel_x": "Velocity (m/s)", "acc_time": "Time"},
                      title="Estimated Velocity X")
    st.plotly_chart(fig_vel, use_container_width=True)

    # Download option
    st.download_button("Download Enriched CSV", data=merged_df.to_csv(index=False), file_name="enriched_session.csv")
