# This file has been replaced and should be deleted
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Accelerometer + Gyroscope Visualization with Manual Label")

# Upload files
acc_file = st.file_uploader("/Users/ming/Downloads/walk_1/Linear Acceleration.csv", type="csv")
gyro_file = st.file_uploader("/Users/ming/Downloads/walk_1/Gyroscope.csv", type="csv")

# Input label manually
label_input = st.text_input("Enter the label for this session (e.g., walk, metro): Walk")

if acc_file and gyro_file and label_input:
    acc_df = pd.read_csv(acc_file)
    gyro_df = pd.read_csv(gyro_file)

    # Merge based on index (assumes same length & order)
    merged_df = pd.concat([acc_df.reset_index(drop=True), gyro_df.reset_index(drop=True)], axis=1)

    # Rename columns to avoid duplication if needed
    merged_df.columns = ['acc_time', 'acc_x', 'acc_y', 'acc_z',
                         'gyro_time', 'gyro_x', 'gyro_y', 'gyro_z']

    # Add label column
    merged_df['label'] = label_input

    # Show preview
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
    st.plotly_chart(fig_gyro, use_container_width=True
