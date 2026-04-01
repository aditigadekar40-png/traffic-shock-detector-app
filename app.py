import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv.gz")

df = df[['date_time', 'temp', 'rain_1h', 'clouds_all', 'traffic_volume']]

# Convert date
df['date_time'] = pd.to_datetime(df['date_time'])

# Remove missing values
df = df.dropna()

# -------------------------------
# Train Model
# -------------------------------
X = df[['temp', 'rain_1h', 'clouds_all']]
y = df['traffic_volume']

model = RandomForestRegressor()
model.fit(X, y)

# -------------------------------
# Shock Detection Column
# -------------------------------
avg_traffic = df['traffic_volume'].mean()
df['shock'] = df['traffic_volume'] > (avg_traffic + 1000)

# -------------------------------
# UI
# -------------------------------
st.title("🚦 Traffic Volume Prediction System")

# Sidebar Inputs
st.sidebar.header("Input Parameters")

temp = st.sidebar.slider("Temperature", 250, 320, 280)
rain = st.sidebar.slider("Rain (1h)", 0, 50, 0)
clouds = st.sidebar.slider("Clouds (%)", 0, 100, 50)

# -------------------------------
# Live Graph
# -------------------------------
st.subheader("Live Weather Input Graph")

x = ["Temperature", "Rain", "Clouds"]
y_vals = [temp, rain, clouds]

fig_live, ax_live = plt.subplots()
ax_live.plot(x, y_vals, marker='o')
st.pyplot(fig_live)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    prediction = model.predict([[temp, rain, clouds]])[0]

    st.success(f"Predicted Traffic Volume: {int(prediction)}")

    if prediction > avg_traffic + 1000:
        st.error("🚨 High Traffic! Shock detected!")
    elif prediction > avg_traffic:
        st.warning("⚡ Moderate Traffic")
    else:
        st.success("✅ Traffic is normal")

# -------------------------------
# Filter Data (WORKING)
# -------------------------------
st.subheader("📅 Filter Data")

option = st.selectbox(
    "Select Range",
    ["Last 6 Months", "Last 1 Year", "Full Data"]
)

max_date = df['date_time'].max()

if option == "Last 6 Months":
    filtered_df = df[df['date_time'] >= (max_date - pd.DateOffset(months=6))]
elif option == "Last 1 Year":
    filtered_df = df[df['date_time'] >= (max_date - pd.DateOffset(years=1))]
else:
    filtered_df = df.copy()

# Sort data
filtered_df = filtered_df.sort_values(by='date_time')

# 👉 SHOW FILTERED DATA (IMPORTANT)
st.write("Filtered Data Preview:")
st.dataframe(filtered_df.head(50))

# -------------------------------
# Dataset Statistics
# -------------------------------
st.subheader("📊 Dataset Statistics")

st.write("Total Data:", len(filtered_df))
st.write("Shock Count:", int(filtered_df['shock'].sum()))

# -------------------------------
# 🔥 Improved Traffic Graph
# -------------------------------
st.subheader("🚦 Traffic Volume with Shock Detection")

fig, ax = plt.subplots(figsize=(12,5))

# Line graph
ax.plot(
    filtered_df['date_time'],
    filtered_df['traffic_volume'],
    linewidth=1.5,
    label="Traffic Volume"
)

# Orange shock points
ax.scatter(
    filtered_df[filtered_df['shock']]['date_time'],
    filtered_df[filtered_df['shock']]['traffic_volume'],
    color='orange',
    s=40,
    label='Shock Points'
)

ax.set_title("Traffic Volume Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Traffic Volume")

ax.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)

ax.legend()

st.pyplot(fig)

# -------------------------------
# 🔥 HEATMAP
# -------------------------------
st.subheader("🔥 Dataset Heatmap")

corr = df[['temp', 'rain_1h', 'clouds_all', 'traffic_volume']].corr()

fig_heat, ax_heat = plt.subplots()
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax_heat)

st.pyplot(fig_heat)