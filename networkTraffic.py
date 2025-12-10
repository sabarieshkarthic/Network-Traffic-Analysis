import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from scapy.all import sniff, IP, TCP, UDP
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

st.set_page_config(page_title="Network Traffic Analyzer", layout="wide")

def extract_features(packets):
    sizes = [len(pkt) for pkt in packets]
    times = [pkt.time for pkt in packets]

    burst_rate = len(packets) / 10.0  # packets/sec (10s window)
    avg_packet_size = np.mean(sizes) if sizes else 0

    tcp_count = sum(1 for pkt in packets if TCP in pkt)
    udp_count = sum(1 for pkt in packets if UDP in pkt)
    total = tcp_count + udp_count

    protocol_ratio_TCP = tcp_count / total if total else 0
    protocol_ratio_UDP = udp_count / total if total else 0

    size_scaled = (avg_packet_size - np.min(sizes)) / (np.max(sizes) - np.min(sizes)) if sizes and np.max(sizes) != np.min(sizes) else 0

    ports = [pkt[TCP].sport if TCP in pkt else pkt[UDP].sport if UDP in pkt else None for pkt in packets]
    ports = [p for p in ports if p is not None]
    if ports:
        port_counts = pd.Series(ports).value_counts(normalize=True)
        port_entropy = -np.sum(port_counts * np.log2(port_counts))
    else:
        port_entropy = 0

    inter_packet_gaps = np.diff(times) if len(times) > 1 else [0]
    inter_packet_gap = np.mean(inter_packet_gaps)
    jitter = np.std(inter_packet_gaps)

    uplink = sum(len(pkt) for pkt in packets if IP in pkt and pkt[IP].src.startswith("192.168."))
    downlink = sum(len(pkt) for pkt in packets if IP in pkt and pkt[IP].dst.startswith("192.168."))
    byte_ratio_uplink_downlink = uplink / (downlink + 1)

    return [
        burst_rate, avg_packet_size, protocol_ratio_TCP, protocol_ratio_UDP,
        size_scaled, port_entropy, inter_packet_gap, jitter, byte_ratio_uplink_downlink
    ]

columns = [
    "burst_rate", "avg_packet_size", "protocol_ratio_TCP", "protocol_ratio_UDP",
    "size_scaled", "port_entropy", "inter_packet_gap", "jitter", "byte_ratio_uplink_downlink"
]

def assign_traffic_type(row, centers_df):
    cluster_id = int(row["kmeans_cluster"])
    center = centers_df.iloc[cluster_id]

    if center["avg_packet_size"] > 1000 and center["burst_rate"] > 50:
        return "Streaming/File Transfer"
    elif center["protocol_ratio_UDP"] > 0.6 and center["jitter"] > 0.01:
        return "VoIP"
    elif center["avg_packet_size"] < 500 and center["burst_rate"] < 20:
        return "Browsing"
    elif center["protocol_ratio_TCP"] > 0.5 and center["jitter"] < 0.01:
        return "Interactive TCP Apps"
    else:
        return "Unknown"

def run_phase1(training_data=None, duration=120):
    features = []
    start_time = time.time()
    batch_count = 0

    st.session_state.phase = "Phase 1: Learning"
    st.session_state.status = " Training models..."

    while time.time() - start_time < duration:
        batch_count += 1
        st.session_state.status = f" Capturing batch {batch_count} in Phase 1..."
        update_status()
        packets = sniff(timeout=10)
        feats = extract_features(packets)
        features.append(feats)
        time.sleep(0.1)

    df = pd.DataFrame(features, columns=columns)

    if training_data is not None and not training_data.empty:
        df = pd.concat([df, training_data[columns]], ignore_index=True)

    kmeans = KMeans(n_clusters=5, random_state=42).fit(df)
    df["kmeans_cluster"] = kmeans.labels_

    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
    df["traffic_type"] = df.apply(lambda row: assign_traffic_type(row, centers_df), axis=1)

    X = df[columns]
    y = df["traffic_type"]
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    df["bandwidth_mbps"] = (df["burst_rate"] * df["avg_packet_size"] * 8) / 1e6
    df["jitter_ms"] = df["jitter"] * 1000

    X_reg = df[columns]
    bw_model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    jit_model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    bw_model.fit(X_reg, df["bandwidth_mbps"])
    jit_model.fit(X_reg, df["jitter_ms"])

    joblib.dump(clf, "traffic_classifier.pkl")
    joblib.dump(bw_model, "bandwidth_predictor.pkl")
    joblib.dump(jit_model, "jitter_predictor.pkl")

    st.session_state.status = "✅ Phase 1 complete. Models updated."
    update_status()
    return df

@st.cache_resource
def load_models():
    try:
        clf = joblib.load("traffic_classifier.pkl")
        bw_model = joblib.load("bandwidth_predictor.pkl")
        jit_model = joblib.load("jitter_predictor.pkl")
        return clf, bw_model, jit_model
    except FileNotFoundError:
        st.error(" Model files not found. Running initial Phase 1.")
        run_phase1()
        return load_models()

def run_phase2(duration=120):
    clf, bw_model, jit_model = load_models()

    monitoring_features = []
    batch_count = st.session_state.global_batch_count
    start_time = time.time()

    st.session_state.phase = "Phase 2: Monitoring"
    st.session_state.status = " Monitoring network traffic..."

    while time.time() - start_time < duration:
        batch_count += 1
        st.session_state.global_batch_count = batch_count
        st.session_state.status = f" Capturing batch {batch_count} in Phase 2..."
        update_status()

        packets = sniff(timeout=10)
        feats = extract_features(packets)
        feats_df = pd.DataFrame([feats], columns=columns)
        monitoring_features.append(feats)

        traffic_type = clf.predict(feats_df)[0]
        burst_rate = feats[0]
        avg_packet_size = feats[1]
        bandwidth = (burst_rate * avg_packet_size * 8) / 1e6
        jitter_ms = jit_model.predict(feats_df)[0]

        action = "No action needed"
        alert_level = "🟢"

        if traffic_type == "VoIP" and jitter_ms > 20:
            action = " Increase QoS for VoIP"
            alert_level = "🔴"
        elif traffic_type == "Streaming/File Transfer" and bandwidth > 100:
            action = " Throttle bandwidth"
            alert_level = "🟡"
        elif traffic_type == "Browsing":
            action = " Minimal allocation"
            alert_level = "🟢"
        elif "Unknown" in str(traffic_type):
            action = " Investigate traffic pattern"
            alert_level = "🔴"

        timestamp = pd.Timestamp.now()
        st.session_state.history.append({
            "timestamp": timestamp,
            "batch": batch_count,
            "traffic_type": traffic_type,
            "bandwidth": bandwidth,
            "jitter": jitter_ms,
            "action": action,
            "alert": alert_level
        })

        st.session_state.log.append({
            "timestamp": timestamp,
            "batch": batch_count,
            "traffic_type": traffic_type,
            "bandwidth": f"{bandwidth:.1f} Mbps",
            "action": action
        })

        update_dashboard(st.session_state.history, st.session_state.log, traffic_type, bandwidth, jitter_ms, action, alert_level, burst_rate, avg_packet_size)

        time.sleep(0.5)

    st.session_state.last_prediction = st.session_state.history[-1] if st.session_state.history else None
    monitoring_df = pd.DataFrame(monitoring_features, columns=columns)
    return monitoring_df


def update_status():
    with st.session_state.status_placeholder.container():
        st.info(f"**Phase:** {st.session_state.phase}")
        st.info(f"**Status:** {st.session_state.status}")
        st.info(f"**Cycle:** {st.session_state.cycle}")

def display_learning():
    with st.session_state.metrics_placeholder.container():
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            last = st.session_state.last_prediction
            st.subheader(" Last Prediction from Previous Cycle")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Traffic Type", last['traffic_type'])
            with col2:
                st.metric("Bandwidth", f"{last['bandwidth']:.1f} Mbps")
            with col3:
                st.metric("Jitter", f"{last['jitter']:.1f} ms")
            with col4:
                st.metric("Status", last['alert'])
            st.info(f"**Optimization Action:** {last['action']}")
        else:
            st.info("No previous prediction available.")

    if st.session_state.history:
        update_dashboard(st.session_state.history, st.session_state.log, None, None, None, None, None, None, None)
    else:
        with st.session_state.charts_placeholder.container():
            st.info("Charts unavailable during initial learning phase.")
        with st.session_state.log_placeholder.container():
            st.info("Log unavailable during initial learning phase.")

def update_dashboard(history, log, traffic_type, bandwidth, jitter_ms, action, alert_level, burst_rate, avg_packet_size):
    with st.session_state.metrics_placeholder.container():
        if traffic_type is not None:
            st.subheader(" Current Batch Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Traffic Type", str(traffic_type))
            with col2:
                st.metric("Bandwidth", f"{bandwidth:.1f} Mbps")
            with col3:
                st.metric("Jitter", f"{jitter_ms:.1f} ms")
            with col4:
                st.metric("Status", alert_level)
            st.info(f"**Optimization Action:** {action}")

            with st.expander(" Feature Details"):
                st.write(f"Burst Rate: {burst_rate:.1f} pkts/sec")
                st.write(f"Avg Packet Size: {avg_packet_size:.1f} bytes")
                st.write(f"Bandwidth Calculation: ({burst_rate:.1f} × {avg_packet_size:.1f} × 8) / 1,000,000 = {bandwidth:.1f} Mbps")
        else:
            if 'last_prediction' in st.session_state and st.session_state.last_prediction:
                last = st.session_state.last_prediction
                st.subheader(" Last Prediction from Previous Cycle")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Traffic Type", last['traffic_type'])
                with col2:
                    st.metric("Bandwidth", f"{last['bandwidth']:.1f} Mbps")
                with col3:
                    st.metric("Jitter", f"{last['jitter']:.1f} ms")
                with col4:
                    st.metric("Status", last['alert'])
                st.info(f"**Optimization Action:** {last['action']}")
            else:
                st.info("No metrics available.")

    if len(history) > 0:
        hist_df = pd.DataFrame(history)
        with st.session_state.charts_placeholder.container():
            st.subheader(" Network Analytics")
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Bandwidth Trend',
                    'Jitter Trend',
                    'Traffic Type Distribution',
                    'Bandwidth by Traffic Type'
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"type": "pie"}, {"type": "bar"}]
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=hist_df['batch'],
                    y=hist_df['bandwidth'],
                    mode='lines+markers',
                    name='Bandwidth',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=hist_df['batch'],
                    y=hist_df['jitter'],
                    mode='lines+markers',
                    name='Jitter',
                    line=dict(color='red')
                ),
                row=1, col=2
            )

            traffic_counts = hist_df['traffic_type'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=traffic_counts.index,
                    values=traffic_counts.values,
                    name="Traffic Distribution"
                ),
                row=2, col=1
            )

            avg_bw_by_type = hist_df.groupby('traffic_type')['bandwidth'].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=avg_bw_by_type['traffic_type'],
                    y=avg_bw_by_type['bandwidth'],
                    name="Avg Bandwidth"
                ),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.cycle}_{st.session_state.global_batch_count}")

    with st.session_state.log_placeholder.container():
        st.subheader(" Optimization Log")
        if len(log) > 0:
            log_df = pd.DataFrame(log)
            st.dataframe(log_df.sort_values('batch', ascending=False).head(10),
                         use_container_width=True)

st.title(" Network Traffic Analysis & Optimization")
st.markdown("---")

if 'cycle' not in st.session_state:
    st.session_state.cycle = 0
if 'phase' not in st.session_state:
    st.session_state.phase = "Idle"
if 'status' not in st.session_state:
    st.session_state.status = "Ready"
if 'previous_monitoring_data' not in st.session_state:
    st.session_state.previous_monitoring_data = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'log' not in st.session_state:
    st.session_state.log = []
if 'global_batch_count' not in st.session_state:
    st.session_state.global_batch_count = 0
if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = st.empty()
if 'metrics_placeholder' not in st.session_state:
    st.session_state.metrics_placeholder = st.empty()
if 'charts_placeholder' not in st.session_state:
    st.session_state.charts_placeholder = st.empty()
if 'log_placeholder' not in st.session_state:
    st.session_state.log_placeholder = st.empty()


col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("▶️ Start"):
        st.session_state.is_running = True
with col2:
    if st.button(" Stop"):
        st.session_state.is_running = False
        st.session_state.history = []
        st.session_state.log = []
        st.session_state.global_batch_count = 0
with col3:
    st.info(f"**Current Phase:** {st.session_state.phase}")

if st.session_state.is_running:
    while st.session_state.is_running:
        st.session_state.cycle += 1
        update_status()

        display_learning()
        _ = run_phase1(st.session_state.previous_monitoring_data)

        monitoring_data = run_phase2()

        st.session_state.previous_monitoring_data = monitoring_data

else:
    with st.session_state.status_placeholder.container():
        st.info("Click 'Start' ")
    with st.session_state.charts_placeholder.container():
        st.warning("Charts will appear here when process is running.")
    with st.session_state.log_placeholder.container():
        st.warning("Log will appear here when process is running.") 