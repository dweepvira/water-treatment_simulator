import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# --- Configuration ---
# Make sure the CSV file is in the same directory as this script, or provide the full path.
CSV_FILE = 'datasets/1_hour_random_attack_run.csv'
OUTPUT_PLOT_FILE = 'attack_timeline_graph.png'

# Attack timings are now taken directly from your water_plant_config.yaml
ATTACK_SCENARIOS = {
    "Short Coagulant Pump Outage": {
        "start_time": "00:12:00",
        "end_time": "00:13:30",
        "color": "orange"
    },
    "Brief False High Turbidity": {
        "start_time": "00:28:00",
        "end_time": "00:29:00",
        "color": "red"
    },
    "Stealthy Turbidity Offset": {
        "start_time": "00:45:00",
        "end_time": "00:50:00",
        "color": "purple"
    }
}

# --- Load Data ---
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE}' was not found. Please ensure it's in the correct directory.")
    exit()

# --- Data Preprocessing ---
# **FIX 1: Strip whitespace from all column headers**
df.columns = df.columns.str.strip()

try:
    df['sim_time_dt'] = pd.to_timedelta(df['sim_time'])
    # For proper plotting, we need a full datetime object.
    # We anchor it to an arbitrary date (e.g., today).
    df['plot_time'] = df['sim_time_dt'].apply(lambda t: datetime.datetime(2025, 1, 1) + t)
except Exception as e:
    print(f"Error converting 'sim_time' column to timedelta: {e}")
    print("Please check the format of the 'sim_time' column in your CSV.")
    exit()


# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(15, 8))

# Plot Raw Water Turbidity on primary Y-axis
ax1.plot(df['plot_time'], df['raw_water_turbidity_after_attack'], label='Raw Water Turbidity (NTU)', color='#0077b6')
ax1.set_xlabel('Simulation Time (HH:MM:SS)', fontsize=12)
ax1.set_ylabel('Raw Water Turbidity (NTU)', color='#0077b6', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#0077b6')
ax1.grid(True, linestyle='--', alpha=0.6)

# Create a second Y-axis for Coagulant Pump Speed
ax2 = ax1.twinx()
ax2.plot(df['plot_time'], df['coagulant_pump_speed_after_attack'], label='Coagulant Pump Speed (RPM)', color='#02c39a', linestyle='--')
ax2.set_ylabel('Coagulant Pump Speed (RPM)', color='#02c39a', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#02c39a')

# --- Attack Annotation ---
# Anchor the attack times to the same arbitrary date for comparison
base_date = datetime.datetime(2025, 1, 1)
for attack_name, attack_info in ATTACK_SCENARIOS.items():
    start_dt = base_date + pd.to_timedelta(attack_info["start_time"])
    end_dt = base_date + pd.to_timedelta(attack_info["end_time"])

    # Shade the attack window
    ax1.axvspan(start_dt, end_dt, color=attack_info["color"], alpha=0.2, label=f'Attack: {attack_name}')

# --- Formatting ---
fig.suptitle('Analysis of Cyber Attacks on SCADA Water Treatment Simulator', fontsize=18, weight='bold')
ax1.set_title('Turbidity and Pump Speed vs. Time')

# Format the X-axis to show time nicely
xfmt = mdates.DateFormatter('%H:%M:%S')
ax1.xaxis.set_major_formatter(xfmt)
fig.autofmt_xdate() # Rotate and align the x-labels

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# Manually create a combined legend to avoid duplicates from axvspan
legend_elements = [
    plt.Line2D([0], [0], color='#0077b6', lw=2, label='Raw Water Turbidity (NTU)'),
    plt.Line2D([0], [0], color='#02c39a', lw=2, linestyle='--', label='Coagulant Pump Speed (RPM)'),
    plt.Rectangle((0, 0), 1, 1, fc=ATTACK_SCENARIOS["Short Coagulant Pump Outage"]["color"], alpha=0.2, label='Pump Outage Attack'),
    plt.Rectangle((0, 0), 1, 1, fc=ATTACK_SCENARIOS["Brief False High Turbidity"]["color"], alpha=0.2, label='High Turbidity FDI'),
    plt.Rectangle((0, 0), 1, 1, fc=ATTACK_SCENARIOS["Stealthy Turbidity Offset"]["color"], alpha=0.2, label='Stealthy Offset FDI')
]
ax1.legend(handles=legend_elements, loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust layout to prevent title overlap
plt.savefig(OUTPUT_PLOT_FILE, dpi=300) # Save in high resolution

print(f"\nGraph generated successfully and saved to '{OUTPUT_PLOT_FILE}'")