import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
import numpy as np

# --- Configuration ---
CSV_FILE = 'datasets/6_hour_balanced_simulation.csv'

# Set a visual style
plt.style.use('ggplot')
sns.set_palette("tab10")

def load_and_prep_data(filepath):
    """Loads CSV, cleans headers, and prepares time columns."""
    try:
        # skipinitialspace=True helps if there are spaces after commas in the CSV
        df = pd.read_csv(filepath, skipinitialspace=True)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        exit()

    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()

    # Convert sim_time to timedelta
    # errors='coerce' turns bad data (like "PLC Logic Cycle") into NaT instead of crashing
    df['sim_time_dt'] = pd.to_timedelta(df['sim_time'], errors='coerce')

    # Check for and drop rows that failed conversion
    initial_count = len(df)
    df = df.dropna(subset=['sim_time_dt'])
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} rows due to malformed time data (likely row shifts in CSV).")

    # Create a plotting datetime anchored to a base date
    try:
        base_date = datetime.datetime(2025, 1, 1)
        df['plot_time'] = df['sim_time_dt'].apply(lambda t: base_date + t)
    except Exception as e:
        print(f"Error preparing plot_time: {e}")
        exit()
    
    return df

def get_attack_intervals(df):
    """
    Scans the 'active_attack' column to dynamically find start/end times 
    of attacks instead of hardcoding them.
    """
    intervals = []
    
    # Filter out 'None' or NaNs
    df['active_attack'] = df['active_attack'].fillna('None')
    
    # Identify where the attack status changes
    # We create a boolean mask: True if the attack name is different from the previous row
    df['attack_change'] = df['active_attack'] != df['active_attack'].shift(1)
    
    # Get indices where changes happen
    change_indices = df[df['attack_change']].index.tolist()
    
    # Iterate through changes to build start/end pairs
    for idx in change_indices:
        attack_name = df.loc[idx, 'active_attack']
        
        # We only care if the new status is NOT 'None'
        if attack_name != 'None':
            start_time = df.loc[idx, 'plot_time']
            
            # Find the next change index (end of this attack)
            next_change_list = [i for i in change_indices if i > idx]
            if next_change_list:
                end_idx = next_change_list[0] - 1 # The row before the next change
                # Ensure end_idx is valid
                if end_idx in df.index:
                    end_time = df.loc[end_idx, 'plot_time']
                else:
                    end_time = df.loc[next_change_list[0], 'plot_time'] # Fallback
            else:
                # If no more changes, attack goes to end of file
                end_time = df['plot_time'].iloc[-1]
            
            intervals.append({
                'name': attack_name,
                'start': start_time,
                'end': end_time,
                'color': 'red' 
            })
            
    return intervals

def plot_dual_axis(df, intervals, y1_col, y2_col, title, filename, y1_label=None, y2_label=None):
    """Generates a dual-axis line chart with attack shading."""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Y1
    color1 = 'tab:blue'
    ax1.set_xlabel('Simulation Time (HH:MM:SS)')
    ax1.set_ylabel(y1_label if y1_label else y1_col, color=color1, fontsize=12, fontweight='bold')
    ax1.plot(df['plot_time'], df[y1_col], color=color1, label=y1_label)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Y2
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel(y2_label if y2_label else y2_col, color=color2, fontsize=12, fontweight='bold')
    ax2.plot(df['plot_time'], df[y2_col], color=color2, linestyle='--', label=y2_label)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Shade Attacks
    added_labels = set()
    colors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
    
    for i, attack in enumerate(intervals):
        c = colors[i % len(colors)]
        label = attack['name'] if attack['name'] not in added_labels else "_nolegend_"
        
        # Ensure we don't crash if start/end are out of bounds or NaT (though we filtered them)
        if pd.notnull(attack['start']) and pd.notnull(attack['end']):
            ax1.axvspan(attack['start'], attack['end'], color=c, alpha=0.2, label=label)
            added_labels.add(attack['name'])

    # Formatting
    plt.title(title, fontsize=16, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, -0.1), ncol=3)

    xfmt = mdates.DateFormatter('%H:%M:%S')
    ax1.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Generated: {filename}")
    plt.close()

def plot_network_traffic(df, intervals, filename):
    """Specific plot for packet sizes/network anomalies."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    color = 'tab:purple'
    ax.scatter(df['plot_time'], df['packet_size'], alpha=0.6, s=10, c=color, label='Packet Size')
    ax.set_ylabel('Packet Size (Bytes)', fontsize=12)
    ax.set_xlabel('Simulation Time')
    
    # Shade Attacks
    added_labels = set()
    for i, attack in enumerate(intervals):
        label = attack['name'] if attack['name'] not in added_labels else "_nolegend_"
        if pd.notnull(attack['start']) and pd.notnull(attack['end']):
            ax.axvspan(attack['start'], attack['end'], color='red', alpha=0.15, label=label)
            added_labels.add(attack['name'])

    plt.title('Network Traffic Analysis: Packet Size vs. Attack Windows', fontsize=16, fontweight='bold')
    
    xfmt = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Generated: {filename}")
    plt.close()

def plot_attack_gantt(intervals, filename):
    """
    Generates a Gantt chart of attacks to show randomness/distribution.
    This helps visualize the 'Uncoordinated / Real-World' scenario.
    """
    if not intervals:
        print("No attacks found to plot Gantt chart.")
        return

    # Convert list of dicts to DataFrame for easier handling
    df_attacks = pd.DataFrame(intervals)
    
    # Get unique attack names to assign Y-axis positions
    unique_attacks = df_attacks['name'].unique()
    y_pos = {name: i for i, name in enumerate(unique_attacks)}
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create a color map so each attack type has a distinct color
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_attacks)))
    color_map = dict(zip(unique_attacks, colors))
    
    # Plot each interval as a horizontal bar
    for _, row in df_attacks.iterrows():
        start_num = mdates.date2num(row['start'])
        end_num = mdates.date2num(row['end'])
        duration = end_num - start_num
        
        ax.barh(
            y=y_pos[row['name']], 
            width=duration, 
            left=start_num, 
            height=0.4, 
            color=color_map[row['name']],
            edgecolor='black',
            alpha=0.8
        )
        
    # Formatting
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()), fontsize=11, fontweight='bold')
    ax.set_xlabel("Simulation Time", fontsize=12)
    ax.set_title("Timeline of Cyber Attacks", fontsize=16, fontweight='bold')
    
    # Format X axis dates to show HH:MM:SS
    xfmt = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    
    # Grid just on the X axis to show time progression
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Generated: {filename}")
    plt.close()

# --- Main Execution ---

# 1. Load Data
df = load_and_prep_data(CSV_FILE)

# 2. Find Attacks Dynamically
attack_intervals = get_attack_intervals(df)
print(f"Detected {len(attack_intervals)} attack intervals.")

# 3. Generate Visualizations

# Graph 1: Turbidity vs Pump Speed
plot_dual_axis(
    df, attack_intervals,
    y1_col='raw_water_turbidity_after_attack',
    y2_col='coagulant_pump_speed_after_attack',
    title='1. Coagulation Process: Turbidity vs Pump Speed',
    filename='1_turbidity_pump_response.png',
    y1_label='Raw Turbidity (NTU)',
    y2_label='Pump Speed (RPM)'
)

# Graph 2: Chemical Dosing
plot_dual_axis(
    df, attack_intervals,
    y1_col='raw_water_flow_rate_after_attack',
    y2_col='chlorine_level_after_attack',
    title='2. Disinfection Process: Flow Rate vs Chlorine Level',
    filename='2_chlorine_dosing.png',
    y1_label='Flow Rate (L/m)',
    y2_label='Chlorine Level (ppm)'
)

# Graph 3: Filter Performance
plot_dual_axis(
    df, attack_intervals,
    y1_col='sedimentation_turbidity_after_attack',
    y2_col='filter_outlet_turbidity_after_attack',
    title='3. Filtration Efficacy: Sedimentation vs Outlet Turbidity',
    filename='3_filter_performance.png',
    y1_label='Sedimentation Turbidity (NTU)',
    y2_label='Filter Outlet Turbidity (NTU)'
)

# Graph 4: Network Traffic
plot_network_traffic(df, attack_intervals, '4_network_packet_analysis.png')

# Graph 5: Attack Timeline (Gantt Chart) - NEW!
plot_attack_gantt(attack_intervals, '5_attack_timeline_distribution.png')

print("\nAll graphs generated successfully.")