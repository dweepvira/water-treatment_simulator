"""
Water Treatment Plant Simulator: Main Entry Point

This is the main script to launch the simulator and the Flask web UI.
It handles configuration loading, dependency version checking,
and starting the Modbus server, simulation loop, and web interface in separate threads.
"""
import logging
import yaml
import argparse
import os
import threading
from flask import Flask, render_template_string, request, redirect, url_for
from plc.modbus_server import WaterPlantSimulator

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

# --- Global variable for the simulator ---
simulator = None
app = Flask(__name__)

# --- Flask Web UI Routes ---
@app.route('/')
def index():
    """Renders the main dashboard."""
    if simulator:
        # Create a sorted list of register names
        sorted_registers = sorted(simulator.plc.state.keys())
        
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SCADA Simulator Control Panel</title>
            <style>
                body { font-family: sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }
                .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1, h2 { color: #444; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
                th { background-color: #007bff; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                form { display: flex; align-items: center; }
                input[type="number"] { padding: 8px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }
                input[type="submit"] { padding: 8px 15px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
                input[type="submit"]:hover { background-color: #218838; }
                .status { padding: 10px; border-radius: 4px; margin-top: 15px; }
                .status-normal { background-color: #d4edda; color: #155724; }
                .status-attack { background-color: #f8d7da; color: #721c24; }
            </style>
            <meta http-equiv="refresh" content="5">
        </head>
        <body>
            <div class="container">
                <h1>SCADA Water Plant Live Status</h1>
                <p><strong>Simulation Time:</strong> {{ sim_time }}</p>
                <div class="status {{ 'status-attack' if active_attack != 'None' else 'status-normal' }}">
                    <strong>Current Status:</strong> {{ active_attack }}
                </div>
                <h2>Register Values</h2>
                <table>
                    <tr>
                        <th>Register Name</th>
                        <th>Current Value</th>
                        <th>Manual Override</th>
                    </tr>
                    {% for reg in registers %}
                    <tr>
                        <td>{{ reg }}</td>
                        <td>{{ plc_state[reg] }}</td>
                        <td>
                            <form action="/update" method="post">
                                <input type="hidden" name="register" value="{{ reg }}">
                                <input type="number" name="value" placeholder="New value">
                                <input type="submit" value="Set">
                            </form>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """
        return render_template_string(
            template_str,
            registers=sorted_registers,
            plc_state=simulator.plc.state,
            active_attack=simulator.attacker.active_attack,
            sim_time=str(simulator.get_current_sim_time())
        )
    return "<h1>Simulator not running.</h1>"

@app.route('/update', methods=['POST'])
def update_register():
    """Handles manual override of register values."""
    if simulator:
        register = request.form['register']
        value = request.form['value']
        if register in simulator.plc.state and value:
            try:
                # Manually set the state in the PLC logic
                simulator.plc.state[register] = int(value)
                log.info(f"WEB UI: Manually set {register} to {value}")
            except ValueError:
                log.warning(f"WEB UI: Invalid value '{value}' for {register}")
    return redirect(url_for('index'))

def run_flask_app():
    """Runs the Flask web server."""
    log.info("Starting Flask web UI on http://127.0.0.1:5000")
    # Use '127.0.0.1' to keep it local
    app.run(host='127.0.0.1', port=5000, debug=False)

def main():
    """Main function to run the simulator."""
    global simulator
    
    print("--- Launching Water Treatment Plant Simulator ---")
    parser = argparse.ArgumentParser(description="Water Treatment Plant SCADA Simulator")
    parser.add_argument(
        "--config",
        default="configs/water_plant_config.yaml",
        help="Path to the master YAML configuration file."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        log.error(f"FATAL: Configuration file not found at '{args.config}'")
        log.error("Please ensure the config file exists and you are running from the project root.")
        return
        
    log.info(f"Using configuration file: {args.config}")

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        log.info("Configuration loaded successfully.")
    except yaml.YAMLError as e:
        log.error(f"FATAL: Error parsing YAML configuration file: {e}")
        return

    # Initialize the simulator
    simulator = WaterPlantSimulator(config)

    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    try:
        # Run the main simulation loop in the main thread
        simulator.run()
    except KeyboardInterrupt:
        log.info("Simulator interrupted by user.")
    except Exception as e:
        log.critical(f"A critical error occurred during simulation: {e}", exc_info=True)
    finally:
        log.info("Simulator shutdown complete.")

if __name__ == "__main__":
    main()
