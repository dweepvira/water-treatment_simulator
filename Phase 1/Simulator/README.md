Dynamic Water Treatment Plant SCADA Simulator
This is an advanced SCADA simulator designed to generate datasets for cybersecurity research. It simulates a water treatment plant's physical processes, its PLC control logic, and its Modbus TCP network communication.

Features
Modbus TCP Server: Emulates a real PLC, making its data accessible over the network via standard Modbus clients.

Dynamic Process Simulation: The plant's state (turbidity, flow, etc.) changes realistically over time.

Reactive Control Logic: The simulated PLC responds to sensor readings (e.g., increases chemical dosage when turbidity is high).

Fault & Attack Injection: The simulator can introduce faulty sensor readings and execute scripted cyber attacks defined in the config file.

Detailed Packet-like Logging: At each time step, it captures snapshots of all registers before logic, after logic, and after attacks to create a rich dataset for traffic analysis.

Centralized Configuration: A single YAML file controls the entire simulation.

Project Structure
configs/water_plant_config.yaml: The main configuration file.

plc/: Contains the core Python modules for the simulator.

run_simulator.py: The main script to start the simulation.

README.md: This file.

datasets/: This directory is created automatically to store the output CSV file.

Setup Instructions

Install Dependencies
You will need Python 3. Open your terminal or command prompt and install the required libraries using pip:

pip install pyModbusTCP pyyaml

Configure the Simulation
Open configs/water_plant_config.yaml in a text editor. This is the control center for your simulation.

Adjust simulation parameters like duration_hours.

Design your cyber attack scenarios under attack_scenarios.

An example for a 1-hour automated run with pre-set attacks is included at the bottom of the file. To use it, simply comment out the default simulation and attack_scenarios sections and uncomment the example sections.

Run the Simulator
Navigate to the root water_treatment_simulator directory in your terminal and execute the run_simulator.py script:

python run_simulator.py

The simulation will start, and you will see log messages in your terminal.

Get Your Dataset
The simulation will run for the configured duration. When it is complete, it will automatically save the detailed dataset to the path specified in the config file (e.g., datasets/simulation_output.csv).