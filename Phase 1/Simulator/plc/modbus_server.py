"""
Water Treatment Plant Simulator: Main Modbus Server Class (using pyModbusTCP)

This script defines the WaterPlantSimulator class which initializes and runs
the Modbus TCP server, orchestrates the simulation loops (PLC logic, attacks),
and handles the detailed data logging, including simulated network traffic details.
"""
import logging
import datetime
import csv
import os
import time
from pyModbusTCP.server import ModbusServer, DataBank
from plc.plc_logic import PLCLogic
from plc.attack_simulator import AttackSimulator

log = logging.getLogger(__name__)

class WaterPlantSimulator:
    def __init__(self, config):
        self.config = config
        self.register_map = config['register_map']
        self.is_running = True
        self.simulation_data = []
        self.start_time = None

        log.info("Initializing PLC logic and process...")
        self.plc = PLCLogic(self.register_map, config['process_parameters'])
        
        log.info("Initializing attack simulator...")
        # MODIFICATION: Pass the *entire* config object to the new AttackSimulator
        # This allows it to read the 'balanced_attack_config' and 'simulation' sections
        self.attacker = AttackSimulator(
            self.plc,
            config  # Changed from (self.plc, self.register_map, config.get(...))
        )
        self.data_bank = DataBank()

        server_config = self.config['plc']
        self.server = ModbusServer(
            host=server_config['ip_address'],
            port=server_config['port'],
            no_block=True,
            data_bank=self.data_bank
        )
        log.info("Simulator components initialized.")

    def get_current_sim_time(self):
        """Returns the current simulation time delta."""
        if self.start_time:
            # Use monotonic time for duration calculation to avoid issues with system time changes
            return datetime.timedelta(seconds=time.monotonic() - self.start_time_mono)
        return datetime.timedelta(0)

    def _log_state(self, log_entry, state_suffix):
        """Helper function to log the current value of all registers with a suffix."""
        for name in self.register_map:
            value = self.plc.state.get(name, 0)
            log_entry[f"{name}{state_suffix}"] = value

    def run(self):
        """Starts the server and runs the main simulation loop."""
        try:
            log.info(f"Starting Modbus TCP server on {self.config['plc']['ip_address']}:{self.config['plc']['port']}...")
            self.server.start()
            log.info("Server is running.")

            self.start_time = datetime.datetime.now()
            self.start_time_mono = time.monotonic() # Use for duration calculation
            
            total_sim_duration_delta = datetime.timedelta(hours=self.config['simulation']['duration_hours'])
            time_step_seconds = self.config['simulation']['time_step_seconds']
            
            last_step_time = time.monotonic()

            while self.is_running:
                current_time_mono = time.monotonic()
                
                # Check if it's time for the next step
                if current_time_mono - last_step_time < time_step_seconds:
                    time.sleep(0.001) # Sleep briefly to yield CPU
                    continue
                
                last_step_time = current_time_mono
                current_sim_time_delta = self.get_current_sim_time()

                if current_sim_time_delta > total_sim_duration_delta:
                    self.is_running = False
                    break

                # --- 1. Base Logging Info ---
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "sim_time": str(current_sim_time_delta),
                    "src_ip": self.config['network']['hmi_ip'],
                    "dest_ip": self.config['plc']['ip_address'],
                }
                
                # --- 2. Simulate PLC Logic and log as a 'Read/Write Cycle' ---
                self._log_state(log_entry, "_before_update")
                
                # Simulate the PLC reading sensor values
                read_registers, changed_actuators = self.plc.update(current_sim_time_delta)
                
                # Log details of the PLC logic operation
                log_entry["modbus_function"] = "PLC Logic Cycle (Read Sensors, Write Actuators)"
                log_entry["read_registers"] = str(read_registers)
                log_entry["written_registers"] = str(changed_actuators)
                log_entry["packet_size"] = 8 + (len(read_registers) * 2) + 8 + (len(changed_actuators) * 4) # Estimate
                
                self._log_state(log_entry, "_after_logic")
                
                # --- 3. Simulate Attacks and log as a 'Write Operation' ---
                attack_details = self.attacker.update(current_sim_time_delta)
                self._log_state(log_entry, "_after_attack")
                
                if attack_details:
                    log_entry["modbus_function"] = f"ATTACK: {attack_details['name']} ({attack_details['type']})"
                    log_entry["written_registers"] = str(attack_details['manipulated_registers'])
                    # Overwrite packet size for attack
                    log_entry["packet_size"] = 12 + (len(attack_details['manipulated_registers']) * 4) # Estimate for write multiple
                
                log_entry["active_attack"] = self.attacker.active_attack
                self.simulation_data.append(log_entry)

                # --- 4. Manually Update Modbus Server DataBank ---
                # This makes the *attacked* state visible to any connecting Modbus client
                for name, details in self.register_map.items():
                    address = details['address'] - 1
                    value = self.plc.state.get(name, 0) # This gets the most recent (potentially attacked) value
                    reg_type = details.get('type', 'hr').lower()

                    try:
                        if reg_type == 'hr':
                            self.data_bank.set_holding_registers(address, [int(value)])
                        elif reg_type == 'ir':
                            self.data_bank.set_input_registers(address, [int(value)])
                        elif reg_type == 'co':
                            self.data_bank.set_coils(address, [bool(value)])
                        elif reg_type == 'di':
                            self.data_bank.set_discrete_inputs(address, [bool(value)])
                    except Exception as e:
                        log.warning(f"Error setting register {name} (Addr {address}) to {value}: {e}")
                
                # We don't use time.sleep() here anymore, we rely on the loop timer
                
        finally:
            log.info("Simulation finished. Shutting down...")
            self.server.stop()
            log.info("Server stopped.")
            self.save_data_to_csv()

    def save_data_to_csv(self):
        """Writes the collected simulation data to a CSV file."""
        csv_path = self.config['simulation']['output_csv_path']
        
        if os.path.dirname(csv_path):
             os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not self.simulation_data:
            log.warning("No data to save.")
            return

        # Dynamically create header from all keys present in the data
        header = set()
        for row in self.simulation_data:
            header.update(row.keys())
        
        # Ensure a consistent order for the header
        # We can create a preferred order
        preferred_order = [
            "timestamp", "sim_time", "active_attack", "modbus_function",
            "src_ip", "dest_ip", "packet_size", 
            "read_registers", "written_registers"
        ]
        
        # Get all sensor/actuator names from register map
        process_cols = []
        for name in self.register_map.keys():
            process_cols.append(f"{name}_before_update")
            process_cols.append(f"{name}_after_logic")
            process_cols.append(f"{name}_after_attack")
            
        # Combine preferred, process, and any remaining columns
        sorted_header = preferred_order + sorted(process_cols)
        remaining_cols = sorted(list(header - set(sorted_header)))
        sorted_header.extend(remaining_cols)
        
        # Filter header to only include keys actually present
        final_header = [h for h in sorted_header if h in header]

        try:
            with open(csv_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=final_header)
                dict_writer.writeheader()
                dict_writer.writerows(self.simulation_data)
            log.info(f"Data successfully saved to {csv_path}")
        except IOError as e:
            log.error(f"Could not write to file {csv_path}: {e}")