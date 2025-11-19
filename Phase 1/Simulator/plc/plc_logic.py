"""
Water Treatment Plant Simulator: PLC Logic and Process Simulation

This module simulates the physical water treatment process and the
control logic that a real PLC would execute.
"""
import random

class PLCLogic:
    """
    Simulates the water treatment process and control logic.
    It manages the plant's state internally in a dictionary.
    """
    def __init__(self, register_map, params):
        self.r_map = register_map
        self.params = params
        self.stuck_sensors = {}
        
        # Initialize all registers to 0, with some defaults
        self.state = {name: 0 for name in self.r_map.keys()}
        self.state["intake_pump_status"] = 1 # Start with the pump on
        self.state["raw_water_flow_rate"] = self.params['flow_rate_normal']
        self.state["flocculator_speed"] = 100 # Constant speed
        self.state["chlorine_level"] = 1.5

    def update(self, current_time):
        """
        Executes one step of the simulation logic.
        Returns a tuple of (read_registers, written_registers) for logging.
        """
        read_registers = {}
        written_registers = {}

        # --- 1. SIMULATE EXTERNAL CONDITIONS (RAW WATER) ---
        base_turbidity = self.params['raw_water_turbidity_normal'] + random.uniform(-5, 5)
        if random.random() < 0.01: # Chance of a random spike
            base_turbidity += random.uniform(50, 150)
        self.state["raw_water_turbidity"] = int(base_turbidity)

        # --- 2. FAULTY SENSOR SIMULATION ---
        for name in self.r_map:
            if 'turbidity' in name and random.random() < self.params['faulty_reading_chance']:
                if name not in self.stuck_sensors:
                    current_value = self.state.get(name, 0)
                    stuck_duration = random.randint(10, 50)
                    self.stuck_sensors[name] = {"value": current_value, "steps_left": stuck_duration}

        for name in list(self.stuck_sensors.keys()):
            self.stuck_sensors[name]["steps_left"] -= 1
            if self.stuck_sensors[name]["steps_left"] <= 0:
                del self.stuck_sensors[name]
            else:
                self.state[name] = self.stuck_sensors[name]["value"]

        # --- 3. READ SENSOR VALUES FROM INTERNAL STATE (Logging which are read) ---
        raw_turbidity = self.state["raw_water_turbidity"]
        read_registers['raw_water_turbidity'] = raw_turbidity
        
        filter_outlet_turbidity = self.state["filter_outlet_turbidity"]
        read_registers['filter_outlet_turbidity'] = filter_outlet_turbidity

        # --- 4. EXECUTE CONTROL LOGIC (Logging which actuators are written) ---
        
        # Coagulant pump logic
        if raw_turbidity > self.params['raw_water_turbidity_high_threshold']:
            new_speed = self.params['coagulant_pump_speed_high']
        else:
            new_speed = self.params['coagulant_pump_speed_normal']
        
        if self.state["coagulant_pump_speed"] != new_speed:
            self.state["coagulant_pump_speed"] = new_speed
            written_registers['coagulant_pump_speed'] = new_speed

        # Backwash logic
        if filter_outlet_turbidity > self.params['filter_turbidity_backwash_threshold']:
            new_status = 1
        else:
            new_status = 0
            
        if self.state["filter_backwash_pump_status"] != new_status:
            self.state["filter_backwash_pump_status"] = new_status
            written_registers['filter_backwash_pump_status'] = new_status


        # --- 5. SIMULATE THE PHYSICAL PROCESS ---
        coag_pump_speed = self.state["coagulant_pump_speed"]
        coag_pump_high = self.params['coagulant_pump_speed_high']
        coag_effectiveness_ratio = coag_pump_speed / coag_pump_high if coag_pump_high > 0 else 0
        coag_effectiveness = coag_effectiveness_ratio * random.uniform(0.85, 0.95)

        turbidity_after_sed = raw_turbidity * (1 - coag_effectiveness)
        self.state["sedimentation_turbidity"] = int(turbidity_after_sed)

        # Filtration effectiveness decreases if backwash is needed but not active
        filter_effectiveness = 0.9 if self.state["filter_backwash_pump_status"] == 0 else 0.5
        final_turbidity = turbidity_after_sed * (1 - filter_effectiveness) * random.uniform(0.9, 1.1)
        self.state["filter_outlet_turbidity"] = int(final_turbidity)

        # Update final chlorine based on turbidity
        self.state["chlorine_level"] = max(1.0, 2.5 - (final_turbidity / 10.0))
        
        return read_registers, written_registers
