"""
Water Treatment Plant Simulator: Cyber Attack Simulation

NEW: This version uses a timeline-based generation method to create
a balanced and randomized dataset for ML training.
"""
import datetime
import logging
import random
import math

log = logging.getLogger(__name__)

class AttackSimulator:
    """
    Builds and executes a pre-generated, randomized attack timeline
    to ensure all attack types are present and balanced in the dataset.
    """
    def __init__(self, plc_object, config):
        """
        Initializes the attack timeline.

        Args:
            plc_object (PLCLogic): The PLC instance to be attacked.
            config (dict): The full, loaded configuration YAML file.
        """
        self.plc = plc_object
        self.config = config
        self.active_attack = "None"
        self.active_scenario = None
        self.attack_timeline = []

        try:
            self._build_balanced_timeline()
        except Exception as e:
            log.error(f"FATAL: Could not build balanced attack timeline: {e}", exc_info=True)
            # Create an empty timeline to prevent a crash
            self.attack_timeline = []

    def _build_balanced_timeline(self):
        """
        Generates a balanced and shuffled timeline of attacks and normal behavior.
        """
        log.info("Building balanced attack timeline...")
        
        # Get simulation parameters
        total_duration_s = self.config['simulation']['duration_hours'] * 3600
        block_duration_s = self.config['balanced_attack_config']['attack_block_duration_seconds']
        normal_data_pct = self.config['balanced_attack_config']['normal_data_percentage']
        
        # Get the library of defined attacks
        attack_library = self.config['attack_scenarios']
        num_attack_types = len(attack_library)
        
        # Calculate number of blocks
        total_blocks = math.ceil(total_duration_s / block_duration_s)
        normal_blocks = int(total_blocks * normal_data_pct)
        attack_blocks_total = total_blocks - normal_blocks
        
        if num_attack_types == 0:
            log.warning("No attack scenarios defined. Simulation will be 'Normal' only.")
            self.attack_timeline = [(
                datetime.timedelta(seconds=0),
                datetime.timedelta(seconds=total_duration_s),
                None # 'None' indicates normal operation
            )]
            return

        # Calculate blocks per attack type
        blocks_per_attack = attack_blocks_total // num_attack_types
        remainder = attack_blocks_total % num_attack_types
        
        log.info(f"Timeline settings: {total_duration_s}s total, {total_blocks} blocks of {block_duration_s}s each.")
        log.info(f"{normal_blocks} 'Normal' blocks, {attack_blocks_total} 'Attack' blocks.")
        log.info(f"Distributing {blocks_per_attack} blocks per attack type, with {remainder} attacks getting one extra block.")

        # Create a "playlist" of all blocks
        playlist = []
        
        # Add normal blocks
        for _ in range(normal_blocks):
            playlist.append(None) # None = Normal
            
        # Add attack blocks
        for i in range(num_attack_types):
            num_blocks_for_this_attack = blocks_per_attack
            if i < remainder: # Distribute the remainder
                num_blocks_for_this_attack += 1
            
            scenario = attack_library[i]
            for _ in range(num_blocks_for_this_attack):
                playlist.append(scenario)
                
        # Shuffle the playlist to randomize the order
        random.shuffle(playlist)
        
        # Build the final timeline with start and end times
        current_time_s = 0
        for i, scenario in enumerate(playlist):
            start_time = datetime.timedelta(seconds=current_time_s)
            end_time = datetime.timedelta(seconds=current_time_s + block_duration_s)
            
            # Ensure the last block doesn't go over the total duration
            if end_time.total_seconds() > total_duration_s:
                end_time = datetime.timedelta(seconds=total_duration_s)
                
            self.attack_timeline.append((start_time, end_time, scenario))
            
            current_time_s += block_duration_s
            if current_time_s >= total_duration_s:
                break
                
        log.info(f"Successfully built timeline with {len(self.attack_timeline)} randomized blocks.")

    def update(self, current_sim_time_delta):
        """
        Checks the timeline and executes the scheduled attack for the current time step.
        Returns details of the attack if one is active.
        """
        self.active_attack = "None"
        self.active_scenario = None
        attack_details_to_return = None

        # Find the current block in the timeline
        for start_time, end_time, scenario in self.attack_timeline:
            if start_time <= current_sim_time_delta < end_time:
                if scenario:
                    # We are in an attack block
                    self.active_attack = scenario['name']
                    self.active_scenario = scenario
                    manipulated_registers = self._execute_attack(scenario)
                    attack_details_to_return = {
                        "name": scenario['name'],
                        "type": scenario['attack_type'],
                        "manipulated_registers": manipulated_registers
                    }
                # If scenario is 'None', we stay in 'Normal' mode
                break
        
        return attack_details_to_return


    def _execute_attack(self, scenario):
        """
        Performs the malicious action on the PLC's internal state.
        Returns a dictionary of the registers it manipulated and their new values.
        """
        attack_type = scenario['attack_type']
        target_reg_name = scenario['target_register']
        params = scenario['parameters']
        manipulated = {}

        if target_reg_name not in self.plc.state:
            log.warning(f"Attack scenario '{scenario['name']}' references an unknown register '{target_reg_name}'. Skipping.")
            return manipulated

        if attack_type == "set_value":
            value_to_set = int(params['value'])
            self.plc.state[target_reg_name] = value_to_set
            manipulated[target_reg_name] = value_to_set

        elif attack_type == "offset":
            original_value = self.plc.state.get(target_reg_name, 0)
            offset = int(params['offset'])
            new_value = max(0, original_value + offset) # Don't let values go negative
            self.plc.state[target_reg_name] = new_value
            manipulated[target_reg_name] = new_value
            
        elif attack_type == "man_in_the_middle_write":
            value_to_set = int(params['value'])
            self.plc.state[target_reg_name] = value_to_set
            manipulated[target_reg_name] = value_to_set

        elif attack_type == "intermittent_false_readings":
            if random.random() < params['probability']:
                false_value = int(params['false_value'])
                self.plc.state[target_reg_name] = false_value
                manipulated[target_reg_name] = false_value
        
        return manipulated