import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
import warnings
import os
import joblib
from collections import deque

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class SCADA_ML_Pipeline:
    """
    A comprehensive class to load, process, and model the SCADA dataset
    for various classification and prediction scenarios.
    
    NEW: Now includes stateful feature engineering with rolling windows.
    """

    def __init__(self, csv_path):
        """
        Initializes the pipeline with the path to the dataset.

        Args:
            csv_path (str): The file path to the 3-hour simulation CSV.
        """
        self.csv_path = csv_path
        self.full_clean_data = None  # Original data is preserved here
        self.le = LabelEncoder()
        self.scalers = {}
        self.models = {}
        self.feature_columns = []
        self.attack_class_names = []
        self.test_df_for_simulation = None  # To hold test data for scenario 5
        self.simulation_initial_state = None # To hold the last row of training data
        
        # --- NEW: Configurable window size for stateful features ---
        self.window_size = 5 
        self.simulation_history = deque(maxlen=self.window_size) # Holds state for simulation
        
        print(f"Pipeline initialized for dataset: {self.csv_path}\n")

    def load_and_preprocess(self):
        """
        Loads the dataset and performs initial preprocessing and target engineering.
        The cleaned, full dataset is stored in `self.full_clean_data`.
        """
        print("--- 1. Loading and Preprocessing Data ---")
        try:
            # Load data, stripping any leading/trailing whitespace from column names
            # Use skipinitialspace=True to handle whitespace after delimiters
            data = pd.read_csv(self.csv_path, skipinitialspace=True)
            data.columns = data.columns.str.strip()
            print(f"Successfully loaded data. Shape: {data.shape}")

        except FileNotFoundError:
            print(f"FATAL ERROR: Dataset file not found at {self.csv_path}")
            return False
        except Exception as e:
            print(f"FATAL ERROR: An error occurred while loading the data: {e}")
            return False

        # --- Target Variable Engineering ---
        data["active_attack"] = data["active_attack"].fillna("None").astype(str)

        # Force all other columns to be numeric
        numeric_cols = [col for col in data.columns if col not in ["active_attack"]]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Scenario 1 & 2 Target: Binary Anomaly Detection (Normal vs. Attack)
        data["is_attack"] = (data["active_attack"].str.strip() != "None").astype(
            int
        )
        print("\nBinary Target 'is_attack' created:")
        print(data["is_attack"].value_counts(normalize=True).to_frame())

        # Scenario 3 Target: Multiclass Attack Classification
        data["attack_type"] = self.le.fit_transform(
            data["active_attack"].str.strip()
        )
        print("\nMulticlass Target 'attack_type' created:")
        print(
            pd.Series(
                self.le.inverse_transform(np.sort(data["attack_type"].unique())),
                name="Attack Classes",
            ).to_frame()
        )
        self.attack_class_names = self.le.classes_

        # Handle any potential NaN values
        data = data.fillna(method="bfill")
        data = data.fillna(method="ffill")

        # Store the full, cleaned data
        self.full_clean_data = data
        print("\nPreprocessing complete. Full, clean dataset is preserved.")
        return True

    def feature_engineer(self, data_copy):
        """
        Creates new features on a *copy* of the data.
        Returns the modified DataFrame and the list of feature columns.
        """
        print("Engineering features on data copy...")

        sensors = [
            "chlorine_level", "coagulant_pump_speed", "filter_backwash_pump_status",
            "filter_outlet_turbidity", "intake_pump_status", "raw_water_flow_rate",
            "raw_water_turbidity", "sedimentation_turbidity",
        ]
        
        # --- NEW: Define key sensors for rolling features ---
        sensors_for_rolling = [
            "raw_water_turbidity",
            "coagulant_pump_speed",
            "filter_outlet_turbidity",
            "chlorine_level"
        ]
        
        base_features = []
        
        for sensor in sensors:
            current_col = f"{sensor}_after_attack"
            prev_col = f"{sensor}_before_update"
            delta_col = f"delta_{sensor}"

            if current_col in data_copy.columns and prev_col in data_copy.columns:
                data_copy[delta_col] = data_copy[current_col] - data_copy[prev_col]
                base_features.append(delta_col)
            if current_col in data_copy.columns:
                base_features.append(current_col)

        # --- NEW: Create Rolling Window Features ---
        rolling_features = []
        for sensor in sensors_for_rolling:
            col_name = f"{sensor}_after_attack"
            if col_name in data_copy.columns:
                mean_col = f"{sensor}_rolling_mean_{self.window_size}"
                std_col = f"{sensor}_rolling_std_{self.window_size}"
                
                data_copy[mean_col] = data_copy[col_name].rolling(window=self.window_size).mean()
                data_copy[std_col] = data_copy[col_name].rolling(window=self.window_size).std()
                
                rolling_features.extend([mean_col, std_col])

        # Create Process-Based Ratio Features
        data_copy["coag_to_raw_turb_ratio"] = data_copy[
            "coagulant_pump_speed_after_attack"
        ] / (data_copy["raw_water_turbidity_after_attack"] + 1e-6)
        data_copy["sed_effectiveness_ratio"] = data_copy[
            "sedimentation_turbidity_after_attack"
        ] / (data_copy["raw_water_turbidity_after_attack"] + 1e-6)
        data_copy["filter_effectiveness_ratio"] = data_copy[
            "filter_outlet_turbidity_after_attack"
        ] / (data_copy["sedimentation_turbidity_after_attack"] + 1e-6)

        ratio_features = [
            "coag_to_raw_turb_ratio",
            "sed_effectiveness_ratio",
            "filter_effectiveness_ratio",
        ]

        # Define the final feature set
        feature_columns = sorted(list(set(base_features + ratio_features + rolling_features)))

        # Final check for NaNs/Infs
        data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaNs from rolling windows (at the start)
        data_copy = data_copy.fillna(method="bfill")
        data_copy = data_copy.fillna(method="ffill")
        data_copy = data_copy.fillna(0) # Fill any remaining

        # Store feature columns for later use (e.g., in simulation)
        if not self.feature_columns:
            self.feature_columns = feature_columns
            print(f"Feature engineering complete. Total features: {len(self.feature_columns)}")
            print("Final features being used for modeling:")
            for f in self.feature_columns:
                print(f"  - {f}")
        
        return data_copy, feature_columns

    def _get_temporal_split(self, data, feature_cols, target_col, train_split_pct=0.7):
        """
        Splits the data into train/test sets based on time, not randomly.
        """
        print(f"Performing temporal split: {train_split_pct*100}% train, {(1-train_split_pct)*100}% test")
        
        split_index = int(len(data) * train_split_pct)
        
        train_df = data.iloc[:split_index]
        test_df = data.iloc[split_index:]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Save the test dataframe for the real-time simulation
        self.test_df_for_simulation = test_df.copy()
        
        # --- NEW: Save the last `window_size` rows for priming the simulation ---
        self.simulation_initial_state = train_df.iloc[-self.window_size:].copy()

        return X_train, X_test, y_train, y_test

    def _get_scaled_data(self, X_train, X_test, key="default"):
        """Helper to scale data and save the scaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[key] = scaler
        joblib.dump(scaler, f"{key}_scaler.joblib")
        print(f"Saved '{key}_scaler.joblib'")
        return X_train_scaled, X_test_scaled

    def run_scenario_1_anomaly_detection(self):
        """
        SCENARIO 1: Binary Classification (Normal vs. Attack)
        Trains, tunes, and saves a model to detect if an attack is happening.
        """
        print("\n--- 3. SCENARIO 1: Anomaly Detection (Normal vs. Attack) ---")
        
        # Work on a copy of the data
        data_s1, features_s1 = self.feature_engineer(self.full_clean_data.copy())
        
        X_train, X_test, y_train, y_test = self._get_temporal_split(
            data_s1, features_s1, "is_attack"
        )

        # --- Model: Random Forest with Hyperparameter Tuning ---
        print("\n=== Model: Tuning Random Forest Classifier ===")
        rf_binary = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_leaf': [2, 5]
        }
        
        grid_search = GridSearchCV(
            estimator=rf_binary,
            param_grid=param_grid,
            cv=tscv,
            n_jobs=-1,
            scoring='f1',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters found: {grid_search.best_params_}")
        best_rf_binary = grid_search.best_estimator_

        # Evaluate the *best* model on the unseen test set
        y_pred_rf = best_rf_binary.predict(X_test)
        print("\nClassification Report (Tuned Random Forest on Test Set):")
        print(classification_report(y_test, y_pred_rf, target_names=["Normal", "Attack"]))
        
        # Save the trained model
        joblib.dump(best_rf_binary, 'anomaly_detector.joblib')
        print("Saved 'anomaly_detector.joblib' to disk.")
        self.models["anomaly_detector"] = best_rf_binary

        # Plot Feature Importance
        plt.figure(figsize=(10, 10)) # Made taller for new features
        feat_importances = (
            pd.Series(best_rf_binary.feature_importances_, index=self.feature_columns)
            .sort_values(ascending=False)
        )
        sns.barplot(x=feat_importances, y=feat_importances.index)
        plt.title("Feature Importance for Anomaly Detection (Scenario 1)")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("scenario_1_feature_importance.png")
        print("\nSaved 'scenario_1_feature_importance.png'")

    def run_scenario_2_unsupervised_detection(self):
        """
        SCENARIO 2: Unsupervised Anomaly Detection (Isolation Forest)
        Trains on *normal training data* and evaluates on the *test set*.
        """
        print(
            "\n--- 4. SCENARIO 2: Unsupervised Anomaly Detection (Isolation Forest) ---"
        )
        
        data_s2, features_s2 = self.feature_engineer(self.full_clean_data.copy())
        
        X_train, X_test, y_train, y_test = self._get_temporal_split(
            data_s2, features_s2, "is_attack"
        )
        
        X_train_normal = X_train[y_train == 0]

        if len(X_train_normal) == 0:
            print(
                "FATAL ERROR: No 'Normal' data found in the training split. Skipping Scenario 2."
            )
            return

        contamination_rate = y_test.mean()

        if contamination_rate >= 0.5:
            print(
                f"WARNING: Test set contamination rate is {contamination_rate:.2f}, which is >= 0.5."
            )
            print("Capping contamination parameter at 0.5 for IsolationForest.")
            contamination_rate = 0.5
        
        if contamination_rate == 0.0:
            print("WARNING: No contamination found in test set. Using a small value (0.01).")
            contamination_rate = 0.01

        print(f"Training Isolation Forest on {len(X_train_normal)} 'Normal' training samples...")
        iso_forest = IsolationForest(
            contamination=contamination_rate, random_state=42, n_jobs=-1
        )
        iso_forest.fit(X_train_normal)

        # Evaluate on the *test set*
        y_pred_iso = iso_forest.predict(X_test)
        y_pred_iso_mapped = np.array([1 if p == -1 else 0 for p in y_pred_iso])

        print("\nClassification Report (Isolation Forest on Test Set):")
        print(
            classification_report(
                y_test, y_pred_iso_mapped, target_names=["Normal", "Attack"]
            )
        )

    def run_scenario_3_attack_classification(self):
        """
        SCENARIO 3: Multiclass Classification
        Trains, tunes, and saves a model to identify the *specific type* of attack.
        """
        print("\n--- 5. SCENARIO 3: Attack Type Classification (Multiclass) ---")
        
        data_s3, features_s3 = self.feature_engineer(self.full_clean_data.copy())
        
        X_train, X_test, y_train, y_test = self._get_temporal_split(
            data_s3, features_s3, "attack_type"
        )
        
        if len(np.unique(y_train)) <= 1:
            print(
                "FATAL ERROR: Not enough class variety in training data. Skipping Scenario 3."
            )
            return

        # --- Model: Random Forest with Hyperparameter Tuning & Class Weights ---
        print("\n=== Model: Tuning Random Forest Classifier (Multiclass) ===")
        # --- NEW: Added class_weight='balanced' for robustness ---
        rf_multi = RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        tscv = TimeSeriesSplit(n_splits=3)
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
        }
        
        grid_search = GridSearchCV(
            estimator=rf_multi,
            param_grid=param_grid,
            cv=tscv,
            n_jobs=-1,
            scoring='f1_weighted',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters found: {grid_search.best_params_}")
        best_rf_multi = grid_search.best_estimator_

        # Evaluate the *best* model on the unseen test set
        y_pred_rf = best_rf_multi.predict(X_test)
        
        all_labels = np.arange(len(self.attack_class_names))

        print("\nClassification Report (Tuned Random Forest Multiclass on Test Set):")
        print(
            classification_report(
                y_test,
                y_pred_rf,
                labels=all_labels,
                target_names=self.attack_class_names,
                zero_division=0 
            )
        )

        # Save the trained model
        joblib.dump(best_rf_multi, 'attack_classifier.joblib')
        print("Saved 'attack_classifier.joblib' to disk.")
        self.models["attack_classifier"] = best_rf_multi

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_rf, labels=all_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.attack_class_names,
            yticklabels=self.attack_class_names,
        )
        plt.title("Confusion Matrix for Attack Type Classification (Scenario 3)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("scenario_3_confusion_matrix.png")
        print("\nSaved 'scenario_3_confusion_matrix.png'")

    def run_scenario_4_process_integrity(self):
        """
        SCENARIO 4: Regression to Detect Process Anomalies
        Trains on *normal training data* and evaluates on the *test set*.
        """
        print("\n--- 6. SCENARIO 4: Process Integrity (Regression) ---")
        
        # Note: We don't need all the engineered features for this one.
        data_s4, _ = self.feature_engineer(self.full_clean_data.copy())
        
        train_df = data_s4.iloc[:int(len(data_s4) * 0.7)]
        test_df = data_s4.iloc[int(len(data_s4) * 0.7):].copy()
        
        normal_train_data = train_df[train_df["is_attack"] == 0]

        if len(normal_train_data) == 0:
            print(
                "FATAL ERROR: No 'Normal' data in training split. Skipping Scenario 4."
            )
            return

        features_reg = [
            "raw_water_turbidity_after_attack",
            "coagulant_pump_speed_after_attack",
        ]
        target_reg = "sedimentation_turbidity_after_attack"

        X_train_reg = normal_train_data[features_reg]
        y_train_reg = normal_train_data[target_reg]

        print(
            f"Training Process Integrity Model on {len(X_train_reg)} normal training samples..."
        )

        # --- Model: Random Forest Regressor ---
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train_reg, y_train_reg)
        
        # Save the trained model
        joblib.dump(rf_reg, 'process_integrity_model.joblib')
        print("Saved 'process_integrity_model.joblib' to disk.")
        self.models["process_integrity"] = rf_reg

        # Evaluate R-squared on normal training data
        y_pred_train = rf_reg.predict(X_train_reg)
        r2 = r2_score(y_train_reg, y_pred_train)
        print(f"\nModel R-squared on normal training data: {r2:.4f}")
        print("This shows the model has learned the physical process rules.")

        # --- Use model for anomaly detection on the test set ---
        X_test_all = test_df[features_reg]
        y_true_all = test_df[target_reg]
        y_pred_all = rf_reg.predict(X_test_all)

        test_df["process_error"] = (y_true_all - y_pred_all).abs()

        # Calculate threshold from *normal training data*
        normal_error = (y_train_reg - y_pred_train).abs()
        error_threshold = normal_error.mean() + 3 * normal_error.std()
        print(f"Anomaly error threshold set at: {error_threshold:.4f}")

        test_df["process_anomaly"] = (
            test_df["process_error"] > error_threshold
        ).astype(int)

        print("\nProcess Integrity Anomaly Report (on Test Set):")
        print(
            classification_report(
                test_df["is_attack"],  # True label
                test_df["process_anomaly"],  # Predicted label
                target_names=["Normal", "Attack"],
            )
        )

        # Plot the error over time *for the test set only*
        plt.figure(figsize=(15, 7))
        test_df["process_error"].plot(label="Prediction Error (Anomaly Score)")
        plt.axhline(
            error_threshold,
            color="r",
            linestyle="--",
            label=f"Anomaly Threshold ({error_threshold:.2f})",
        )

        attack_periods = test_df[test_df["is_attack"] == 1]
        for start, end in self._find_consecutive_ranges(attack_periods.index):
            plt.axvspan(start, end, color="red", alpha=0.2, label="_nolegend_")

        plt.title("Process Integrity Anomaly Score Over Time (Test Set Only)")
        plt.xlabel("Data Point Index (Time)")
        plt.ylabel("Absolute Prediction Error (NTU)")
        plt.legend()
        plt.savefig("scenario_4_process_error_timeline.png")
        print("\nSaved 'scenario_4_process_error_timeline.png'")

    @staticmethod
    def _find_consecutive_ranges(data):
        """Helper to find continuous blocks in a list of indices."""
        from itertools import groupby
        from operator import itemgetter

        ranges = []
        for k, g in groupby(enumerate(data), lambda i_x: i_x[0] - i_x[1]):
            group = list(map(itemgetter(1), g))
            ranges.append((group[0], group[-1]))
        return ranges

    def _engineer_single_datapoint(self, current_row_dict):
        """
        Applies stateful feature engineering to a single new datapoint.
        It uses and updates the `self.simulation_history` deque.
        
        Args:
            current_row_dict (dict): The new row of data as a dictionary.
        
        Returns:
            pd.DataFrame: A single-row DataFrame with all engineered features.
        """
        
        # Add current data to our history
        self.simulation_history.append(current_row_dict)
        
        # If history isn't full, we can't calculate rolling features yet
        if len(self.simulation_history) < self.window_size:
            # Return an empty/zeroed DataFrame of the correct shape
            return pd.DataFrame(columns=self.feature_columns, data=[[0]*len(self.feature_columns)])

        # Convert history to DataFrame for easy processing
        hist_df = pd.DataFrame(list(self.simulation_history))
        
        record = {}
        sensors = [
            "chlorine_level", "coagulant_pump_speed", "filter_backwash_pump_status",
            "filter_outlet_turbidity", "intake_pump_status", "raw_water_flow_rate",
            "raw_water_turbidity", "sedimentation_turbidity",
        ]
        sensors_for_rolling = [
            "raw_water_turbidity", "coagulant_pump_speed",
            "filter_outlet_turbidity", "chlorine_level"
        ]

        # Get the current and previous rows
        current_row = hist_df.iloc[-1]
        last_row = hist_df.iloc[-2]

        for sensor in sensors:
            current_col = f"{sensor}_after_attack"
            prev_col = f"{sensor}_before_update"
            delta_col = f"delta_{sensor}"
            
            current_val = current_row.get(current_col, 0)
            prev_val = last_row.get(prev_col, 0) # Use previous row for "before"

            record[current_col] = current_val
            record[delta_col] = current_val - prev_val

        # Calculate Rolling Features on the history DataFrame
        for sensor in sensors_for_rolling:
            col_name = f"{sensor}_after_attack"
            if col_name in hist_df.columns:
                mean_col = f"{sensor}_rolling_mean_{self.window_size}"
                std_col = f"{sensor}_rolling_std_{self.window_size}"
                
                record[mean_col] = hist_df[col_name].rolling(window=self.window_size).mean().iloc[-1]
                record[std_col] = hist_df[col_name].rolling(window=self.window_size).std().iloc[-1]

        # Create Process-Based Ratio Features for the current row
        record["coag_to_raw_turb_ratio"] = record.get(
            "coagulant_pump_speed_after_attack", 0
        ) / (record.get("raw_water_turbidity_after_attack", 0) + 1e-6)
        
        record["sed_effectiveness_ratio"] = record.get(
            "sedimentation_turbidity_after_attack", 0
        ) / (record.get("raw_water_turbidity_after_attack", 0) + 1e-6)
        
        record["filter_effectiveness_ratio"] = record.get(
            "filter_outlet_turbidity_after_attack", 0
        ) / (record.get("sedimentation_turbidity_after_attack", 0) + 1e-6)

        # Create a DataFrame and ensure all columns are present in the correct order
        df = pd.DataFrame([record])
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Final check for NaNs/Infs
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df = df.fillna(0)

        return df

    def simulate_real_time_alerting(self):
        """
        Simulates a real-time alerting service using the saved models.
        It loads the models from disk and processes the test set
        one datapoint at a time, managing state.
        """
        print("\n--- 7. SCENARIO 5: Real-Time Alerting Simulation (Inference) ---")

        # --- Load models from disk (simulating an online service) ---
        try:
            anomaly_model = joblib.load('anomaly_detector.joblib')
            classifier_model = joblib.load('attack_classifier.joblib')
            print("Successfully loaded 'anomaly_detector.joblib' and 'attack_classifier.joblib'.")
        except FileNotFoundError:
            print("FATAL ERROR: Model files not found. Run Scenarios 1 & 3 first.")
            return

        if self.test_df_for_simulation is None or self.simulation_initial_state is None:
            print("FATAL ERROR: No test data or initial state. Run a training scenario first.")
            return

        print(f"Simulating real-time alerts for all {len(self.test_df_for_simulation)} data points in the test set...")
        
        # --- Initialize State ---
        # NEW: Prime the simulation history with the last rows from the train set
        self.simulation_history.clear()
        for _, row in self.simulation_initial_state.iterrows():
            self.simulation_history.append(row.to_dict())
        
        print(f"Simulation state primed with {len(self.simulation_history)} historical data points.")

        alerts_found = 0
        alert_log = [] # Initialize an empty list for the log
        
        # --- Simulation Loop (Iterate over the test set) ---
        for i, (idx, current_point_row) in enumerate(self.test_df_for_simulation.iterrows()):
            
            # 1. Engineer features for the new datapoint
            # This function now uses and updates self.simulation_history
            features_df = self._engineer_single_datapoint(current_point_row.to_dict())

            # 2. Predict anomaly
            anomaly_pred = anomaly_model.predict(features_df)[0] # 0 = Normal, 1 = Attack

            # 3. Issue Alert
            if anomaly_pred == 1:
                alerts_found += 1
                attack_type_pred = classifier_model.predict(features_df)[0]
                attack_name = self.attack_class_names[attack_type_pred]
                confidence = anomaly_model.predict_proba(features_df)[0][1] # Get confidence

                # Create a log entry
                alert_entry = {
                    "Test_Datapoint_Num": i + 1,
                    "Original_Data_Index": idx,
                    "Predicted_Attack_Type": attack_name,
                    "Prediction_Confidence": f"{confidence:.2%}",
                    "Ground_Truth_Attack": current_point_row["active_attack"],
                    "Is_True_Attack": current_point_row["is_attack"]
                }
                alert_log.append(alert_entry)
                
                # Print a detailed alert for the *first 5 alerts*
                if alerts_found <= 5:
                    print("\n***********************************")
                    print(f"! A L E R T (Datapoint {i+1} at index {idx}) : POTENTIAL ATTACK DETECTED !")
                    print(f"  Classified Attack: {attack_name}")
                    print(f"  Confidence:        {confidence:.2%}")
                    print(f"  (Ground Truth was: {current_point_row['active_attack']})")
                    print("***********************************")
                elif alerts_found == 6:
                     print("\n... (suppressing further alert details for brevity) ...")

            # 4. State is already updated inside _engineer_single_datapoint
        
        print("\n--- Real-Time Simulation Complete ---")
        print(f"Total datapoints processed: {len(self.test_df_for_simulation)}")
        print(f"Total alerts generated:     {alerts_found}")
        
        # Optional: Final check against ground truth
        true_attacks = self.test_df_for_simulation['is_attack'].sum()
        print(f"(Ground Truth: There were {true_attacks} attack datapoints in the test set)")

        # Save the alert log to a CSV file
        if alert_log:
            alert_df = pd.DataFrame(alert_log)
            alert_df.to_csv("real_time_alert_log.csv", index=False)
            print("\nSuccessfully saved all detected alerts to 'real_time_alert_log.csv'")
        else:
            print("\nNo alerts were generated during the simulation.")


    def run_pipeline(self):
        """
        Executes the full ML pipeline from start to finish.
        """
        # 1. Load and clean the data
        if not self.load_and_preprocess():
            return
        
        # 2. Run training/evaluation scenarios
        self.run_scenario_1_anomaly_detection()
        self.run_scenario_2_unsupervised_detection()
        self.run_scenario_3_attack_classification()
        self.run_scenario_4_process_integrity()
        
        # 3. Run the final inference simulation using the models saved by the scenarios
        self.simulate_real_time_alerting()
        
        print("\n--- ML Pipeline Finished Successfully ---")
        print(
            "Generated plots: 'scenario_1_feature_importance.png', 'scenario_3_confusion_matrix.png', 'scenario_4_process_error_timeline.png'"
        )


if __name__ == "__main__":
    # Ensure the CSV file is in the same directory as this script,
    # or provide the full path.
    DATASET_PATH = "Phase 1/Simulator/datasets/6_hour_balanced_simulation.csv"

    pipeline = SCADA_ML_Pipeline(csv_path=DATASET_PATH)
    pipeline.run_pipeline()