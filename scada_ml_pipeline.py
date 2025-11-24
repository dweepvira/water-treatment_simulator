import pandas as pd
import numpy as np
import matplotlib
# --- FIX 1: Force non-interactive backend to prevent Tkinter crashes ---
matplotlib.use('Agg') 
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
    GradientBoostingClassifier
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import warnings
import os
import joblib
from collections import deque

# --- FIX 2: Suppress Warnings ---
warnings.filterwarnings("ignore")
# Specifically ignore joblib resource tracker warnings which can be noisy
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

class SCADA_ML_Pipeline:
    """
    A comprehensive class to load, process, and model the SCADA dataset.
    
    UPGRADES IN V2:
    - Scenario 1B: Model Benchmarking (RF vs GBM vs LogReg) with ROC, PR Curves, and Bar Charts.
    - Scenario 6: Dimensionality Reduction (PCA) for visual separation
    - Scenario 7: Model Explainability (Permutation Importance)
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.full_clean_data = None
        self.le = LabelEncoder()
        self.scalers = {}
        self.models = {}
        self.feature_columns = []
        self.attack_class_names = []
        self.test_df_for_simulation = None
        self.simulation_initial_state = None
        
        # Configurable window size for stateful features
        self.window_size = 5 
        self.simulation_history = deque(maxlen=self.window_size)
        
        print(f"Pipeline V2 initialized for dataset: {self.csv_path}\n")

    def load_and_preprocess(self):
        print("--- 1. Loading and Preprocessing Data ---")
        try:
            data = pd.read_csv(self.csv_path, skipinitialspace=True)
            data.columns = data.columns.str.strip()
            print(f"Successfully loaded data. Shape: {data.shape}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Dataset file not found at {self.csv_path}")
            return False
        except Exception as e:
            print(f"FATAL ERROR: An error occurred while loading the data: {e}")
            return False

        # --- Target Engineering ---
        data["active_attack"] = data["active_attack"].fillna("None").astype(str)

        numeric_cols = [col for col in data.columns if col not in ["active_attack"]]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Binary Target
        data["is_attack"] = (data["active_attack"].str.strip() != "None").astype(int)
        
        # Multiclass Target
        data["attack_type"] = self.le.fit_transform(data["active_attack"].str.strip())
        self.attack_class_names = self.le.classes_

        data = data.fillna(method="bfill").fillna(method="ffill")
        self.full_clean_data = data
        return True

    def feature_engineer(self, data_copy):
        # (Same robust feature engineering as previous version)
        sensors = [
            "chlorine_level", "coagulant_pump_speed", "filter_backwash_pump_status",
            "filter_outlet_turbidity", "intake_pump_status", "raw_water_flow_rate",
            "raw_water_turbidity", "sedimentation_turbidity",
        ]
        sensors_for_rolling = [
            "raw_water_turbidity", "coagulant_pump_speed",
            "filter_outlet_turbidity", "chlorine_level"
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

        rolling_features = []
        for sensor in sensors_for_rolling:
            col_name = f"{sensor}_after_attack"
            if col_name in data_copy.columns:
                mean_col = f"{sensor}_rolling_mean_{self.window_size}"
                std_col = f"{sensor}_rolling_std_{self.window_size}"
                data_copy[mean_col] = data_copy[col_name].rolling(window=self.window_size).mean()
                data_copy[std_col] = data_copy[col_name].rolling(window=self.window_size).std()
                rolling_features.extend([mean_col, std_col])

        # Ratios
        data_copy["coag_to_raw_turb_ratio"] = data_copy["coagulant_pump_speed_after_attack"] / (data_copy["raw_water_turbidity_after_attack"] + 1e-6)
        data_copy["sed_effectiveness_ratio"] = data_copy["sedimentation_turbidity_after_attack"] / (data_copy["raw_water_turbidity_after_attack"] + 1e-6)
        data_copy["filter_effectiveness_ratio"] = data_copy["filter_outlet_turbidity_after_attack"] / (data_copy["sedimentation_turbidity_after_attack"] + 1e-6)

        ratio_features = ["coag_to_raw_turb_ratio", "sed_effectiveness_ratio", "filter_effectiveness_ratio"]
        feature_columns = sorted(list(set(base_features + ratio_features + rolling_features)))

        data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_copy = data_copy.fillna(method="bfill").fillna(method="ffill").fillna(0)

        if not self.feature_columns:
            self.feature_columns = feature_columns
        
        return data_copy, feature_columns

    def _get_temporal_split(self, data, feature_cols, target_col, train_split_pct=0.7):
        split_index = int(len(data) * train_split_pct)
        train_df = data.iloc[:split_index]
        test_df = data.iloc[split_index:]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        self.test_df_for_simulation = test_df.copy()
        self.simulation_initial_state = train_df.iloc[-self.window_size:].copy()

        return X_train, X_test, y_train, y_test

    # -------------------------------------------------------------------------
    # SCENARIO 1: Basic Anomaly Detection
    # -------------------------------------------------------------------------
    def run_scenario_1_anomaly_detection(self):
        print("\n--- 3. SCENARIO 1: Anomaly Detection (RF Tuning) ---")
        data_s1, features_s1 = self.feature_engineer(self.full_clean_data.copy())
        X_train, X_test, y_train, y_test = self._get_temporal_split(data_s1, features_s1, "is_attack")

        # --- FIX 3: Set n_jobs=1 to prevent semaphore leaks on small datasets ---
        rf_binary = RandomForestClassifier(random_state=42, n_jobs=1)
        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        
        grid_search = GridSearchCV(rf_binary, param_grid, cv=tscv, n_jobs=1, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
        joblib.dump(best_rf, 'anomaly_detector.joblib')
        self.models["anomaly_detector"] = best_rf

    # -------------------------------------------------------------------------
    # NEW SCENARIO 1B: Model Benchmarking
    # -------------------------------------------------------------------------
    def run_scenario_1b_model_comparison(self):
        """
        SCENARIO 1B: Benchmarking
        Compares Random Forest (Bagging), Gradient Boosting (Boosting), 
        and Logistic Regression (Linear).
        """
        print("\n--- 3B. SCENARIO 1B: Model Benchmarking (RF vs GBM vs LogReg) ---")
        
        data, features = self.feature_engineer(self.full_clean_data.copy())
        X_train, X_test, y_train, y_test = self._get_temporal_split(data, features, "is_attack")
        
        # Scale data for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000),
            "Random Forest (Bagging)": self.models.get("anomaly_detector", RandomForestClassifier(random_state=42, n_jobs=1)),
            "Gradient Boosting (Boosting)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }

        # Metrics storage
        model_metrics = []

        # Setup figures
        fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(10, 6))

        print(f"{'Model':<30} | {'AUC':<10} | {'Accuracy':<10} | {'F1-Score':<10}")
        print("-" * 70)

        for name, model in models.items():
            is_linear = "Logistic" in name
            X_tr = X_train_scaled if is_linear else X_train
            X_te = X_test_scaled if is_linear else X_test

            try:
                model.predict(X_te.iloc[:1] if not is_linear else X_te[:1])
            except:
                model.fit(X_tr, y_train)

            y_pred = model.predict(X_te)
            y_probs = model.predict_proba(X_te)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_test, y_probs)
            pr_auc = auc(recall, precision)
            
            acc = np.mean(y_pred == y_test)
            f1_val = f1_score(y_test, y_pred)
            
            model_metrics.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_val,
                "AUC": roc_auc
            })
            
            print(f"{name:<30} | {roc_auc:.4f}     | {acc:.4f}     | {f1_val:.4f}")

            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            ax_pr.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

        # Finalize ROC Plot
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve Comparison')
        ax_roc.legend(loc="lower right")
        fig_roc.savefig("scenario_1b_model_comparison_roc.png")
        print("Saved 'scenario_1b_model_comparison_roc.png'")
        plt.close(fig_roc)

        # Finalize PR Plot
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve Comparison')
        ax_pr.legend(loc="lower left")
        fig_pr.savefig("scenario_1b_model_comparison_pr.png")
        print("Saved 'scenario_1b_model_comparison_pr.png'")
        plt.close(fig_pr)

        # Finalize Bar Chart Plot
        metrics_df = pd.DataFrame(model_metrics)
        metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="viridis")
        plt.title("Model Performance Metrics Comparison")
        min_score = metrics_melted["Score"].min()
        plt.ylim(max(0, min_score - 0.1), 1.0) 
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("scenario_1b_model_comparison_bars.png")
        print("Saved 'scenario_1b_model_comparison_bars.png'")

    # -------------------------------------------------------------------------
    # SCENARIO 2: Unsupervised
    # -------------------------------------------------------------------------
    def run_scenario_2_unsupervised_detection(self):
        print("\n--- 4. SCENARIO 2: Unsupervised Anomaly Detection ---")
        data_s2, features_s2 = self.feature_engineer(self.full_clean_data.copy())
        X_train, X_test, y_train, y_test = self._get_temporal_split(data_s2, features_s2, "is_attack")
        
        X_train_normal = X_train[y_train == 0]
        if len(X_train_normal) == 0: return

        contamination = min(0.5, max(0.01, y_test.mean()))
        
        # --- FIX 3: Set n_jobs=1 ---
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
        iso.fit(X_train_normal)
        
        y_pred = iso.predict(X_test)
        y_pred_mapped = [1 if x == -1 else 0 for x in y_pred]
        
        print(classification_report(y_test, y_pred_mapped, target_names=["Normal", "Attack"]))

    # -------------------------------------------------------------------------
    # SCENARIO 3: Multiclass
    # -------------------------------------------------------------------------
    def run_scenario_3_attack_classification(self):
        print("\n--- 5. SCENARIO 3: Attack Classification ---")
        data_s3, features_s3 = self.feature_engineer(self.full_clean_data.copy())
        X_train, X_test, y_train, y_test = self._get_temporal_split(data_s3, features_s3, "attack_type")
        
        if len(np.unique(y_train)) <= 1: return

        # --- FIX 3: Set n_jobs=1 ---
        rf_multi = RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced")
        rf_multi.fit(X_train, y_train)
        
        y_pred = rf_multi.predict(X_test)
        
        all_labels = np.arange(len(self.attack_class_names))
        
        print(classification_report(
            y_test, 
            y_pred, 
            labels=all_labels, 
            target_names=self.attack_class_names, 
            zero_division=0
        ))
        
        joblib.dump(rf_multi, 'attack_classifier.joblib')
        self.models["attack_classifier"] = rf_multi
        
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.attack_class_names, yticklabels=self.attack_class_names)
        plt.tight_layout()
        plt.savefig("scenario_3_confusion_matrix.png")

    # -------------------------------------------------------------------------
    # SCENARIO 4: Process Integrity
    # -------------------------------------------------------------------------
    def run_scenario_4_process_integrity(self):
        print("\n--- 6. SCENARIO 4: Process Integrity (Physics Check) ---")
        data_s4, _ = self.feature_engineer(self.full_clean_data.copy())
        train_df = data_s4.iloc[:int(len(data_s4) * 0.7)]
        test_df = data_s4.iloc[int(len(data_s4) * 0.7):].copy()
        
        normal_train = train_df[train_df["is_attack"] == 0]
        if len(normal_train) == 0: return

        features = ["raw_water_turbidity_after_attack", "coagulant_pump_speed_after_attack"]
        target = "sedimentation_turbidity_after_attack"
        
        # --- FIX 3: Set n_jobs=1 ---
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf_reg.fit(normal_train[features], normal_train[target])
        self.models["process_integrity"] = rf_reg

        y_pred = rf_reg.predict(test_df[features])
        error = (test_df[target] - y_pred).abs()
        
        threshold = error.mean() + 3 * error.std()
        test_df["anomaly"] = (error > threshold).astype(int)
        
        print(classification_report(test_df["is_attack"], test_df["anomaly"], target_names=["Normal", "Attack"]))
        
        plt.figure(figsize=(15, 6))
        error.plot()
        plt.axhline(threshold, color='r', linestyle='--')
        plt.title("Physics Deviation (Regression Error)")
        plt.savefig("scenario_4_process_error.png")

    # -------------------------------------------------------------------------
    # SCENARIO 6: Dimensionality Reduction
    # -------------------------------------------------------------------------
    def run_scenario_6_pca_visualization(self):
        """
        SCENARIO 6: PCA Visualization
        """
        print("\n--- 8. SCENARIO 6: PCA Visualization (Data Separability) ---")
        
        data, features = self.feature_engineer(self.full_clean_data.copy())
        X_train, X_test, y_train, y_test = self._get_temporal_split(data, features, "attack_type")
        
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test_scaled)
        
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['label'] = y_test.values
        
        plt.figure(figsize=(12, 8))
        unique_labels = np.unique(y_test)
        
        for label in unique_labels:
            subset = pca_df[pca_df['label'] == label]
            label_name = self.attack_class_names[label]
            plt.scatter(subset['PC1'], subset['PC2'], label=label_name, alpha=0.6, s=15)
            
        plt.title(f'PCA: 2D Projection of Test Data (Explained Var: {np.sum(pca.explained_variance_ratio_):.2%})')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("scenario_6_pca_clusters.png")
        print("Saved 'scenario_6_pca_clusters.png'. Check this image to see how well attacks cluster.")

    # -------------------------------------------------------------------------
    # SCENARIO 7: Explainability
    # -------------------------------------------------------------------------
    def run_scenario_7_explainability(self):
        """
        SCENARIO 7: Model Explainability (Permutation Importance)
        """
        print("\n--- 9. SCENARIO 7: Explainability (Feature Contribution) ---")
        
        if "anomaly_detector" not in self.models:
            print("Skipping: Anomaly model not trained yet.")
            return

        model = self.models["anomaly_detector"]
        data, features = self.feature_engineer(self.full_clean_data.copy())
        X_train, X_test, y_train, y_test = self._get_temporal_split(data, features, "is_attack")

        subset_size = min(1000, len(X_test))
        X_subset = X_test[:subset_size]
        y_subset = y_test[:subset_size]

        print("Calculating Permutation Importance (this mimics SHAP but is faster)...")
        # --- FIX 3: Set n_jobs=1 ---
        result = permutation_importance(model, X_subset, y_subset, n_repeats=5, random_state=42, n_jobs=1)
        
        sorted_idx = result.importances_mean.argsort()[-10:] # Top 10 features
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(features)[sorted_idx]
        )
        plt.title("Top 10 Sensors Contributing to Attack Detection")
        plt.xlabel("Importance (Decrease in Accuracy if Shuffled)")
        plt.tight_layout()
        plt.savefig("scenario_7_explainability.png")
        print("Saved 'scenario_7_explainability.png'. This tells operators which sensors to check first.")

    # -------------------------------------------------------------------------
    # SIMULATION
    # -------------------------------------------------------------------------
    def _engineer_single_datapoint(self, current_row_dict):
        self.simulation_history.append(current_row_dict)
        if len(self.simulation_history) < self.window_size:
            return pd.DataFrame(columns=self.feature_columns, data=[[0]*len(self.feature_columns)])

        hist_df = pd.DataFrame(list(self.simulation_history))
        record = {}
        
        sensors = ["chlorine_level", "coagulant_pump_speed", "filter_backwash_pump_status",
                   "filter_outlet_turbidity", "intake_pump_status", "raw_water_flow_rate",
                   "raw_water_turbidity", "sedimentation_turbidity"]
        
        cur = hist_df.iloc[-1]
        last = hist_df.iloc[-2]
        for s in sensors:
            c = f"{s}_after_attack"
            p = f"{s}_before_update"
            record[c] = cur.get(c, 0)
            record[f"delta_{s}"] = cur.get(c, 0) - last.get(p, 0)

        rolling_sensors = ["raw_water_turbidity", "coagulant_pump_speed", "filter_outlet_turbidity", "chlorine_level"]
        for s in rolling_sensors:
            col = f"{s}_after_attack"
            if col in hist_df.columns:
                record[f"{s}_rolling_mean_{self.window_size}"] = hist_df[col].rolling(self.window_size).mean().iloc[-1]
                record[f"{s}_rolling_std_{self.window_size}"] = hist_df[col].rolling(self.window_size).std().iloc[-1]

        record["coag_to_raw_turb_ratio"] = record.get("coagulant_pump_speed_after_attack", 0) / (record.get("raw_water_turbidity_after_attack", 0) + 1e-6)
        record["sed_effectiveness_ratio"] = record.get("sedimentation_turbidity_after_attack", 0) / (record.get("raw_water_turbidity_after_attack", 0) + 1e-6)
        record["filter_effectiveness_ratio"] = record.get("filter_outlet_turbidity_after_attack", 0) / (record.get("sedimentation_turbidity_after_attack", 0) + 1e-6)

        df = pd.DataFrame([record]).reindex(columns=self.feature_columns, fill_value=0)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df.fillna(0)

    def simulate_real_time_alerting(self):
        print("\n--- 10. SCENARIO 5: Real-Time Alerting Simulation ---")
        try:
            anomaly_model = joblib.load('anomaly_detector.joblib')
            classifier_model = joblib.load('attack_classifier.joblib')
        except:
            print("Models not found. Run training first.")
            return

        self.simulation_history.clear()
        for _, row in self.simulation_initial_state.iterrows():
            self.simulation_history.append(row.to_dict())

        alerts_found = 0
        alert_log = [] # New list to store alert data
        
        print(f"Streaming {len(self.test_df_for_simulation)} datapoints...")
        
        for i, (idx, row) in enumerate(self.test_df_for_simulation.iterrows()):
            feat = self._engineer_single_datapoint(row.to_dict())
            
            if anomaly_model.predict(feat)[0] == 1:
                alerts_found += 1
                
                # Get details for the log
                pred_label_idx = classifier_model.predict(feat)[0]
                atype = self.attack_class_names[pred_label_idx]
                conf = anomaly_model.predict_proba(feat)[0][1]
                
                # Append to alert log
                alert_log.append({
                    "Timestamp_Index": i,
                    "Original_CSV_Index": idx,
                    "Predicted_Type": atype,
                    "Confidence": conf,
                    "Actual_Type": row['active_attack']
                })
                
                if alerts_found <= 3:
                    print(f"ALERT [T={i}]: {atype} (Conf: {conf:.2%})")

        print(f"Total Alerts: {alerts_found}")
        
        # Save alerts to CSV
        if alert_log:
            pd.DataFrame(alert_log).to_csv("real_time_alerts.csv", index=False)
            print("Alerts saved to 'real_time_alerts.csv'")

    def run_pipeline(self):
        if not self.load_and_preprocess(): return
        
        # Original Scenarios
        self.run_scenario_1_anomaly_detection()
        
        self.run_scenario_1b_model_comparison()
        
        self.run_scenario_2_unsupervised_detection()
        self.run_scenario_3_attack_classification()
        self.run_scenario_4_process_integrity()
        
        # Visualizations
        self.run_scenario_6_pca_visualization()
        self.run_scenario_7_explainability()
        
        self.simulate_real_time_alerting()
        
        print("\n--- Pipeline Finished. Check the 5 generated PNG files. ---")

if __name__ == "__main__":
    DATASET_PATH = "Phase 1/Simulator/datasets/6_hour_balanced_simulation.csv"
    pipeline = SCADA_ML_Pipeline(csv_path=DATASET_PATH)
    pipeline.run_pipeline()