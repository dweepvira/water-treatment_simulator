Pipeline initialized for dataset: Phase 1/Simulator/datasets/3_hour_random_attack_simulation.csv

--- 1. Loading and Preprocessing Data ---
Successfully loaded data. Shape: (2160, 36)

Binary Target 'is_attack' created:
           proportion
is_attack            
1            0.569444
0            0.430556

Multiclass Target 'attack_type' created:
                          Attack Classes
0      False Data Injection (High Value)
1         Intermittent Command Injection
2  Man-in-the-Middle (Data Manipulation)
3                                   None
4   Process Denial of Service (Actuator)
5    Stealthy Data Manipulation (Offset)

Preprocessing complete. Full, clean dataset is preserved.

--- 3. SCENARIO 1: Anomaly Detection (Normal vs. Attack) ---
Engineering features on data copy...
Feature engineering complete. Total features: 19
Final features being used for modeling:
  - chlorine_level_after_attack
  - coag_to_raw_turb_ratio
  - coagulant_pump_speed_after_attack
  - delta_chlorine_level
  - delta_coagulant_pump_speed
  - delta_filter_backwash_pump_status
  - delta_filter_outlet_turbidity
  - delta_intake_pump_status
  - delta_raw_water_flow_rate
  - delta_raw_water_turbidity
  - delta_sedimentation_turbidity
  - filter_backwash_pump_status_after_attack
  - filter_effectiveness_ratio
  - filter_outlet_turbidity_after_attack
  - intake_pump_status_after_attack
  - raw_water_flow_rate_after_attack
  - raw_water_turbidity_after_attack
  - sed_effectiveness_ratio
  - sedimentation_turbidity_after_attack
Performing temporal split: 70.0% train, 30.000000000000004% test
Train set shape: (1512, 19), Test set shape: (648, 19)

=== Model: Tuning Random Forest Classifier ===
Fitting 3 folds for each of 8 candidates, totalling 24 fits
Best parameters found: {'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 50}

Classification Report (Tuned Random Forest on Test Set):
              precision    recall  f1-score   support

      Normal       1.00      0.98      0.99       285
      Attack       0.99      1.00      0.99       363

    accuracy                           0.99       648
   macro avg       0.99      0.99      0.99       648
weighted avg       0.99      0.99      0.99       648

Saved 'anomaly_detector.joblib' to disk.

Saved 'scenario_1_feature_importance.png'

--- 4. SCENARIO 2: Unsupervised Anomaly Detection (Isolation Forest) ---
Engineering features on data copy...
Performing temporal split: 70.0% train, 30.000000000000004% test
Train set shape: (1512, 19), Test set shape: (648, 19)
WARNING: Test set contamination rate is 0.56, which is >= 0.5.
Capping contamination parameter at 0.5 for IsolationForest.
Training Isolation Forest on 645 'Normal' training samples...

Classification Report (Isolation Forest on Test Set):
              precision    recall  f1-score   support

      Normal       1.00      0.44      0.61       285
      Attack       0.70      1.00      0.82       363

    accuracy                           0.75       648
   macro avg       0.85      0.72      0.72       648
weighted avg       0.83      0.75      0.73       648


--- 5. SCENARIO 3: Attack Type Classification (Multiclass) ---
Engineering features on data copy...
Performing temporal split: 70.0% train, 30.000000000000004% test
Train set shape: (1512, 19), Test set shape: (648, 19)

=== Model: Tuning Random Forest Classifier (Multiclass) ===
Fitting 3 folds for each of 4 candidates, totalling 12 fits
Best parameters found: {'max_depth': 10, 'n_estimators': 100}

Classification Report (Tuned Random Forest Multiclass on Test Set):
                                       precision    recall  f1-score   support

    False Data Injection (High Value)       0.00      0.00      0.00         0
       Intermittent Command Injection       0.00      0.00      0.00         0
Man-in-the-Middle (Data Manipulation)       0.92      1.00      0.96       333
                                 None       1.00      0.98      0.99       285
 Process Denial of Service (Actuator)       0.00      0.00      0.00        30
  Stealthy Data Manipulation (Offset)       0.00      0.00      0.00         0

                             accuracy                           0.95       648
                            macro avg       0.32      0.33      0.32       648
                         weighted avg       0.91      0.95      0.93       648

Saved 'attack_classifier.joblib' to disk.

Saved 'scenario_3_confusion_matrix.png'

--- 6. SCENARIO 4: Process Integrity (Regression) ---
Engineering features on data copy...
Training Process Integrity Model on 645 normal training samples...
Saved 'process_integrity_model.joblib' to disk.

Model R-squared on normal training data: 0.9385
This shows the model has learned the physical process rules.
Anomaly error threshold set at: 3.0800

Process Integrity Anomaly Report (on Test Set):
              precision    recall  f1-score   support

      Normal       0.43      0.98      0.60       285
      Attack       0.12      0.00      0.01       363

    accuracy                           0.43       648
   macro avg       0.28      0.49      0.30       648
weighted avg       0.26      0.43      0.27       648


Saved 'scenario_4_process_error_timeline.png'

--- 7. SCENARIO 5: Real-Time Alerting Simulation (Inference) ---
Successfully loaded 'anomaly_detector.joblib' and 'attack_classifier.joblib'.
Simulating real-time alerts for all 648 data points in the test set...

***********************************
! A L E R T (Datapoint 1 at index 1512) : POTENTIAL ATTACK DETECTED !
  Classified Attack: Man-in-the-Middle (Data Manipulation)
  Confidence:        89.52%
***********************************

***********************************
! A L E R T (Datapoint 2 at index 1513) : POTENTIAL ATTACK DETECTED !
  Classified Attack: Man-in-the-Middle (Data Manipulation)
  Confidence:        94.56%
***********************************

***********************************
! A L E R T (Datapoint 3 at index 1514) : POTENTIAL ATTACK DETECTED !
  Classified Attack: Man-in-the-Middle (Data Manipulation)
  Confidence:        87.25%
***********************************

***********************************
! A L E R T (Datapoint 4 at index 1515) : POTENTIAL ATTACK DETECTED !
  Classified Attack: Man-in-the-Middle (Data Manipulation)
  Confidence:        83.55%
***********************************

***********************************
! A L E R T (Datapoint 5 at index 1516) : POTENTIAL ATTACK DETECTED !
  Classified Attack: Man-in-the-Middle (Data Manipulation)
  Confidence:        95.80%
***********************************

... (suppressing further alert details for brevity) ...

--- Real-Time Simulation Complete ---
Total datapoints processed: 648
Total alerts generated:     365
(Ground Truth: There were 363 attack datapoints in the test set)

--- ML Pipeline Finished Successfully ---
Generated plots: 'scenario_1_feature_importance.png', 'scenario_3_confusion_matrix.png', 'scenario_4_process_error_timeline.png'