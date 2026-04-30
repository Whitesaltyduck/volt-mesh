# Volt-Mesh: Autonomous Multi-Agent Smart Grid System

Volt-Mesh is an intelligent, decentralized multi-agent system designed to diagnose smart grid health and balance residential loads. It transforms a passive electrical network into an adaptive ecosystem capable of predicting overloads and scheduling high-power appliances during cost-effective "Eco" time windows.

## 📋 Prerequisites

To run Volt-Mesh on your machine, you will need the following installed:

1. **Python 3.8+**
2. **Required Python Packages:**
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```
3. **Dataset:** 
   The system is built to ingest the *UCI Individual Household Electric Power Consumption Dataset*. 
   * Download the dataset and ensure the CSV is located exactly at: `C:\data\power_usage.csv`
   * The file should be semicolon-delimited (`;`) and use `?` for missing values.

## 🏗️ System Architecture (The Agents)

Volt-Mesh uses a heterogeneous multi-agent mesh configured in `opencode.json`. The agents communicate via a shared memory interface located in the `skills/` directory.

* **Agent 1: Usage Profiler (Clusternaut)**
  * **File:** `skills/usage_profiler.py`
  * **Role:** Analyzes the UCI dataset and uses **K-Means Clustering** to segment the 24-hour day into three distinct energy zones: `Eco`, `Moderate`, and `Peak`.
  * **Output:** Generates `cluster_map.json`.

* **Agent 2: Reliability Sentinel (The Predictor)**
  * **File:** `skills/reliability_sentinel.py`
  * **Role:** Employs a **Naive Bayes Classifier** trained on multivariate telemetry (`Global_active_power`, `Voltage`, `Global_intensity`) to calculate the probability of a grid overload. If $P(\text{Overload}) > 0.75$, it triggers a `"load_balance_request"`.
  * **Output:** Generates `reliability_sentinel_model.joblib`.

* **Agent 3: Load Optimizer (The Searcher)**
  * **File:** `skills/load_optimizer.py`
  * **Role:** Listens for load balance requests and uses a **Greedy Best-First Search** to identify the most cost-effective 1-hour window for heavy appliances. Tie-breakers are resolved by selecting the Eco hour with the absolute lowest historical power load.
  * **Output:** Generates `load_optimizer_recommendation.json`.

## 🚀 How to Run the System

The system is designed to be highly modular. You must first train the agents on your data, after which you can run real-time simulations. All operations should be run from within the `skills/` directory.

### Step 1: Train the Profiler
Initialize the cluster mapping to understand the household's daily fingerprint.
```bash
python skills/usage_profiler.py
```
*Expected Result: `cluster_map.json` is created in the skills directory.*

### Step 2: Train the Sentinel
Train the Naive Bayes probability model on the 95th percentile power usage thresholds.
```bash
python skills/reliability_sentinel.py
```
*Expected Result: `reliability_sentinel_model.joblib` is created.*

### Step 3: Run the Load Optimizer (Standalone Test)
You can manually test the optimizer to see which hour it recommends for heavy loads today.
```bash
python skills/load_optimizer.py
```
*Expected Result: Prints the optimal hour and creates `load_optimizer_recommendation.json`.*

### Step 4: Run the Full Mesh Simulation
Once the models are trained, run the stress-test simulation. This script simulates an 8.0 kW spike event, triggering the Sentinel to flag an overload, which dynamically awakens the Optimizer to generate a load-balancing schedule.
```bash
python skills/run_simulation.py
```

## 📊 Expected Outputs

When running the simulation, the system will output a detailed markdown report named **`efficiency_audit.md`** inside the `skills/` folder. This report includes:
* The simulated power consumption vs. the grid's safety threshold.
* The calculated $P(\text{Overload})$ probability.
* The Load Optimizer's recommended Eco hour for load displacement.
* Strategic insights for grid stability and customer notifications.

## 🛠️ Troubleshooting

* **FileNotFoundError for `C:\data\power_usage.csv`:** Ensure you have downloaded the UCI dataset and placed it in the exact `C:\data\` directory as expected by the pandas data ingestion logic.
* **KeyError or Misaligned Hours:** The Usage Profiler includes resilient time-series imputation. If the dataset changes drastically, ensure there are no entirely empty days that might break the 24-hour `.reindex()` boundary.
* **Agent Event Mismatches:** If altering `opencode.json`, ensure the `notification_events` emitted by the Sentinel exactly match the `triggers` listened to by the Optimizer (e.g., `"load_balance_request"`).

---
*Developed for L3 (Applying) Competence in Smart Grid Engineering.*
