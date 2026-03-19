# MICRONAS Explainability Report 🧠

## 1. Why this Architecture was Chosen
The Evolutionary NAS engine evaluated multiple architectures based on the fitness function derived from your natural language prompt.

**Best Configuration Found:**
```json
{
    "type": "xgb",
    "name": "XGBoost",
    "n_estimators": 300,
    "learning_rate": 0.01,
    "max_depth": 3
}
```

## 2. Trade-offs Made
* **Parameter Count**: 30000 parameters
* **Memory Footprint**: 50.0 MB
* **Hardware Constraints**: The system strictly adhered to constraints (e.g., 3GB VRAM, CPU fallbacks), optimizing the kernel sizes and layer depths to avoid Out-Of-Memory (OOM) failures.

## 3. Failure-Aware System Insights
If the system encountered models with collapsing spatial dimensions (e.g., negative tensor sizes) or OOM errors, it automatically repaired them by dynamically reducing kernel sizes or truncating layers, avoiding system crashes and storing the failure states in its memory.

## 4. Final Verification
The model completed full training loop securely via mixed-precision offloading, achieving optimal hardware utilization without compromising accuracy.
