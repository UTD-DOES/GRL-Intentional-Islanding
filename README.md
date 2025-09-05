# Self-Healing Power Systems Using Reinforcement Learning Over Graphs

This repository contains code and data for the paper **“Self-Healing Power Systems Using Reinforcement Learning Over Graphs for Controlled Grid Islanding on Simulated Transmission Networks in PSS®E”**. It demonstrates how graph-based reinforcement learning (RL) can improve grid resilience by intelligently controlling network islands during disturbances, preventing cascading failures, and ensuring stable operation.

---

## 📖 Overview

Modern power systems are increasingly complex, and unexpected outages can lead to cascading failures. This project develops a **graph-based RL framework** to enable **controlled islanding** and rapid self-healing of power grids. Key objectives include:

- **Rapid grid restoration:** Using RL to learn optimal switching strategies for forming stable islands.
- **Collapse prevention:** Isolating vulnerable network areas before failures propagate.
- **Stable frequency control:** Incorporating generator inertia in decision-making to maintain frequency stability.
- **Graph-based modeling:** Representing the transmission network as a graph to leverage topological features for RL.

The framework is tested on the **IEEE 118-bus system** using **PSS®E** as the simulation environment.

---

<!--## ⚡ Features-->
<!--
| File | Description |
|------|-------------|
| **CustomizedPolicy.py** | Implements the custom RL policy for islanding control. |
| **FeatureExtractor.py** | Extracts graph-based features from the power network for RL input. |
| **RunTensorboard.py** | Launches TensorBoard to visualize training metrics. |
| **TrainingConfiguration.py** | Defines RL training configurations and hyperparameters. |
| **training3.py** | Main training script integrating feature extraction and environment interaction. |
| **testing.py** | Evaluates trained RL models on unseen contingencies. |
| **pypsseEnv118.py** | PSS®E interface implementing the Gym environment for IEEE 118-bus. |
| **IEEE118_V33.sav** | Saved IEEE 118-bus case file. |
| **IEEE118.snp** | Contingency snapshot file for simulations. |
-->


## 🔄 How the Code Works Together

1. **Environment (`pypsseEnv118.py` + `IEEE118.snp` + `IEEE118_V33.sav`)**  
   The environment is the IEEE 118-bus network in **PSS®E**, with transmission grid simulated using `IEEE118_V33.sav` and dynamics data `IEEE118.snp`, allowing actions such as line switching and returning rewards and observations based on power flow, stability, load preservation, and islanding success.


2. **Feature Extraction (`FeatureExtractor.py`)**  
   Converts network topology and generator parameters (e.g., inertia, voltage) into features for the GRL agent.

3. **Custom GRL Policy (`CustomizedPolicy.py`)**  
   Learns switching actions that maximize long-term system stability while considering topology and generator characteristics.

4. **Training (`training3.py` + `TrainingConfiguration.py`)**  
   The GRL agent interacts with the environment, using features to decide islanding actions. Rewards guide learning toward resilient grid configurations.

5. **Monitoring (`RunTensorboard.py`)**  
   Logs and visualizes training metrics in real-time using TensorBoard.

6. **Testing (`testing.py`)**  
   Evaluates trained agents on new contingencies using saved network files to verify self-healing effectiveness and island stability.

---

## 🛠️ Requirements

- **Python 3.9+**
- **PSS®E 35**
- **NumPy, Pandas, PyTorch, NetworkX**
- **OpenAI Gym**
- **TensorBoard**

> ⚠️ Ensure PSS®E is installed and properly licensed. The Python API must be accessible.

---

## 📄 Citation
If you use this code, please cite:

Sobhan Badakhsh, Roshni Anna Jacoba, Binghui Li, Pingfeng Wang, Jie Zhang, "Self-Healing Power Systems Using Reinforcement Learning Over Graphs for Controlled Grid Islanding," Sustainable Energy, Grids and Networks, 2025.

---

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/self-healing-power-systems.git
cd self-healing-power-systems
