# Philippines Grid Capacity Expansion Model (2050)

This repository contains a **least‑cost capacity expansion model** for the Philippine electricity grid in 2050.  
It decides how much solar PV, wind, natural gas, and battery storage to install – minimising total annualised cost (investment + fuel) while meeting hourly demand, technical limits, and a CO₂ emission cap.

## 🎯 What the model does

- **Decision variables**: installed capacity (MW) of solar, wind, gas, battery power, battery energy.
- **Operational variables**: hourly generation, battery charge/discharge, curtailment.
- **Objective**: minimise total annual cost (capital + fuel + variable O&M).
- **Constraints**:
  - Power balance (generation + discharge = demand + charge + curtailment)
  - Renewable output limited by hourly capacity factors
  - Gas plant must run between 20% and 100% of capacity
  - Battery energy balance (with round‑trip efficiency) and cycling
  - Annual CO₂ emissions from gas ≤ user‑defined cap

The model uses **4 representative days** (96 hours) scaled to a full year.  
Time series for demand, solar, and wind are synthetic but can be replaced with real Philippine data.

## ⚙️ How to run

### 1. Install required packages (no admin rights needed)

```bash
pip install --user numpy matplotlib cbcpy pyomo
