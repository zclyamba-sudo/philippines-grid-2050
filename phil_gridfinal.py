"""
Philippines Grid Capacity Expansion Model (2050)
With sensitivity analysis on emission cap
"""

import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *

# ============================================================
# 1. SOLVER SETUP (use CBC – no admin rights needed)
# ============================================================
# Try to use CBC, fallback to GLPK if needed
solver_name = 'cbc'
try:
    solver = SolverFactory(solver_name)
    if not solver.available():
        raise Exception(f"{solver_name} not available")
except:
    print(f"{solver_name} not found, trying glpk...")
    solver = SolverFactory('glpk')
    solver_name = 'glpk'

print(f"Using solver: {solver_name}")

# ============================================================
# 2. MODEL PARAMETERS (Philippines 2050)
# ============================================================
T = 96                     # 4 representative days × 24 hours
np.random.seed(123)        # reproducible results

# Hourly demand (MW) – sinusoidal with daily & seasonal pattern
demand = 20000 + 5000 * np.sin(np.linspace(0, 4*np.pi, T))

# Solar availability (capacity factor)
solar_cf = np.zeros(T)
wind_cf = np.zeros(T)
for hour in range(T):
    hour_of_day = hour % 24
    # Solar: available 6am–6pm, peaking at noon
    if 6 <= hour_of_day <= 18:
        solar_cf[hour] = 0.1 + 0.7 * np.sin(np.pi * (hour_of_day-6)/12)
    # Wind: simple diurnal pattern (higher at night)
    wind_cf[hour] = 0.3 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24)

# Economic and technical parameters
cap_cost = {
    'solar': 800_000,          # $/MW
    'wind': 1_200_000,
    'gas': 600_000,
    'storage_power': 250_000,  # $/MW (power capacity)
    'storage_energy': 300_000  # $/MWh (energy capacity)
}
var_cost_gas = 40              # $/MWh fuel + variable O&M
emission_factor = 0.4          # tCO2/MWh of natural gas

# Storage efficiency
eta_charge = 0.95
eta_discharge = 0.95

# Time scaling (our 96 hours represent a whole year)
hours_per_year = 8760
scale_factor = hours_per_year / T   # ~91.25

# ============================================================
# 3. FUNCTION TO BUILD AND SOLVE MODEL FOR GIVEN EMISSION CAP
# ============================================================
def solve_for_emission_cap(emission_cap, silent=False):
    """Build and solve the model for a specific CO2 cap (tonnes/year)."""
    model = ConcreteModel()
    model.T = RangeSet(T)

    # ----- Decision variables -----
    model.cap_solar = Var(domain=NonNegativeReals)
    model.cap_wind = Var(domain=NonNegativeReals)
    model.cap_gas = Var(domain=NonNegativeReals)
    model.cap_storage_power = Var(domain=NonNegativeReals)
    model.cap_storage_energy = Var(domain=NonNegativeReals)

    model.gen_solar = Var(model.T, domain=NonNegativeReals)
    model.gen_wind = Var(model.T, domain=NonNegativeReals)
    model.gen_gas = Var(model.T, domain=NonNegativeReals)
    model.charge = Var(model.T, domain=NonNegativeReals)
    model.discharge = Var(model.T, domain=NonNegativeReals)
    model.soc = Var(model.T, domain=NonNegativeReals)
    model.curtail = Var(model.T, domain=NonNegativeReals)

    # ----- Objective: minimise total cost -----
    def total_cost(model):
        inv = (cap_cost['solar'] * model.cap_solar +
               cap_cost['wind'] * model.cap_wind +
               cap_cost['gas'] * model.cap_gas +
               cap_cost['storage_power'] * model.cap_storage_power +
               cap_cost['storage_energy'] * model.cap_storage_energy)
        fuel = sum(var_cost_gas * model.gen_gas[t] for t in model.T)
        return inv + fuel
    model.cost = Objective(rule=total_cost, sense=minimize)

    # ----- Constraints -----
    # Power balance
    def balance_rule(model, t):
        return (model.gen_solar[t] + model.gen_wind[t] + model.gen_gas[t] + model.discharge[t] ==
                demand[t-1] + model.charge[t] + model.curtail[t])
    model.balance = Constraint(model.T, rule=balance_rule)

    # Renewable limits
    def solar_limit(model, t):
        return model.gen_solar[t] <= model.cap_solar * solar_cf[t-1]
    model.solar_limit = Constraint(model.T, rule=solar_limit)

    def wind_limit(model, t):
        return model.gen_wind[t] <= model.cap_wind * wind_cf[t-1]
    model.wind_limit = Constraint(model.T, rule=wind_limit)

    # Gas plant minimum (20% of capacity)
    def gas_min(model, t):
        return model.gen_gas[t] >= 0.2 * model.cap_gas
    model.gas_min = Constraint(model.T, rule=gas_min)

    def gas_max(model, t):
        return model.gen_gas[t] <= model.cap_gas
    model.gas_max = Constraint(model.T, rule=gas_max)

    # Storage dynamics
    def soc_evolution(model, t):
        if t == 1:
            return model.soc[t] == 0
        else:
            return model.soc[t] == (model.soc[t-1] + eta_charge * model.charge[t] -
                                    model.discharge[t] / eta_discharge)
    model.soc_evolution = Constraint(model.T, rule=soc_evolution)

    def soc_upper(model, t):
        return model.soc[t] <= model.cap_storage_energy
    model.soc_upper = Constraint(model.T, rule=soc_upper)

    def charge_limit(model, t):
        return model.charge[t] <= model.cap_storage_power
    model.charge_limit = Constraint(model.T, rule=charge_limit)

    def discharge_limit(model, t):
        return model.discharge[t] <= model.cap_storage_power
    model.discharge_limit = Constraint(model.T, rule=discharge_limit)

    def soc_cyclic(model):
        return model.soc[T] == 0
    model.soc_cyclic = Constraint(rule=soc_cyclic)

    # Emission cap (scaled to full year)
    def emission_rule(model):
        annual_emissions = scale_factor * emission_factor * sum(model.gen_gas[t] for t in model.T)
        return annual_emissions <= emission_cap
    model.emission = Constraint(rule=emission_rule)

    # Solve
    solver.solve(model, tee=not silent)

    # Extract results
    return {
        'cap_solar': model.cap_solar(),
        'cap_wind': model.cap_wind(),
        'cap_gas': model.cap_gas(),
        'cap_storage_power': model.cap_storage_power(),
        'cap_storage_energy': model.cap_storage_energy(),
        'total_cost': model.cost()
    }

# ============================================================
# 4. SENSITIVITY ANALYSIS: VARY EMISSION CAP
# ============================================================
# Emission caps to test (million tonnes CO2 per year)
caps_mt = [5, 10, 15, 20, 30, 40, 50]   # MtCO2/year
caps_tonnes = [cap * 1e6 for cap in caps_mt]

results = []
print("\n" + "="*70)
print("SENSITIVITY ANALYSIS: VARYING EMISSION CAP")
print("="*70)

for cap_tonnes in caps_tonnes:
    print(f"\nSolving for emission cap = {cap_tonnes/1e6:.0f} MtCO2/year...")
    sol = solve_for_emission_cap(cap_tonnes, silent=True)
    results.append(sol)
    print(f"  Gas: {sol['cap_gas']:.0f} MW, Solar: {sol['cap_solar']:.0f} MW, "
          f"Wind: {sol['cap_wind']:.0f} MW, Cost: ${sol['total_cost']/1e6:.1f}M")

# ============================================================
# 5. CREATE SENSITIVITY PLOT
# ============================================================
# Extract data for plotting
gas_cap = [r['cap_gas'] for r in results]
solar_cap = [r['cap_solar'] for r in results]
wind_cap = [r['cap_wind'] for r in results]
storage_power_cap = [r['cap_storage_power'] for r in results]
total_cost = [r['total_cost']/1e6 for r in results]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot capacities
ax1.plot(caps_mt, solar_cap, 'o-', label='Solar PV', linewidth=2, markersize=8)
ax1.plot(caps_mt, wind_cap, 's-', label='Wind', linewidth=2, markersize=8)
ax1.plot(caps_mt, gas_cap, '^-', label='Natural Gas', linewidth=2, markersize=8)
ax1.plot(caps_mt, storage_power_cap, 'd-', label='Battery Power', linewidth=2, markersize=8)
ax1.set_xlabel('Emission Cap (Mt CO₂/year)', fontsize=12)
ax1.set_ylabel('Installed Capacity (MW)', fontsize=12)
ax1.set_title('Technology Capacity vs. CO₂ Emission Limit', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot total cost
ax2.plot(caps_mt, total_cost, 'o-', color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Emission Cap (Mt CO₂/year)', fontsize=12)
ax2.set_ylabel('Total Annual Cost (Million USD)', fontsize=12)
ax2.set_title('System Cost vs. CO₂ Emission Limit', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensitivity_plot.png', dpi=150)
plt.show()

print("\n" + "="*70)
print("Sensitivity plot saved as 'sensitivity_plot.png'")
print("="*70)