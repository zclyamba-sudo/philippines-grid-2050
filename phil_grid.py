from pyomo.environ import *
import numpy as np

# We'll simulate 96 hours (4 days × 24 hours)
T = 96    # number of time periods

# Create synthetic demand (MW) – just a wavy line
np.random.seed(123)   # so numbers are the same every run
demand = 20000 + 5000 * np.sin(np.linspace(0, 4*np.pi, T))

# Solar and wind availability (capacity factor) for each hour
solar_cf = np.zeros(T)   # start all zeros
wind_cf = np.zeros(T)
for hour in range(T):
    hour_of_day = hour % 24
    if 6 <= hour_of_day <= 18:
        # Solar is available between 6am and 6pm
        solar_cf[hour] = 0.1 + 0.7 * np.sin(np.pi * (hour_of_day-6)/12)
    # Wind: simple diurnal pattern, higher at night
    wind_cf[hour] = 0.3 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24)

# Costs (simplified for learning)
cap_cost_solar = 800000      # $ per MW (annualised already – don't worry)
cap_cost_wind  = 1200000
cap_cost_gas   = 600000
cap_cost_storage_power = 250000   # $ per MW of power capacity
cap_cost_storage_energy = 300000  # $ per MWh of energy capacity

var_cost_gas   = 40          # $ per MWh (fuel + variable O&M)
emission_factor = 0.4        # tonnes CO2 per MWh of gas
emission_cap   = 10_000_000  # tonnes CO2 per year (scaled later)

# Storage efficiencies
eff_charge = 0.95
eff_discharge = 0.95

# Scaling factor because we only model 96 hours, not full 8760 hours of a year
hours_per_year = 8760
scale_factor = hours_per_year / T   # about 91.25

model = ConcreteModel()

# Time set
model.T = RangeSet(T)   # creates indices 1..T (Pyomo starts at 1, not 0)

# Decision variables: how much capacity to build (MW or MWh)
model.cap_solar = Var(domain=NonNegativeReals)
model.cap_wind = Var(domain=NonNegativeReals)
model.cap_gas = Var(domain=NonNegativeReals)
model.cap_storage_power = Var(domain=NonNegativeReals)   # MW charge/discharge rate
model.cap_storage_energy = Var(domain=NonNegativeReals)  # MWh

# Operational variables: how much we generate/charge at each hour
model.gen_solar = Var(model.T, domain=NonNegativeReals)
model.gen_wind  = Var(model.T, domain=NonNegativeReals)
model.gen_gas   = Var(model.T, domain=NonNegativeReals)
model.charge    = Var(model.T, domain=NonNegativeReals)    # battery charging
model.discharge = Var(model.T, domain=NonNegativeReals)    # battery discharging
model.soc       = Var(model.T, domain=NonNegativeReals)    # state of charge (MWh)
model.curtail   = Var(model.T, domain=NonNegativeReals)    # wasted renewable energy

def total_cost(model):
    # Investment cost (building)
    inv = (cap_cost_solar * model.cap_solar +
           cap_cost_wind  * model.cap_wind +
           cap_cost_gas   * model.cap_gas +
           cap_cost_storage_power * model.cap_storage_power +
           cap_cost_storage_energy * model.cap_storage_energy)
    
    # Variable cost (fuel for gas)
    fuel = sum(var_cost_gas * model.gen_gas[t] for t in model.T)
    
    # Note: fixed O&M is ignored for simplicity here, but you can add later
    return inv + fuel

model.cost = Objective(rule=total_cost, sense=minimize)

def balance_rule(model, t):
    return (model.gen_solar[t] + model.gen_wind[t] + model.gen_gas[t] + model.discharge[t] ==
            demand[t-1] + model.charge[t] + model.curtail[t])   # note: t starts at 1, so demand index t-1
model.balance = Constraint(model.T, rule=balance_rule)

def solar_limit_rule(model, t):
    return model.gen_solar[t] <= model.cap_solar * solar_cf[t-1]
model.solar_limit = Constraint(model.T, rule=solar_limit_rule)

def wind_limit_rule(model, t):
    return model.gen_wind[t] <= model.cap_wind * wind_cf[t-1]
model.wind_limit = Constraint(model.T, rule=wind_limit_rule)

def gas_min_rule(model, t):
    return model.gen_gas[t] >= 0.2 * model.cap_gas   # min 20% load
model.gas_min = Constraint(model.T, rule=gas_min_rule)

def gas_max_rule(model, t):
    return model.gen_gas[t] <= model.cap_gas
model.gas_max = Constraint(model.T, rule=gas_max_rule)

def soc_evolution(model, t):
    if t == 1:
        # Start with empty battery
        return model.soc[t] == 0
    else:
        # Energy stored = previous energy + charge - discharge (with losses)
        return model.soc[t] == model.soc[t-1] + eff_charge * model.charge[t] - model.discharge[t] / eff_discharge
model.soc_evolution = Constraint(model.T, rule=soc_evolution)

# Battery can't store more than its energy capacity
def soc_upper(model, t):
    return model.soc[t] <= model.cap_storage_energy
model.soc_upper = Constraint(model.T, rule=soc_upper)

# Charge and discharge can't exceed power capacity
def charge_limit(model, t):
    return model.charge[t] <= model.cap_storage_power
model.charge_limit = Constraint(model.T, rule=charge_limit)

def discharge_limit(model, t):
    return model.discharge[t] <= model.cap_storage_power
model.discharge_limit = Constraint(model.T, rule=discharge_limit)

# At the end, battery should be empty (cyclic operation)
def soc_cyclic(model):
    return model.soc[T] == 0
model.soc_cyclic = Constraint(rule=soc_cyclic)

def emission_rule(model):
    total_emissions = scale_factor * emission_factor * sum(model.gen_gas[t] for t in model.T)
    return total_emissions <= emission_cap
model.emission = Constraint(rule=emission_rule)

# Solve
solver = SolverFactory('glpk')
result = solver.solve(model, tee=True)   # tee=True prints solver log

# Sensitivity analysis
print("\n===== SENSITIVITY: VARYING EMISSION CAP =====")
caps = [5_000_000, 10_000_000, 15_000_000, 20_000_000, 30_000_000]
for new_cap in caps:
    # Change the emission cap in the model
    model.emission.set_value(scale_factor * emission_factor * sum(model.gen_gas[t] for t in model.T) <= new_cap)
    solver.solve(model, tee=False)
    print(f"Cap = {new_cap/1e6:.0f} MtCO2 -> Gas = {value(model.cap_gas):.0f} MW, Cost = ${value(model.cost)/1e6:.1f}M")

# Print results
print("\n===== RESULTS =====")
print(f"Solar capacity:    {value(model.cap_solar):.0f} MW")
print(f"Wind capacity:     {value(model.cap_wind):.0f} MW")
print(f"Gas capacity:      {value(model.cap_gas):.0f} MW")
print(f"Battery power:     {value(model.cap_storage_power):.0f} MW")
print(f"Battery energy:    {value(model.cap_storage_energy):.0f} MWh")
print(f"Total annual cost: ${value(model.cost)/1e6:.1f} million")

