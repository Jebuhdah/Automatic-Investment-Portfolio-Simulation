import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pyomo.environ import *
import warnings
warnings.filterwarnings("ignore")

# =======================================================
# STRATEGY CONTROL PANEL (LIVE PRODUCTION)
# =======================================================
lookback_window = 15
num_stocks_to_pick = 10
min_sector_weight = 0.05
max_sector_weight = 0.40
min_stock_weight = 0.05
max_stock_weight = 0.25
risk_aversion = 2.0  # Deploying the 'High Defense' model for live trading

print(f"Waking up... Fetching last {lookback_window} trading days of market data.")
end_date = datetime.now()
start_date = end_date - timedelta(days=50) # Wide buffer for weekends/holidays
current_today = end_date.strftime('%Y-%m-%d')

# 1. Asset Universe & Mapping
tech = ["NVDA", "MSFT", "GOOGL", "META", "AVGO", "PLTR", "AMD", "CRWD", "TSM", "ASML"]
energy = ["ENB", "ENB.TO", "XOM", "CVX", "COP", "VLO", "EOG", "OXY", "MPC", "PSX"]
health = ["LLY", "NVO", "VRTX", "REGN", "ISRG", "BSX", "SYK", "UNH", "ABBV", "TMO"]
finance = ["V", "MA", "JPM", "GS", "MS", "BLK", "CME", "SPGI", "AXP", "MCO"]
consumer = ["COST", "WMT", "PG", "PEP", "KO", "MCD", "CMG", "HD", "SBUX", "NKE"]
all_tickers = tech + energy + health + finance + consumer

sector_map = {t: "Tech" for t in tech} | {t: "Energy" for t in energy} | \
             {t: "Healthcare" for t in health} | {t: "Financials" for t in finance} | \
             {t: "Consumer" for t in consumer} | {"GIC": "Cash"}

# 2. Data Processing
data = yf.download(all_tickers, start=start_date.strftime('%Y-%m-%d'), end=current_today)['Close'].ffill()
returns_df = data.pct_change().dropna().tail(lookback_window) # Exact window
returns_df['GIC'] = (1.01**(1/252)) - 1
tickers = returns_df.columns.tolist()
unique_sectors = list(set(sector_map.values()))

# 3. Solver Setup
print("Data loaded. Initializing Bonmin MINLP Solver...")
solver = SolverFactory('bonmin', executable='/home/runner/.idaes/bin/bonmin')
solver.options['bonmin.time_limit'] = 10 # 10-second hard cutoff for production

sector_daily = pd.DataFrame({s: returns_df[[t for t in tickers if sector_map[t] == s]].mean(axis=1) for s in unique_sectors})
sec_exp_ret, sec_cov = sector_daily.mean(), sector_daily.cov()

# --- STAGE 1: The CEO ---
m_sec = ConcreteModel()
m_sec.w = Var(unique_sectors, domain=NonNegativeReals, bounds=(min_sector_weight, max_sector_weight))
m_sec.sum_w = Constraint(expr=sum(m_sec.w[s] for s in unique_sectors) == 1.0)
m_sec.obj = Objective(expr=sum(m_sec.w[s]*sec_exp_ret[s] for s in unique_sectors) - (risk_aversion * sum(m_sec.w[u]*sec_cov.loc[u,v]*m_sec.w[v] for u in unique_sectors for v in unique_sectors)), sense=maximize)
try: solver.solve(m_sec, tee=False)
except: pass
opt_sec_weights = {s: value(m_sec.w[s]) for s in unique_sectors}

# --- STAGE 2: The Manager ---
m = ConcreteModel()
m.x = Var(tickers, domain=NonNegativeReals, bounds=(0, 1))
m.y = Var(tickers, domain=Binary)
exp_ret, cov_matrix = returns_df.mean(), returns_df.cov()

m.obj = Objective(expr=sum(m.x[t]*exp_ret[t] for t in tickers) - (risk_aversion * sum(m.x[u]*cov_matrix.loc[u,v]*m.x[v] for u in tickers for v in tickers)), sense=maximize)
m.sum_to_one = Constraint(expr=sum(m.x[t] for t in tickers) == 1.0)
m.sec_low = Constraint(unique_sectors, rule=lambda m, s: sum(m.x[t] for t in tickers if sector_map[t] == s) >= opt_sec_weights[s] - 0.01)
m.sec_high = Constraint(unique_sectors, rule=lambda m, s: sum(m.x[t] for t in tickers if sector_map[t] == s) <= opt_sec_weights[s] + 0.01)
m.link_max = Constraint(tickers, rule=lambda m, t: m.x[t] <= (1.0 if t == 'GIC' else max_stock_weight) * m.y[t])
m.link_min = Constraint(tickers, rule=lambda m, t: m.x[t] >= min_stock_weight * m.y[t])
m.pick_n = Constraint(expr=sum(m.y[t] for t in tickers) == num_stocks_to_pick)

print(f"Solving Target Allocation for: {current_today}...")
sol = solver.solve(m, tee=False)

# 4. Output Generation
print("\n" + "="*45)
print(f"   LIVE ALLOCATION FOR {current_today}")
print("="*45)
allocations = []
if sol.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.maxIterations]:
    for t in tickers:
        if value(m.x[t]) > 0.001:
            print(f"{t:<10} | Sector: {sector_map[t]:<12} | Weight: {value(m.x[t])*100:>5.2f}%")
            allocations.append({"Date": current_today, "Ticker": t, "Sector": sector_map[t], "Weight": round(value(m.x[t]), 4)})
else:
    print("Solver failed to find a feasible solution today.")

# Save to CSV so GitHub Actions can push it to your repository
pd.DataFrame(allocations).to_csv("latest_allocation.csv", index=False)
print("Saved to latest_allocation.csv. Automation cycle complete.")
