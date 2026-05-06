# Automatic Investment Portfolio Simulation (Modern Portfolio Theory 2.0)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pyomo](https://img.shields.io/badge/Optimization-Pyomo-green)
![GitHub Actions](https://img.shields.io/badge/Automation-GitHub_Actions-lightgrey)

## Overview
This repository houses a fully automated, active-management trading pipeline that mathematically calculates the optimal portfolio allocation to maximize the Sharpe ratio. Moving beyond static "Buy & Hold" strategies, this bot acts as an autonomous quantitative researcher—recalculating the covariance matrix of a 51-asset universe every 24 hours based on a 15-day sliding momentum window.

## Core Architecture: The "CEO and Manager" Model
The optimization engine uses a two-stage **Mixed-Integer Non-Linear Programming (MINLP)** logic powered by the **Bonmin** solver to mimic institutional fund management:
1. **The CEO Stage (Sector Allocation):** Calculates safe macroeconomic sector weights to ensure broad diversification and shield against industry-specific crashes.
2. **The Manager Stage (Stock Selection):** Tactically selects specific assets within the CEO's boundaries to capture momentum alpha.

## Mathematical Guardrails & Constraints
To prevent dangerous over-concentration and ensure production reliability, strict integer programming constraints are enforced:
* **Cardinality:** The solver is forced to select *exactly* 10 stocks daily.
* **Stock Ceilings & Floors:** Absolute limits of 5% (minimum) to 25% (maximum) per asset.
* **Sector Ceilings & Floors:** Absolute limits of 5% (minimum) to 40% (maximum) per sector.
* **The GIC Safety Net:** Integrates a 1% yield Guaranteed Investment Certificate (GIC) as a cash-equivalent fallback. During periods of extreme market variance, the solver autonomously parks capital here to protect the principal.

## Cloud-Native Automation Pipeline
This project requires **zero manual intervention**. 
A GitHub Actions Cron Job is scheduled to run every Monday through Friday at `19:55 UTC` (3:55 PM EDT), minutes before the market closes. The pipeline:
1. Spins up an Ubuntu server.
2. Installs Linux math libraries (`liblapack3`, `libblas3`) and IDAES extensions.
3. Fetches the latest 15 days of market data via the `yfinance` API.
4. Solves the MINLP optimization model.
5. Commits the daily targets directly to `latest_allocation.csv` in this repository.

## Repository Structure
* `.github/workflows/automate.yml`: The CI/CD pipeline configuring the daily cron job.
* `daily_optimizer.py`: The lightweight, production-ready Pyomo execution script.
* `latest_allocation.csv`: The live output file containing today's optimal 10-stock distribution.
* `5641_Final.ipynb`: The comprehensive research notebook detailing the backtesting, visual dashboards, and performance scorecards.

## Performance Benchmark (Q1 2026)
In historical backtesting against the S&P 500 baseline, the "Balanced Risk" (RA=2.0) strategy achieved a **20.72% return**, generating roughly +10% alpha over the market by successfully identifying and pivoting between Energy and Tech momentum waves.
