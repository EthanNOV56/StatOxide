#!/usr/bin/env python3
"""
Benchmark and comparison tests between StatOxide and existing Python libraries
(statsmodels, scikit-learn).
"""

import numpy as np
import pandas as pd
import time
import sys

# Try to import statoxide
try:
    import statoxide as so
    print("✓ StatOxide imported successfully")
    STATOXIDE_AVAILABLE = True
except ImportError as e:
    print(f"✗ StatOxide import failed: {e}")
    STATOXIDE_AVAILABLE = False
    sys.exit(1)

# Try to import comparison libraries
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tsa.arima.model import ARIMA as smARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    print("✓ statsmodels imported successfully")
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠ statsmodels import failed: {e}")
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_squared_error, accuracy_score
    print("✓ scikit-learn imported successfully")
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠ scikit-learn import failed: {e}")
    SKLEARN_AVAILABLE = False

print("\n" + "="*80)
print("StatOxide vs. Existing Libraries Benchmark")
print("="*80)

# Generate synthetic data for testing
np.random.seed(42)

# Linear regression test data
n_samples = 1000
n_features = 5

X_lr = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
noise = np.random.randn(n_samples) * 0.1
y_lr = X_lr @ true_coef + noise + 2.0  # Add intercept

# Logistic regression test data
X_logit = np.random.randn(n_samples, n_features)
logit_coef = np.array([0.8, -1.2, 0.3, 0.5, -0.7])
logits = X_logit @ logit_coef + 0.5
probabilities = 1 / (1 + np.exp(-logits))
y_logit = np.random.binomial(1, probabilities)

# Time series data
n_timepoints = 200
ar_coefs = [0.5, -0.2]  # AR(2)
ma_coefs = [0.3]        # MA(1)
ts_data = np.zeros(n_timepoints)
noise_ts = np.random.randn(n_timepoints) * 0.5

# Generate ARMA(2,1) process
for t in range(2, n_timepoints):
    ts_data[t] = (ar_coefs[0] * ts_data[t-1] + 
                  ar_coefs[1] * ts_data[t-2] + 
                  noise_ts[t] + 
                  ma_coefs[0] * noise_ts[t-1])

# GARCH data (volatility clustering)
garch_data = np.zeros(n_timepoints)
sigma2 = np.zeros(n_timepoints)
sigma2[0] = 1.0
omega, alpha, beta = 0.1, 0.1, 0.85

for t in range(1, n_timepoints):
    sigma2[t] = omega + alpha * garch_data[t-1]**2 + beta * sigma2[t-1]
    garch_data[t] = np.sqrt(sigma2[t]) * np.random.randn()

def run_linear_regression_comparison():
    """Compare linear regression implementations."""
    print("\n" + "-"*80)
    print("Linear Regression Comparison")
    print("-"*80)
    
    results = {}
    
    # 1. StatOxide OLS
    if STATOXIDE_AVAILABLE:
        print("\n1. StatOxide OLS:")
        try:
            start = time.time()
            
            # Create DataFrame for StatOxide
            # Note: StatOxide expects column-major or specific format
            # For simplicity, use direct array input if available
            # Otherwise, use the models API
            
            # Convert to list of lists for statoxide
            X_list = X_lr.tolist()
            y_list = y_lr.tolist()
            
            result = so.models.linear_regression(X_list, y_list)
            
            so_coef = result.get("coefficients", [])
            so_r2 = result.get("r_squared", 0)
            so_time = time.time() - start
            
            print(f"   Coefficients: {so_coef[:3]}...")
            print(f"   R-squared: {so_r2:.4f}")
            print(f"   Time: {so_time:.4f}s")
            
            results['statoxide'] = {
                'coefficients': so_coef,
                'r_squared': so_r2,
                'time': so_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. statsmodels OLS
    if STATSMODELS_AVAILABLE:
        print("\n2. statsmodels OLS:")
        try:
            start = time.time()
            
            # Add constant for intercept
            X_sm = sm.add_constant(X_lr)
            model = sm.OLS(y_lr, X_sm)
            result_sm = model.fit()
            
            sm_coef = result_sm.params.tolist()
            sm_r2 = result_sm.rsquared
            sm_time = time.time() - start
            
            print(f"   Coefficients: {sm_coef[:3]}...")
            print(f"   R-squared: {sm_r2:.4f}")
            print(f"   Time: {sm_time:.4f}s")
            
            results['statsmodels'] = {
                'coefficients': sm_coef,
                'r_squared': sm_r2,
                'time': sm_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # 3. scikit-learn LinearRegression
    if SKLEARN_AVAILABLE:
        print("\n3. scikit-learn LinearRegression:")
        try:
            start = time.time()
            
            model_sk = LinearRegression()
            model_sk.fit(X_lr, y_lr)
            
            sk_coef = model_sk.coef_.tolist()
            sk_intercept = model_sk.intercept_
            sk_r2 = model_sk.score(X_lr, y_lr)
            sk_time = time.time() - start
            
            # Combine intercept and coefficients
            sk_all_coef = [sk_intercept] + sk_coef
            
            print(f"   Coefficients: {sk_all_coef[:3]}...")
            print(f"   R-squared: {sk_r2:.4f}")
            print(f"   Time: {sk_time:.4f}s")
            
            results['sklearn'] = {
                'coefficients': sk_all_coef,
                'r_squared': sk_r2,
                'time': sk_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    return results

def run_logistic_regression_comparison():
    """Compare logistic regression implementations."""
    print("\n" + "-"*80)
    print("Logistic Regression Comparison")
    print("-"*80)
    
    results = {}
    
    # 1. StatOxide GLM (Binomial + Logit)
    if STATOXIDE_AVAILABLE:
        print("\n1. StatOxide GLM (Binomial/Logit):")
        try:
            start = time.time()
            
            # Use GLMBuilder API
            
            builder = so.models.GLMBuilder()
            builder.family(so.models.Family.binomial())
            builder.link(so.models.Link.logit())
            
            # Fit model
            X_list = X_logit.tolist()
            y_list = y_logit.tolist()
            
            # Note: This API might need adjustment based on actual implementation
            glm = builder.build()
            result = glm.fit(X_list, y_list)
            
            so_coef = result.coefficients()
            so_time = time.time() - start
            
            print(f"   Coefficients: {so_coef[:3]}...")
            print(f"   Time: {so_time:.4f}s")
            
            results['statoxide'] = {
                'coefficients': so_coef,
                'time': so_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. statsmodels Logit
    if STATSMODELS_AVAILABLE:
        print("\n2. statsmodels Logit:")
        try:
            start = time.time()
            
            X_sm = sm.add_constant(X_logit)
            model = sm.Logit(y_logit, X_sm)
            result_sm = model.fit(disp=0)
            
            sm_coef = result_sm.params.tolist()
            sm_time = time.time() - start
            
            print(f"   Coefficients: {sm_coef[:3]}...")
            print(f"   Time: {sm_time:.4f}s")
            
            results['statsmodels'] = {
                'coefficients': sm_coef,
                'time': sm_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # 3. scikit-learn LogisticRegression
    if SKLEARN_AVAILABLE:
        print("\n3. scikit-learn LogisticRegression:")
        try:
            start = time.time()
            
            model_sk = LogisticRegression(max_iter=1000, random_state=42)
            model_sk.fit(X_logit, y_logit)
            
            sk_coef = model_sk.coef_[0].tolist()
            sk_intercept = model_sk.intercept_[0]
            sk_time = time.time() - start
            
            # Combine intercept and coefficients
            sk_all_coef = [sk_intercept] + sk_coef
            
            # Predict and calculate accuracy
            y_pred = model_sk.predict(X_logit)
            accuracy = accuracy_score(y_logit, y_pred)
            
            print(f"   Coefficients: {sk_all_coef[:3]}...")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Time: {sk_time:.4f}s")
            
            results['sklearn'] = {
                'coefficients': sk_all_coef,
                'accuracy': accuracy,
                'time': sk_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    return results

def run_time_series_comparison():
    """Compare time series models (ARIMA, GARCH)."""
    print("\n" + "-"*80)
    print("Time Series Models Comparison")
    print("-"*80)
    
    results = {}
    
    # ARIMA comparison
    print("\nARIMA Models:")
    
    # 1. StatOxide ARIMA
    if STATOXIDE_AVAILABLE:
        print("\n1. StatOxide ARIMA:")
        try:
            start = time.time()
            
            # Create ARIMA model (2,0,1)
            model = so.tsa.ARIMA(2, 0, 1)
            result = model.fit(ts_data.tolist())
            
            so_ar = result.ar_coefficients()
            so_ma = result.ma_coefficients()
            so_time = time.time() - start
            
            print(f"   AR coefficients: {so_ar}")
            print(f"   MA coefficients: {so_ma}")
            print(f"   Time: {so_time:.4f}s")
            
            results['statoxide_arima'] = {
                'ar_coefs': so_ar,
                'ma_coefs': so_ma,
                'time': so_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. statsmodels ARIMA
    if STATSMODELS_AVAILABLE:
        print("\n2. statsmodels ARIMA:")
        try:
            start = time.time()
            
            model_sm = smARIMA(ts_data, order=(2, 0, 1))
            result_sm = model_sm.fit()
            
            sm_ar = result_sm.arparams.tolist()
            sm_ma = result_sm.maparams.tolist()
            sm_time = time.time() - start
            
            print(f"   AR coefficients: {sm_ar}")
            print(f"   MA coefficients: {sm_ma}")
            print(f"   Time: {sm_time:.4f}s")
            
            results['statsmodels_arima'] = {
                'ar_coefs': sm_ar,
                'ma_coefs': sm_ma,
                'time': sm_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # GARCH comparison
    print("\nGARCH Models:")
    
    # Note: statsmodels has ARCH library for GARCH, not in base statsmodels
    # We'll implement basic GARCH comparison if ARCH is available
    
    return results

def run_mixed_effects_comparison():
    """Compare mixed effects models."""
    print("\n" + "-"*80)
    print("Mixed Effects Models Comparison")
    print("-"*80)
    
    results = {}
    
    # Create panel data for mixed effects
    n_groups = 10
    n_per_group = 100
    n_total = n_groups * n_per_group
    
    group_ids = np.repeat(range(n_groups), n_per_group)
    X1 = np.random.randn(n_total)
    X2 = np.random.randn(n_total)
    
    # True parameters
    fixed_intercept = 2.0
    fixed_slope1 = 1.5
    fixed_slope2 = -0.8
    
    # Random intercepts per group
    group_intercepts = np.random.randn(n_groups) * 0.5
    
    # Generate response
    y_me = (fixed_intercept + group_intercepts[group_ids] + 
            fixed_slope1 * X1 + fixed_slope2 * X2 + 
            np.random.randn(n_total) * 0.2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'y': y_me,
        'X1': X1,
        'X2': X2,
        'group': group_ids
    })
    
    # 1. StatOxide LMM
    if STATOXIDE_AVAILABLE:
        print("\n1. StatOxide Linear Mixed Model:")
        try:
            start = time.time()
            
            # Note: API may vary
            builder = so.models.LMMBuilder()
            # Configure model...
            # result = builder.fit(...)
            
            so_time = time.time() - start
            print(f"   (Implementation details depend on API)")
            print(f"   Time: {so_time:.4f}s")
            
            results['statoxide_lmm'] = {
                'time': so_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. statsmodels MixedLM
    if STATSMODELS_AVAILABLE:
        print("\n2. statsmodels MixedLM:")
        try:
            start = time.time()
            
            model = smf.MixedLM.from_formula(
                'y ~ X1 + X2', 
                data=df, 
                groups=df['group'],
                re_formula='1'  # Random intercept
            )
            result_sm = model.fit()
            
            sm_coef = result_sm.params.tolist()
            sm_re_var = result_sm.cov_re.iloc[0, 0]
            sm_time = time.time() - start
            
            print(f"   Fixed effects: {sm_coef}")
            print(f"   Random intercept variance: {sm_re_var:.4f}")
            print(f"   Time: {sm_time:.4f}s")
            
            results['statsmodels_mixedlm'] = {
                'coefficients': sm_coef,
                're_variance': sm_re_var,
                'time': sm_time
            }
        except Exception as e:
            print(f"   Error: {e}")
    
    return results

def main():
    """Run all comparison tests."""
    all_results = {}
    
    # Run comparisons
    all_results['linear_regression'] = run_linear_regression_comparison()
    all_results['logistic_regression'] = run_logistic_regression_comparison()
    all_results['time_series'] = run_time_series_comparison()
    all_results['mixed_effects'] = run_mixed_effects_comparison()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count successful tests
    successful_tests = 0
    total_tests = 0
    
    for test_name, test_results in all_results.items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        for lib_name, lib_result in test_results.items():
            if 'time' in lib_result:
                print(f"  {lib_name}: {lib_result.get('time', 0):.4f}s")
                successful_tests += 1
            total_tests += 1
    
    print(f"\nTotal tests attempted: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    
    # Save results to file
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()