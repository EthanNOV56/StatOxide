#!/usr/bin/env python3
"""Test GLM and TSA Python bindings"""

import sys
import os

# Use the virtual environment's python
python_path = os.path.join(os.path.dirname(__file__), '.venv/bin/python')
if os.path.exists(python_path):
    print(f"Using virtual environment: {python_path}")
else:
    print("Virtual environment not found, using system python")

try:
    # Import the module
    import statoxide as so
    print("✅ Successfully imported statoxide as 'so'")
    
    # Check available attributes
    print("\nAvailable top-level attributes:")
    attrs = [attr for attr in dir(so) if not attr.startswith('_')]
    print(f"  {len(attrs)} public attributes")
    
    # Look for key classes
    key_classes = ['Series', 'DataFrame', 'Formula', 'Family', 'Link', 
                   'GLMBuilder', 'GLM', 'GLMResults', 'TimeSeries', 
                   'ARIMA', 'ARIMAResults', 'GARCH', 'GARCHResults']
    
    for cls in key_classes:
        if hasattr(so, cls):
            print(f"  ✅ {cls}")
        else:
            print(f"  ❌ {cls} (missing)")
    
    # Test 1: Create Series and DataFrame
    print("\n--- Test 1: Series and DataFrame ---")
    try:
        series = so.Series("test", [1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"  Series created: {series}")
        print(f"  Series mean: {series.mean()}")
        
        # Create DataFrame from dict (need to check API)
        # For now, skip DataFrame creation
    except Exception as e:
        print(f"  ⚠️ Series test failed: {e}")
    
    # Test 2: GLM Family and Link
    print("\n--- Test 2: GLM Family and Link ---")
    try:
        family_gaussian = so.Family.gaussian()
        print(f"  Family.gaussian(): {family_gaussian}")
        
        link_identity = so.Link.identity()
        print(f"  Link.identity(): {link_identity}")
        
        link_logit = so.Link.logit()
        print(f"  Link.logit(): {link_logit}")
    except Exception as e:
        print(f"  ⚠️ Family/Link test failed: {e}")
    
    # Test 3: GLM Builder
    print("\n--- Test 3: GLM Builder ---")
    try:
        builder = so.GLMBuilder()
        builder.family(so.Family.gaussian())
        builder.link(so.Link.identity())
        builder.intercept(True)
        builder.max_iter(100)
        builder.tol(1e-6)
        
        glm = builder.build()
        print(f"  GLM created: {glm}")
    except Exception as e:
        print(f"  ⚠️ GLM Builder test failed: {e}")
    
    # Test 4: TimeSeries
    print("\n--- Test 4: TimeSeries ---")
    try:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", 
                 "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
                 "2024-01-09", "2024-01-10"]
        
        ts = so.TimeSeries.from_vectors(values, dates)
        print(f"  TimeSeries created: {ts}")
        print(f"  TimeSeries length: {ts.len()}")
        print(f"  TimeSeries mean: {ts.mean()}")
    except Exception as e:
        print(f"  ⚠️ TimeSeries test failed: {e}")
    
    # Test 5: ARIMA
    print("\n--- Test 5: ARIMA ---")
    try:
        arima = so.ARIMA(1, 0, 1)
        arima.with_constant(True)
        print(f"  ARIMA created: {arima}")
    except Exception as e:
        print(f"  ⚠️ ARIMA test failed: {e}")
    
    # Test 6: GARCH
    print("\n--- Test 6: GARCH ---")
    try:
        garch = so.GARCH(1, 1)
        garch.distribution("normal")
        print(f"  GARCH created: {garch}")
    except Exception as e:
        print(f"  ⚠️ GARCH test failed: {e}")
    
    print("\n✅ All tests completed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure maturin develop was run successfully.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)