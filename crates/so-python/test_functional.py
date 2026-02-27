#!/usr/bin/env python3
"""Functional test for GLM and TSA Python bindings"""

import sys
import os

# Use the virtual environment
venv_python = os.path.join(os.path.dirname(__file__), '.venv/bin/python')
if not os.path.exists(venv_python):
    print("Virtual environment not found")
    sys.exit(1)

print("Testing so_python functional bindings...")
print("=" * 60)

try:
    import so_python as sp
    print("✅ Module imported successfully")
    
    # Test 1: Basic data structures
    print("\n1. Testing basic data structures:")
    try:
        # Create Series
        s = sp.Series("test", [1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"   Series created: name='{s.name}', len={s.len()}, mean={s.mean()}")
        
        # Create DataFrame (via from_series? Not implemented yet)
        # We'll test other functionality
    except Exception as e:
        print(f"   ⚠️ Series test failed: {e}")
    
    # Test 2: GLM Families and Links
    print("\n2. Testing GLM Families and Links:")
    families = [
        ("Gaussian", sp.Family.gaussian),
        ("Binomial", sp.Family.binomial),
        ("Poisson", sp.Family.poisson),
        ("Gamma", sp.Family.gamma),
        ("Inverse Gaussian", sp.Family.inverse_gaussian),
    ]
    
    for name, factory in families:
        try:
            family = factory()
            print(f"   ✅ {name}: {family.name()}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    links = [
        ("Identity", sp.Link.identity),
        ("Logit", sp.Link.logit),
        ("Probit", sp.Link.probit),
        ("Log", sp.Link.log),
        ("Inverse", sp.Link.inverse),
    ]
    
    for name, factory in links:
        try:
            link = factory()
            print(f"   ✅ {name}: {link}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    # Test 3: GLM Builder
    print("\n3. Testing GLM Builder:")
    try:
        builder = sp.GLMBuilder()
        builder.family(sp.Family.gaussian())
        builder.link(sp.Link.identity())
        builder.intercept(True)
        builder.max_iter(100)
        builder.tol(1e-6)
        builder.scale(1.0)
        
        glm = builder.build()
        print(f"   ✅ GLM created: {glm}")
    except Exception as e:
        print(f"   ❌ GLM Builder failed: {e}")
    
    # Test 4: TimeSeries
    print("\n4. Testing TimeSeries:")
    try:
        values = list(range(1, 21))
        dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
        ts = sp.TimeSeries.from_vectors(values, dates)
        print(f"   ✅ TimeSeries created: len={ts.len()}, mean={ts.mean()}")
        print(f"   ✅ TimeSeries stats: std={ts.std(1.0)}, var={ts.var(1.0)}")
    except Exception as e:
        print(f"   ❌ TimeSeries failed: {e}")
    
    # Test 5: ARIMA Builder
    print("\n5. Testing ARIMA Builder:")
    try:
        arima = sp.ARIMA(1, 0, 1)
        arima.with_constant(True)
        arima.method("css")
        arima.max_iter(100)
        arima.tol(1e-6)
        print(f"   ✅ ARIMA created")
    except Exception as e:
        print(f"   ❌ ARIMA failed: {e}")
    
    # Test 6: GARCH Builder
    print("\n6. Testing GARCH Builder:")
    try:
        garch = sp.GARCH(1, 1)
        garch.distribution("normal")
        print(f"   ✅ GARCH created")
        
        # Test ARCH as well
        arch = sp.GARCH.arch(1)
        print(f"   ✅ ARCH created")
    except Exception as e:
        print(f"   ❌ GARCH failed: {e}")
    
    # Test 7: Check available functions
    print("\n7. Available top-level functions:")
    functions = [attr for attr in dir(sp) if not attr.startswith('_')]
    functions.sort()
    print(f"   Total {len(functions)} public attributes")
    
    # Categorize
    classes = []
    functions_list = []
    for attr in functions:
        try:
            obj = getattr(sp, attr)
            if isinstance(obj, type):
                classes.append(attr)
            else:
                functions_list.append(attr)
        except:
            functions_list.append(attr)
    
    print(f"   Classes: {len(classes)}")
    print(f"   Functions: {len(functions_list)}")
    
    print("\n" + "=" * 60)
    print("✅ All functional tests completed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)