#!/usr/bin/env python3
"""Test script to verify statoxide Python module structure"""

import sys
import os

# Add the build directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test different import styles
    print("Testing statoxide Python module imports...")
    
    # Test 1: Direct import
    import statoxide
    print("✓ Successfully imported statoxide")
    print(f"  Version: {statoxide.version()}")
    
    # Test 2: Import core classes
    from statoxide import DataFrame, Series, Formula
    print("✓ Successfully imported DataFrame, Series, Formula from top level")
    
    # Create test data
    df = DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "group": [1.0, 1.0, 2.0, 2.0, 2.0]
    })
    print(f"✓ Created DataFrame: {df}")
    
    # Test 3: Access submodules
    print("\nTesting submodule access:")
    
    # Stats submodule
    import statoxide.stats as stats
    mean_x = stats.mean([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"  stats.mean: {mean_x}")
    
    # Models submodule
    import statoxide.models as models
    print(f"  models module available")
    
    # TSA submodule  
    import statoxide.tsa as tsa
    print(f"  tsa module available")
    
    # Utils submodule
    import statoxide.utils as utils
    train, test = utils.train_test_split([1.0, 2.0, 3.0, 4.0, 5.0], 0.2)
    print(f"  utils.train_test_split: train={len(train)}, test={len(test)}")
    
    # Test 4: Top-level functions
    print("\nTesting top-level functions:")
    mean_val = statoxide.mean([1.0, 2.0, 3.0])
    print(f"  statoxide.mean: {mean_val}")
    
    corr = statoxide.correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    print(f"  statoxide.correlation: {corr}")
    
    # Test 5: Mixed effects model (placeholder)
    print("\nTesting mixed effects model (placeholder):")
    result = models.mixed_effects(df, "y ~ x + (1 | group)")
    print(f"  Mixed effects result keys: {list(result.keys())}")
    
    print("\n✅ All imports successful!")
    print("\nAvailable import styles:")
    print("1. import statoxide")
    print("2. from statoxide import DataFrame, Series, Formula")
    print("3. import statoxide.stats, statoxide.models, etc.")
    print("4. Direct function calls: statoxide.mean(), statoxide.correlation()")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nNote: You need to build the module first with:")
    print("  cd crates/so-python")
    print("  maturin develop")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)