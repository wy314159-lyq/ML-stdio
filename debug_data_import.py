#!/usr/bin/env python3
"""
Debug script for data import issues
Tests data loading without GUI dependencies
"""

import os
import sys

def test_basic_imports():
    """Test basic imports"""
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
        return True
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False

def test_sample_data():
    """Test loading sample data"""
    try:
        import pandas as pd
        
        # Check if sample data exists
        if not os.path.exists('sample_data.csv'):
            print("✗ sample_data.csv not found")
            return False
        
        # Try to load sample data
        df = pd.read_csv('sample_data.csv')
        print(f"✓ Sample data loaded successfully")
        print(f"  Type: {type(df)}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:5])}...")  # Show first 5 columns
        
        # Test that it's actually a DataFrame
        if isinstance(df, pd.DataFrame):
            print("✓ Data is correctly loaded as DataFrame")
            return True
        else:
            print(f"✗ Data loaded as {type(df)} instead of DataFrame")
            return False
            
    except Exception as e:
        print(f"✗ Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_utils():
    """Test our data utility functions"""
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from utils.data_utils import load_csv_file, get_data_quality_report
        
        print("✓ Data utils imported successfully")
        
        # Test load_csv_file
        df = load_csv_file('sample_data.csv')
        print(f"✓ load_csv_file returned: {type(df)}")
        
        if isinstance(df, dict):
            print("✗ ERROR: load_csv_file returned dict instead of DataFrame!")
            print(f"  Data: {df}")
            return False
            
        # Test get_data_quality_report
        report = get_data_quality_report(df)
        print(f"✓ get_data_quality_report returned: {type(report)}")
        
        if not isinstance(report, dict):
            print("✗ ERROR: get_data_quality_report should return dict!")
            return False
            
        print("✓ All data utils working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error testing data utils: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("MatSci-ML Studio Data Import Debug")
    print("=" * 40)
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Sample data loading", test_sample_data),
        ("Data utilities", test_data_utils),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Data import should work correctly.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 