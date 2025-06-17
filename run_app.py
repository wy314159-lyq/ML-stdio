#!/usr/bin/env python3
"""
MatSci-ML Studio - Test Runner
Quick test runner for the application
"""

import sys
import os
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'PyQt5', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main function to run the application"""
    print("MatSci-ML Studio v1.0")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("All dependencies found.")
    print("Starting application...")
    
    try:
        from ui.main_window import main as run_main_window
        run_main_window()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 