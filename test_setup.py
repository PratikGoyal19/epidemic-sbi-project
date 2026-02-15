"""
Test that everything is installed correctly
Run this: python test_setup.py
"""

import sys

def test_all_packages():
    """Test all required packages are installed"""
    
    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'tqdm': 'Progress bars',
        'sklearn': 'Scikit-learn',
    }
    
    print("="*50)
    print("TESTING PACKAGE INSTALLATION")
    print("="*50 + "\n")
    
    all_good = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:20s} - Installed")
        except ImportError:
            print(f"❌ {name:20s} - MISSING")
            all_good = False
    
    # Test BayesFlow separately (different import name)
    try:
        import bayesflow
        print(f"✅ {'BayesFlow':20s} - Installed")
    except ImportError:
        print(f"❌ {'BayesFlow':20s} - MISSING")
        all_good = False
    
    print("\n" + "="*50)
    
    if all_good:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("✅ You are ready to start coding!")
    else:
        print("⚠️  SOME PACKAGES ARE MISSING")
        print("Run: pip install -r requirements.txt")
    
    print("="*50)
    
    return all_good

if __name__ == "__main__":
    success = test_all_packages()
    sys.exit(0 if success else 1)