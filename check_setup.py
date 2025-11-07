#!/usr/bin/env python3
"""
Setup Checker for AI Labeling Script

Run this before running ai_labeling_script.py to verify everything is set up correctly.

Usage:
    python check_setup.py
"""

import os
import sys

def check_python_version():
    """Check if Python version is 3.7+"""
    print("Checking Python version...", end=" ")
    if sys.version_info >= (3, 7):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        return True
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.7+)")
        return False

def check_modules():
    """Check if required modules are installed"""
    required_modules = {
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'google.generativeai': 'google-generativeai'
    }
    
    all_good = True
    for module_name, package_name in required_modules.items():
        print(f"Checking {module_name}...", end=" ")
        try:
            __import__(module_name)
            print("✅ Installed")
        except ImportError:
            print(f"❌ Not installed")
            print(f"   Install with: pip install {package_name}")
            all_good = False
    
    return all_good

def check_files():
    """Check if required files exist"""
    print("\nChecking files...")
    
    files_to_check = {
        'ai_labeling_script.py': 'AI labeling script',
        'data/raw_data.csv': 'Raw data (from data collection)'
    }
    
    all_good = True
    for file_path, description in files_to_check.items():
        print(f"Checking {description}...", end=" ")
        if os.path.exists(file_path):
            if file_path.endswith('.csv'):
                # Check if CSV has data
                import pandas as pd
                try:
                    df = pd.read_csv(file_path)
                    print(f"✅ Found ({len(df)} posts)")
                except:
                    print(f"⚠️  Found but can't read")
                    all_good = False
            else:
                print("✅ Found")
        else:
            print("❌ Not found")
            if file_path == 'data/raw_data.csv':
                print("   Run the data collection script first!")
            all_good = False
    
    return all_good

def check_api_key():
    """Check if API key is configured"""
    print("\nChecking API key configuration...")
    print("Reading ai_labeling_script.py...", end=" ")
    
    try:
        with open('ai_labeling_script.py', 'r') as f:
            content = f.read()
            
        if 'YOUR_GEMINI_API_KEY_HERE' in content:
            print("❌ Not configured")
            print("   Edit ai_labeling_script.py and set GEMINI_API_KEY")
            return False
        elif 'GEMINI_API_KEY = ""' in content:
            print("❌ Empty")
            print("   Edit ai_labeling_script.py and set GEMINI_API_KEY")
            return False
        else:
            print("✅ Configured")
            return True
    except FileNotFoundError:
        print("❌ ai_labeling_script.py not found")
        return False

def check_data_structure():
    """Check if data directory structure is correct"""
    print("\nChecking data directory structure...")
    
    if not os.path.exists('data'):
        print("❌ 'data/' directory not found")
        print("   Create it with: mkdir data")
        return False
    
    print("✅ data/ directory exists")
    
    if os.path.exists('data/raw_data.csv'):
        print("✅ data/raw_data.csv exists")
    else:
        print("❌ data/raw_data.csv missing")
        print("   Run data collection script first")
        return False
    
    return True

def main():
    print("="*70)
    print("🔍 AI Labeling Setup Checker")
    print("="*70)
    print()
    
    checks = [
        ("Python version", check_python_version),
        ("Required modules", check_modules),
        ("Data structure", check_data_structure),
        ("Files", check_files),
        ("API key", check_api_key),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error during {check_name} check: {e}")
            results.append((check_name, False))
        print()
    
    # Summary
    print("="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    all_passed = all(result for _, result in results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print()
    
    if all_passed:
        print("🎉 All checks passed! You're ready to run:")
        print("   python ai_labeling_script.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before running the script.")
        print("\n📚 Common fixes:")
        print("   1. Install missing modules: pip install pandas tqdm google-generativeai")
        print("   2. Run data collection script first to create raw_data.csv")
        print("   3. Edit ai_labeling_script.py and set your GEMINI_API_KEY")
    
    print("="*70)

if __name__ == "__main__":
    main()
