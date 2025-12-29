#!/usr/bin/env python3
"""
Verify bahwa semua configuration files sudah correctly setup
"""

import os
from pathlib import Path

def check_configuration_setup():
    """Verify semua configuration files sudah setup dengan benar"""
    
    print("🔍 VERIFYING METRICS LOGGING SETUP")
    print("=" * 70)
    
    configs = {
        'deepseek/configuration-1.py': 'deepseek/metrics_log.csv',
        'deepseek/configuration-2.py': 'deepseek/metrics_log.csv',
        'deepseek/configuration-3.py': 'deepseek/metrics_log.csv',
        'local_rule/configuration-1.py': 'local_rule/metrics_log.csv',
        'local_rule/configuration-2.py': 'local_rule/metrics_log.csv',
        'local_rule/configuration-3.py': 'local_rule/metrics_log.csv',
    }
    
    print("\n📝 Checking configuration files...")
    print("-" * 70)
    
    all_good = True
    for config_file, expected_log in configs.items():
        config_path = Path(config_file)
        
        if not config_path.exists():
            print(f"❌ {config_file} - NOT FOUND")
            all_good = False
            continue
        
        # Read and check content
        content = config_path.read_text(encoding='utf-8')
        
        # Check for required setup code
        checks = {
            'import os': 'import os' in content,
            'script_dir = os.path.dirname': 'script_dir = os.path.dirname' in content,
            'log_file_path = os.path.join': 'log_file_path = os.path.join' in content,
            'MetricsMonitor(log_file=log_file_path)': 'MetricsMonitor(log_file=log_file_path)' in content,
            'start_intent_classification': 'start_intent_classification' in content,
            'end_intent_classification': 'end_intent_classification' in content,
        }
        
        all_passed = all(checks.values())
        
        if all_passed:
            print(f"✅ {config_file}")
            print(f"   └─ Will log to: {expected_log}")
        else:
            print(f"⚠️  {config_file}")
            for check, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"   {status} {check}")
            all_good = False
    
    print("\n📊 Checking metrics files...")
    print("-" * 70)
    
    # Check CSV files
    csv_files = {
        'deepseek/metrics_log.csv': 'DeepSeek',
        'local_rule/metrics_log.csv': 'Local Rule',
        'logs/metrics_log.csv': 'Root (should be empty)',
    }
    
    for csv_file, label in csv_files.items():
        csv_path = Path(csv_file)
        
        if csv_path.exists():
            size = csv_path.stat().st_size
            lines = len(csv_path.read_text(encoding='utf-8').splitlines())
            print(f"✅ {csv_file}")
            print(f"   └─ {label} | {size} bytes | {lines} lines")
        else:
            print(f"⚠️  {csv_file} - NOT FOUND (will be created on first run)")
    
    print("\n" + "=" * 70)
    
    if all_good:
        print("✨ ALL CHECKS PASSED - Setup is correct!")
        print("\n🚀 Ready to run:")
        print("   python deepseek/configuration-1.py")
        print("   python local_rule/configuration-1.py")
        print("\n📊 Then compare with:")
        print("   python compare_metrics.py")
    else:
        print("⚠️  Some issues found - please review above")
    
    print("=" * 70)

if __name__ == "__main__":
    check_configuration_setup()
