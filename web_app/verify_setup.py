"""
Verification script to check Django app setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 8)
    if version >= required:
        print(f"✅ Python {version.major}.{version.minor} (OK)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (Need 3.8+)")
        return False

def check_django_installation():
    """Check if Django is installed."""
    try:
        import django
        print(f"✅ Django {django.get_version()} installed")
        return True
    except ImportError:
        print("❌ Django not installed")
        return False

def check_required_packages():
    """Check required packages."""
    packages = [
        'torch',
        'cv2',
        'numpy',
        'ultralytics',
        'PIL',
    ]
    
    all_ok = True
    for package in packages:
        try:
            if package == 'cv2':
                import cv2 as _
            elif package == 'PIL':
                from PIL import Image as _
            else:
                __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check if all required directories exist."""
    required_dirs = [
        'ppe_detector',
        'detector',
        'ppe_detector/templates',
        'ppe_detector/static/js',
        'detector/migrations',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            all_ok = False
    
    return all_ok

def check_required_files():
    """Check if all required files exist."""
    required_files = [
        'manage.py',
        'run_server.py',
        'requirements.txt',
        'ppe_detector/settings.py',
        'ppe_detector/urls.py',
        'ppe_detector/wsgi.py',
        'detector/views.py',
        'detector/urls.py',
        'detector/inference.py',
        'detector/middleware.py',
        'ppe_detector/templates/index.html',
        'ppe_detector/static/js/app.js',
        'README.md',
        'QUICKSTART.md',
        'DEPLOYMENT.md',
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_ok = False
    
    return all_ok

def check_model_paths():
    """Check configured model paths."""
    try:
        from ppe_detector.settings import MODELS_CONFIG
        
        print("\n📦 Model Configuration:")
        all_ok = True
        for model_name, path in MODELS_CONFIG.items():
            if os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"✅ {model_name}: {path} ({size_mb:.1f} MB)")
            else:
                print(f"⚠️  {model_name}: Path not found - {path}")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("PPE DETECTION DJANGO APP - VERIFICATION")
    print("="*60 + "\n")
    
    print("🔍 Checking Environment...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Django Installation", check_django_installation),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Required Files", check_required_files),
        ("Model Paths", check_model_paths),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ Everything is ready! Run: python run_server.py")
    else:
        print("\n❌ Some checks failed. Please install missing dependencies.")
        print("   Run: pip install -r requirements.txt")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
