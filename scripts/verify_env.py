#!/usr/bin/env python3
"""
Environment verification script for MSRK v3.
Modular verification functions that can be imported and reused.
"""

import sys
import importlib
from typing import Dict, List, Tuple, Optional

class EnvironmentVerifier:
    """Modular environment verification with detailed reporting."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def verify_python_version(self, min_version: Tuple[int, int] = (3, 8)) -> bool:
        """Verify Python version meets minimum requirements."""
        current_version = sys.version_info[:2]
        success = current_version >= min_version
        
        self.results['python_version'] = {
            'success': success,
            'current': f"{current_version[0]}.{current_version[1]}",
            'minimum': f"{min_version[0]}.{min_version[1]}",
            'message': f"Python {current_version[0]}.{current_version[1]}" + 
                      ("" if success else f" (minimum {min_version[0]}.{min_version[1]} required)")
        }
        return success
    
    def verify_package_import(self, package_name: str, import_path: Optional[str] = None) -> bool:
        """Verify a package can be imported successfully."""
        try:
            if import_path:
                # Import specific module/class
                module_name, class_name = import_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                getattr(module, class_name)
            else:
                # Import package directly
                importlib.import_module(package_name)
            
            self.results[package_name] = {
                'success': True,
                'message': f"✅ {package_name} imported successfully"
            }
            return True
            
        except ImportError as e:
            self.results[package_name] = {
                'success': False,
                'message': f"❌ {package_name} import failed: {e}",
                'error': str(e)
            }
            self.errors.append(f"{package_name}: {e}")
            return False
        except Exception as e:
            self.results[package_name] = {
                'success': False,
                'message': f"❌ {package_name} verification failed: {e}",
                'error': str(e)
            }
            self.errors.append(f"{package_name}: {e}")
            return False
    
    def verify_core_packages(self) -> bool:
        """Verify core ML and data science packages."""
        core_packages = [
            'pandas',
            'numpy',
            'torch',
            'transformers',
            'sklearn',
            'matplotlib'
        ]
        
        all_success = True
        for package in core_packages:
            success = self.verify_package_import(package)
            all_success = all_success and success
        
        return all_success
    
    def verify_database_packages(self) -> bool:
        """Verify database-related packages."""
        db_packages = [
            ('prisma', 'prisma.Prisma'),
            'psycopg2',
        ]
        
        all_success = True
        for package_info in db_packages:
            if isinstance(package_info, tuple):
                package, import_path = package_info
                success = self.verify_package_import(package, import_path)
            else:
                success = self.verify_package_import(package_info)
            all_success = all_success and success
        
        return all_success
    
    def verify_data_api_packages(self) -> bool:
        """Verify data API packages."""
        api_packages = [
            'fredapi',
            'yfinance',
            'requests'
        ]
        
        all_success = True
        for package in api_packages:
            success = self.verify_package_import(package)
            all_success = all_success and success
        
        return all_success
    
    def run_full_verification(self) -> Dict:
        """Run complete environment verification."""
        print("🔍 Running MSRK v3 Environment Verification...")
        print("-" * 50)
        
        # Verify Python version
        python_ok = self.verify_python_version((3, 8))
        print(self.results['python_version']['message'])
        
        # Verify core packages
        print("\n📦 Core ML Packages:")
        core_ok = self.verify_core_packages()
        
        # Verify database packages
        print("\n🗄️ Database Packages:")
        db_ok = self.verify_database_packages()
        
        # Verify data API packages
        print("\n📊 Data API Packages:")
        api_ok = self.verify_data_api_packages()
        
        # Print individual results
        for package, result in self.results.items():
            if package != 'python_version':
                print(result['message'])
        
        # Summary
        all_ok = python_ok and core_ok and db_ok and api_ok
        print("\n" + "=" * 50)
        
        if all_ok:
            print("🎉 All verifications passed! Environment ready for MSRK v3.")
        else:
            print("⚠️ Some verifications failed. See details above.")
            print(f"\nErrors found: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        return {
            'success': all_ok,
            'results': self.results,
            'errors': self.errors
        }

def verify_environment() -> bool:
    """Main verification function - can be imported and used elsewhere."""
    verifier = EnvironmentVerifier()
    result = verifier.run_full_verification()
    return result['success']

def verify_minimal() -> bool:
    """Minimal verification for torch and prisma as requested in TODO."""
    try:
        import torch
        from prisma import Prisma
        print("✅ Minimal verification passed: torch and prisma imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Minimal verification failed: {e}")
        return False

if __name__ == "__main__":
    # Run verification based on command line args
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        success = verify_minimal()
    else:
        success = verify_environment()
    
    sys.exit(0 if success else 1)
