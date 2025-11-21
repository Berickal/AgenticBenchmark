"""Script to verify the setup is correct."""
import sys
import subprocess

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ Ollama is installed and running")
            print(f"  Available models: {result.stdout[:100]}...")
            return True
        else:
            print("✗ Ollama is installed but not responding correctly")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed or not in PATH")
        print("  Please install Ollama from https://ollama.ai/")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Ollama command timed out")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = ["ollama"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            missing.append(package)
    
    return len(missing) == 0

def check_project_structure():
    """Check if project structure is correct."""
    import os
    from pathlib import Path
    
    required_files = [
        "src/__init__.py",
        "src/models/task.py",
        "src/models/benchmark.py",
        "src/agents/base_agent.py",
        "src/orchestrator.py",
    ]
    
    project_root = Path(__file__).parent.parent
    missing = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} is missing")
            missing.append(file_path)
    
    return len(missing) == 0

def main():
    """Run all checks."""
    print("Benchmark Factory Multi-Agent Framework - Setup Verification")
    print("=" * 60)
    print()
    
    print("1. Checking Python packages...")
    packages_ok = check_python_packages()
    print()
    
    print("2. Checking Ollama...")
    ollama_ok = check_ollama()
    print()
    
    print("3. Checking project structure...")
    structure_ok = check_project_structure()
    print()
    
    print("=" * 60)
    if packages_ok and ollama_ok and structure_ok:
        print("✓ All checks passed! Setup is complete.")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
