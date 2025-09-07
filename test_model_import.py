#!/usr/bin/env python
"""Test model import without loading JAX dependencies."""

import sys
sys.path.insert(0, '.')

def test_model_import():
    """Test if model import structure is correct."""
    try:
        # Test the import path that was failing
        import importlib.util
        spec = importlib.util.spec_from_file_location("factory", "models/factory.py")
        if spec is None:
            print("❌ Could not load factory module spec")
            return False
        
        # Check if create_model is available in the factory module
        import ast
        import inspect
        
        with open("models/factory.py", "r") as f:
            content = f.read()
        
        # Parse the file to check for create_model method
        tree = ast.parse(content)
        
        has_create_model = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "create_model":
                has_create_model = True
                print("✅ Found create_model method in factory")
                break
            elif isinstance(node, ast.ClassDef) and node.name == "ModelFactory":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "create_model":
                        has_create_model = True
                        print("✅ Found ModelFactory.create_model method")
                        break
        
        if not has_create_model:
            print("❌ create_model method not found in factory")
            return False
        
        # Check if __init__.py exposes create_model
        with open("models/__init__.py", "r") as f:
            init_content = f.read()
        
        if "def create_model" in init_content and "create_model" in init_content:
            print("✅ create_model exposed in __init__.py")
        else:
            print("❌ create_model not properly exposed in __init__.py")
            return False
        
        print("✅ Model import structure looks correct")
        print("   (JAX dependency will be resolved in Colab environment)")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model import: {e}")
        return False

if __name__ == "__main__":
    success = test_model_import()
    if success:
        print("\n🎉 Model factory should work in Colab!")
    else:
        print("\n❌ Model factory needs more fixes")