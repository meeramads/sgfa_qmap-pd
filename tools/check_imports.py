#!/usr/bin/env python
"""
Comprehensive import checker for the SGFA codebase.
Finds broken imports and suggests fixes.
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImportChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.python_files = []
        self.broken_imports = []
        self.successful_imports = []
        
        # Known project modules
        self.project_modules = set()
        self._find_project_modules()
        
    def _find_project_modules(self):
        """Find all Python modules in the project."""
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            root_path = Path(root)
            rel_path = root_path.relative_to(self.project_root)
            
            # Skip certain directories
            if any(part in ['.git', '__pycache__', '.pytest_cache', 'experiment_results'] 
                   for part in rel_path.parts):
                continue
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = root_path / file
                    self.python_files.append(file_path)
                    
                    # Add module path to known modules
                    if file == '__init__.py':
                        if rel_path != Path('.'):
                            module_name = '.'.join(rel_path.parts)
                            self.project_modules.add(module_name)
                    else:
                        module_parts = list(rel_path.parts) + [file[:-3]]
                        if module_parts[0] == '.':
                            module_parts = module_parts[1:]
                        if module_parts:
                            module_name = '.'.join(module_parts)
                            self.project_modules.add(module_name)

    def _extract_imports(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Extract all imports from a Python file."""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno, f"import {alias.name}"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append((node.module, node.lineno, 
                                      f"from {node.module} import {', '.join(alias.name for alias in node.names)}"))
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
        
        return imports

    def _is_project_import(self, module_name: str) -> bool:
        """Check if this is a project-internal import."""
        # Split module name and check against known project modules
        parts = module_name.split('.')
        
        # Check exact match
        if module_name in self.project_modules:
            return True
        
        # Check if it's a submodule of a known module
        for i in range(len(parts)):
            parent = '.'.join(parts[:i+1])
            if parent in self.project_modules:
                return True
        
        # Check for specific project directories
        project_dirs = {'core', 'analysis', 'experiments', 'models', 'data', 
                       'visualization', 'performance', 'tests', 'tools'}
        if parts[0] in project_dirs:
            return True
        
        return False

    def _test_import(self, module_name: str, file_path: Path) -> Tuple[bool, str]:
        """Test if an import actually works."""
        try:
            # Save current sys.path
            original_path = sys.path.copy()
            
            # Add project root and current file's directory to path
            sys.path.insert(0, str(self.project_root))
            sys.path.insert(0, str(file_path.parent))
            
            # Try to import
            importlib.import_module(module_name)
            return True, "OK"
            
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {e}"
        finally:
            # Restore sys.path
            sys.path[:] = original_path

    def check_all_imports(self) -> Dict:
        """Check all imports in all Python files."""
        results = {
            'total_files': len(self.python_files),
            'total_imports': 0,
            'broken_imports': [],
            'successful_imports': 0,
            'project_imports': 0,
            'external_imports': 0
        }
        
        logger.info(f"Checking imports in {len(self.python_files)} Python files...")
        
        for file_path in self.python_files:
            rel_path = file_path.relative_to(self.project_root)
            logger.debug(f"Checking {rel_path}")
            
            imports = self._extract_imports(file_path)
            results['total_imports'] += len(imports)
            
            for module_name, line_no, import_stmt in imports:
                # Skip relative imports (.) and built-in modules
                if module_name.startswith('.') or module_name in {'os', 'sys', 'pathlib', 'logging', 'typing'}:
                    continue
                
                is_project = self._is_project_import(module_name)
                
                if is_project:
                    results['project_imports'] += 1
                    # Test project imports
                    success, error = self._test_import(module_name, file_path)
                    
                    if success:
                        results['successful_imports'] += 1
                    else:
                        results['broken_imports'].append({
                            'file': str(rel_path),
                            'line': line_no,
                            'import': import_stmt,
                            'module': module_name,
                            'error': error
                        })
                else:
                    results['external_imports'] += 1
                    # Count external imports as successful (assume they're installed)
                    results['successful_imports'] += 1
        
        return results

    def suggest_fixes(self, broken_import: Dict) -> List[str]:
        """Suggest fixes for broken imports."""
        suggestions = []
        module_name = broken_import['module']
        
        # Common fixes based on our reorganization
        fixes = {
            'utils': 'core.utils',
            'get_data': 'core.get_data', 
            'run_analysis': 'core.run_analysis',
            'visualization': 'core.visualization',
        }
        
        if module_name in fixes:
            suggestions.append(f"Change 'from {module_name}' to 'from {fixes[module_name]}'")
        
        # Check if it's a partial match
        for old, new in fixes.items():
            if module_name.startswith(old + '.'):
                new_module = module_name.replace(old, new, 1)
                suggestions.append(f"Change '{module_name}' to '{new_module}'")
        
        # Look for similar module names
        similar = [m for m in self.project_modules if module_name.split('.')[-1] in m]
        if similar:
            suggestions.extend([f"Did you mean '{m}'?" for m in similar[:3]])
        
        return suggestions

def main():
    project_root = Path(__file__).parent.parent
    checker = ImportChecker(project_root)
    
    print("Check: Comprehensive Import Check")
    print("=" * 50)
    
    results = checker.check_all_imports()
    
    print(f"Stats: Summary:")
    print(f"   Files checked: {results['total_files']}")
    print(f"   Total imports: {results['total_imports']}")
    print(f"   Project imports: {results['project_imports']}")
    print(f"   External imports: {results['external_imports']}")
    print(f"   Successful imports: {results['successful_imports']}")
    print(f"   Broken imports: {len(results['broken_imports'])}")
    
    if results['broken_imports']:
        print(f"\nFAILED: Broken Imports Found:")
        print("-" * 50)
        
        for i, broken in enumerate(results['broken_imports'], 1):
            print(f"{i}. {broken['file']}:{broken['line']}")
            print(f"   Import: {broken['import']}")
            print(f"   Error: {broken['error']}")
            
            suggestions = checker.suggest_fixes(broken)
            if suggestions:
                print(f"   Suggestions:")
                for suggestion in suggestions:
                    print(f"     • {suggestion}")
            print()
    else:
        print(f"\nPASSED: No broken imports found!")
        print("   All project imports are working correctly.")

    # Summary
    success_rate = (results['successful_imports'] / results['total_imports'] * 100) if results['total_imports'] > 0 else 100
    print(f"\nResults: Import Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    main()