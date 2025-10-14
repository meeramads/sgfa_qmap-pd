#!/usr/bin/env python
"""Quick config validation test."""
import sys
import yaml
from core.config_utils import validate_configuration

# Load config
with open('config_synthetic.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Config loaded:')
print('  Dataset:', config.get('data', {}).get('dataset'))
print('  Data dir:', config.get('data', {}).get('data_dir', 'None'))

# Validate
try:
    validated = validate_configuration(config)
    print('\n[PASS] Validation succeeded!')
    print('  Validated dataset:', validated.get('data', {}).get('dataset'))
    print('  Validated data_dir:', validated.get('data', {}).get('data_dir', 'None'))
except Exception as e:
    print('\n[FAIL] Validation failed:', str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
