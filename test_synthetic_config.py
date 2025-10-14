#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test synthetic config validation."""
import yaml
from core.config_utils import validate_configuration

# Load config
with open('config_synthetic.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Validate
try:
    validated = validate_configuration(config)
    print('[PASS] Configuration validation PASSED')
    dataset = validated['data']['dataset']
    data_dir = validated['data'].get('data_dir', 'None')
    print('Dataset:', dataset)
    print('Data dir:', data_dir)
except Exception as e:
    print('[FAIL] Validation failed:', e)
    import traceback
    traceback.print_exc()
