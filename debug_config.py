import sys
sys.path.insert(0, '.')

import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=== CONFIG LOADED ===")
print(f"sgfa_configuration_comparison section exists: {'sgfa_configuration_comparison' in config}")
print(f"parameter_ranges: {config.get('sgfa_configuration_comparison', {}).get('parameter_ranges', {})}")
print(f"n_factors: {config.get('sgfa_configuration_comparison', {}).get('parameter_ranges', {}).get('n_factors', 'NOT FOUND')}")

# Now test what the actual experiment code does
from core.config_utils import ConfigHelper

config_dict = ConfigHelper.to_dict(config)
sgfa_config = config_dict.get("sgfa_configuration_comparison", {})
parameter_ranges = sgfa_config.get("parameter_ranges", {})

print("\n=== AFTER ConfigHelper.to_dict() ===")
print(f"sgfa_config exists: {bool(sgfa_config)}")
print(f"parameter_ranges: {parameter_ranges}")
print(f"n_factors from parameter_ranges: {parameter_ranges.get('n_factors', 'NOT FOUND')}")

# Show what K_values would be
K_values = sorted(parameter_ranges.get("n_factors", [2, 3, 4]))
print(f"\n=== FINAL K_VALUES ===")
print(f"K_values: {K_values}")
