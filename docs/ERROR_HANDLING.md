# Error Handling Best Practices

## Overview

This document describes standardized error handling patterns for the SGFA qMAP-PD codebase. Consistent error handling improves debugging, user experience, and code maintainability.

## Core Principles

1. **Always log errors** with context
2. **Return structured error dictionaries** instead of None
3. **Use specific exception types** for different failure modes
4. **Include enough context** for debugging
5. **Don't silently swallow exceptions** unless explicitly intended

## Standard Error Result Format

All functions that can fail should return dictionaries with this structure:

```python
{
    "status": "failed",  # or "completed"
    "error": "Error message",
    "error_type": "ValueError",  # Exception class name
    "error_context": "Additional context",  # Optional
    # ... other function-specific fields
}
```

## Utilities

Use utilities from `core.error_handling`:

### Creating Error Results

```python
from core.error_handling import create_error_result, log_and_return_error

# Simple error result
error_dict = create_error_result(
    exception,
    context="Loading data",
    additional_fields={"data_loaded": False}
)

# Log and return error
error_dict = log_and_return_error(
    exception,
    logger,
    context="MCMC execution",
    log_level="error",
    include_traceback=False
)
```

### Exception Types

Use specific exception types for clear error signaling:

```python
from core.error_handling import (
    ConfigurationError,    # Invalid configuration
    DataValidationError,   # Data validation failed
    ConvergenceError,      # MCMC didn't converge
    ModelExecutionError,   # Model execution failed
)

# Example
if K < 1 or K > 50:
    raise ConfigurationError(f"K must be between 1 and 50, got {K}")
```

### Safe Execution

```python
from core.error_handling import safe_execute

# Execute with error handling
result = safe_execute(
    risky_function,
    arg1, arg2,
    logger_instance=logger,
    context="Processing data",
    default_return={}
)
```

### Decorator for Error Handling

```python
from core.error_handling import handle_errors

@handle_errors(return_dict=True, log_level="error")
def my_function(data, logger=None):
    # Function implementation
    # Errors automatically caught, logged, and returned as dict
    pass
```

## Patterns by Use Case

### Pattern 1: Data Loading/Processing

```python
def load_data(filepath: str, logger=None) -> Tuple[Optional[Data], Dict]:
    """Load data from file."""
    from core.error_handling import log_and_return_error

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        data = read_file(filepath)
        validate_data(data)
        return data, {"status": "completed", "filepath": filepath}
    except Exception as e:
        error_info = log_and_return_error(
            e, logger, f"Loading data from {filepath}"
        )
        return None, error_info
```

### Pattern 2: Analysis Execution

```python
def run_analysis(X_list: List[np.ndarray], config: Dict, logger=None) -> Dict:
    """Run SGFA analysis."""
    from core.error_handling import create_error_result, create_success_result

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Validate inputs
        if not X_list:
            raise DataValidationError("Empty data list")

        # Run analysis
        results = execute_mcmc(X_list, config)

        # Return success
        return create_success_result(
            results=results,
            execution_time=elapsed_time
        )

    except ConvergenceError as e:
        logger.warning(f"⚠️  Convergence issue: {e}")
        return create_error_result(e, "MCMC convergence")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return create_error_result(e, "SGFA analysis execution")
```

### Pattern 3: Validation

```python
from core.error_handling import validate_required_keys, validate_convergence

def process_results(results: Dict) -> Dict:
    """Process MCMC results."""
    # Validate required keys
    validate_required_keys(
        results,
        ["samples", "mcmc", "convergence"],
        context="MCMC results"
    )

    # Validate convergence
    validate_convergence(
        results["convergence"],
        min_ess=100.0,
        max_rhat=1.1
    )

    # Process results...
```

### Pattern 4: Component Initialization

```python
def create_components(config: Dict, logger=None) -> Tuple[Optional[Component], Dict]:
    """Create analysis components."""
    from core.error_handling import log_and_return_error, create_success_result

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        component = Component(config)
        return component, create_success_result(component_type="Component")

    except Exception as e:
        error_info = log_and_return_error(
            e, logger, "Component initialization",
            additional_fields={"component_created": False}
        )
        return None, error_info
```

## Migration Guide

### Before (Inconsistent)

```python
try:
    result = do_something()
    return result
except Exception as e:
    logger.error(f"Failed: {e}")
    return {"error": str(e)}  # Missing status, error_type
```

### After (Standardized)

```python
from core.error_handling import log_and_return_error, create_success_result

try:
    result = do_something()
    return create_success_result(result=result)
except Exception as e:
    return log_and_return_error(e, logger, "Operation description")
```

## Common Mistakes to Avoid

### ❌ Don't: Silent failures

```python
try:
    result = risky_operation()
except:
    pass  # Error silently swallowed
```

### ✅ Do: Log and handle

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return create_error_result(e, "risky_operation")
```

### ❌ Don't: Inconsistent error formats

```python
return {"error": "something went wrong"}  # Missing standard fields
```

### ✅ Do: Use standard format

```python
return create_error_result(exception, context)
```

### ❌ Don't: Catch Exception without logging

```python
try:
    do_something()
except Exception:
    return None  # No indication why it failed
```

### ✅ Do: Log with context

```python
try:
    do_something()
except Exception as e:
    return log_and_return_error(e, logger, "Doing something")
```

## Testing Error Handling

```python
def test_error_handling():
    """Test that errors are handled properly."""
    # Test invalid input
    result = my_function(invalid_data)

    assert result["status"] == "failed"
    assert "error" in result
    assert "error_type" in result

    # Test that valid input succeeds
    result = my_function(valid_data)
    assert result["status"] == "completed"
```

## Summary

- **Use `core.error_handling` utilities** for consistent error handling
- **Always return structured error dictionaries**
- **Log errors with context** using `log_and_return_error()`
- **Use specific exception types** for different failure modes
- **Validate inputs early** using validation utilities
- **Include `status` field** in all result dictionaries

Following these patterns makes errors easier to debug, handle, and test.
