# Testing Infrastructure

**Last Updated**: October 6, 2025

## Overview

The SGFA codebase has a comprehensive test suite covering core functionality, analysis components, experiments, and integrations. Tests use pytest as the testing framework.

## Test Statistics

- **Total Test Files**: 36
- **Test Framework**: pytest
- **Total Test Code**: ~5000+ lines
- **New Module Coverage**: 100% (all refactored modules have tests)

## Test Structure

```
tests/
├── core/                              # Core utilities tests
│   ├── test_error_handling.py         # Error handling utilities (NEW)
│   └── __init__.py
├── analysis/                          # Analysis component tests
│   ├── test_component_factory.py      # Component factory (NEW)
│   ├── test_clinical_integration.py   # Clinical integration (NEW)
│   ├── test_data_manager.py           # Data loading
│   ├── test_model_runner.py           # Model execution
│   ├── test_cross_validation.py       # CV orchestration
│   └── __init__.py
├── experiments/                       # Experiment tests
│   ├── test_clinical_validation.py    # Clinical validation
│   ├── test_data_validation.py        # Data quality
│   ├── test_model_comparison.py       # Model comparison
│   └── __init__.py
├── integration/                       # Integration tests
│   ├── test_pipeline.py               # Full pipeline
│   ├── test_brain_mapping.py          # Brain visualization
│   └── __init__.py
├── visualization/                     # Visualization tests
│   ├── test_brain_plots.py            # Brain plotting
│   ├── test_factor_plots.py           # Factor plotting
│   ├── test_manager.py                # Visualization manager
│   └── __init__.py
└── conftest.py                        # Shared fixtures
```

## Recently Added Tests

### 1. test_error_handling.py (400+ lines)

**Purpose**: Test the standardized error handling utilities

**Test Classes**:
- `TestExceptionHierarchy`: Custom exception classes
- `TestCreateErrorResult`: Error result dictionary creation
- `TestLogAndReturnError`: Error logging functionality
- `TestHandleErrorsDecorator`: Automatic error handling decorator
- `TestIntegration`: Complete error handling workflows

**Coverage**:
- Exception hierarchy (SGFAError, ConfigurationError, etc.)
- Error result creation with context
- Error logging with different logger types
- Decorator-based error handling
- Integration with real components

### 2. test_component_factory.py (500+ lines)

**Purpose**: Test the unified component creation and integration

**Test Classes**:
- `TestAnalysisMode`: Enum values and string representations
- `TestAnalysisComponents`: Dataclass functionality and methods
- `TestCreateAnalysisComponents`: Factory function for components
- `TestIntegrateAnalysisWithPipeline`: Pipeline integration
- `TestRunSGFAWithComponents`: SGFA execution with components
- `TestPrepareExperimentData`: Data preparation workflows
- `TestQuickSGFARun`: Convenience function
- `TestIntegration`: Complete analysis workflows

**Coverage**:
- AnalysisMode enum (STRUCTURED, FALLBACK, BASIC)
- AnalysisComponents dataclass properties
- Component creation with configurations
- Data loading and preparation
- Analysis execution
- Fallback mechanisms
- Error handling

### 3. test_clinical_integration.py (500+ lines)

**Purpose**: Test clinical validation workflows and integration

**Test Classes**:
- `TestClinicalValidationComponents`: Component dataclass
- `TestCreateClinicalMetricsCalculator`: Metrics creation
- `TestCreateClinicalProcessor`: Processor creation
- `TestCreateClinicalValidationSuite`: Full suite creation
- `TestRunComprehensiveClinicalValidation`: Comprehensive workflow
- `TestRunTargetedClinicalValidation`: Targeted analysis
- `TestIntegration`: Complete clinical workflows
- `TestEdgeCases`: Edge cases and error conditions

**Coverage**:
- Clinical component creation
- Metrics calculation
- Subtype discovery
- Progression analysis
- Biomarker discovery
- Error handling in clinical context
- Empty data and edge cases

## Test Patterns

### 1. Mocking Pattern

All tests use comprehensive mocking to isolate components:

```python
from unittest.mock import Mock, patch

@patch('module.ClassName')
class TestMyComponent:
    def test_with_mock(self, mock_class):
        mock_instance = Mock()
        mock_instance.method.return_value = expected_value
        mock_class.return_value = mock_instance

        # Test code
        result = function_under_test()

        # Assertions
        mock_class.assert_called_once()
        assert result == expected_value
```

### 2. Fixture Pattern

Shared fixtures defined in `conftest.py`:

```python
@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "dataset": "qmap_pd",
        "K": 10,
        "percW": 20.0,
    }

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "X_list": [np.array([[1, 2], [3, 4]])],
        "dataset": "synthetic",
    }
```

### 3. Parametrized Testing

Testing multiple scenarios efficiently:

```python
@pytest.mark.parametrize(
    "analysis_type",
    ["metrics", "subtypes", "progression", "biomarkers"]
)
def test_targeted_analyses(self, analysis_type):
    result = run_targeted_clinical_validation(
        X_list, clinical_data, factor_results,
        analyses=[analysis_type]
    )
    assert result["status"] == "completed"
```

### 4. Error Testing Pattern

Testing error conditions and recovery:

```python
def test_error_handling(self, mock_create_suite):
    # Setup mock to raise error
    mock_suite = Mock()
    mock_suite.component.side_effect = ValueError("Error message")
    mock_create_suite.return_value = mock_suite

    # Call function
    result = function_under_test(data)

    # Verify error handling
    assert result["status"] == "failed"
    assert "error" in result
    assert "Error message" in result["error"]
```

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-cov numpy
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/analysis/test_component_factory.py
```

### Run Specific Test Class

```bash
pytest tests/analysis/test_component_factory.py::TestAnalysisComponents
```

### Run Specific Test

```bash
pytest tests/analysis/test_component_factory.py::TestAnalysisComponents::test_initialization
```

### Run with Coverage

```bash
pytest --cov=analysis --cov=core --cov=experiments tests/
```

### Run with Verbose Output

```bash
pytest -v tests/
```

### Run Tests Matching Pattern

```bash
pytest -k "error_handling" tests/
```

## Test Coverage by Module

### Core (tests/core/)

| Module | Test File | Status |
|--------|-----------|--------|
| error_handling.py | test_error_handling.py | ✅ Complete |
| model_interface.py | - | ⚠️ Needs tests |
| parameter_resolver.py | - | ⚠️ Needs tests |
| logger_utils.py | - | ⚠️ Needs tests |
| config_utils.py | - | ⚠️ Needs tests |

### Analysis (tests/analysis/)

| Module | Test File | Status |
|--------|-----------|--------|
| component_factory.py | test_component_factory.py | ✅ Complete |
| clinical/integration.py | test_clinical_integration.py | ✅ Complete |
| data_manager.py | test_data_manager.py | ✅ Complete |
| model_runner.py | test_model_runner.py | ✅ Complete |
| cross_validation.py | test_cross_validation.py | ✅ Complete |

### Experiments (tests/experiments/)

| Module | Test File | Status |
|--------|-----------|--------|
| clinical_validation.py | test_clinical_validation.py | ✅ Complete |
| data_validation.py | test_data_validation.py | ✅ Complete |
| model_comparison.py | test_model_comparison.py | ✅ Complete |

### Visualization (tests/visualization/)

| Module | Test File | Status |
|--------|-----------|--------|
| brain_plots.py | test_brain_plots.py | ✅ Complete |
| factor_plots.py | test_factor_plots.py | ✅ Complete |
| manager.py | test_manager.py | ✅ Complete |

## Testing Best Practices

### 1. Test Isolation

- Each test should be independent
- Use mocks to avoid external dependencies
- Clean up resources in teardown

### 2. Test Naming

- Use descriptive names: `test_<what>_<condition>_<expected>`
- Example: `test_error_handling_with_invalid_data_returns_error_result`

### 3. Arrange-Act-Assert

```python
def test_component_creation():
    # Arrange: Set up test data
    config = {"K": 10}

    # Act: Execute the code
    component = create_component(config)

    # Assert: Verify results
    assert component is not None
    assert component.K == 10
```

### 4. Test Edge Cases

- Empty data
- None inputs
- Invalid configurations
- Boundary conditions
- Error conditions

### 5. Integration Tests

- Test complete workflows
- Verify component interactions
- Test real-world scenarios

## Test Markers

Tests can be marked with pytest markers:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_integration():
    """Slower integration test."""
    pass

@pytest.mark.slow
def test_expensive_computation():
    """Very slow test."""
    pass
```

Run specific markers:

```bash
pytest -m unit  # Run only unit tests
pytest -m "not slow"  # Skip slow tests
```

## Next Steps

### Recommended Test Additions

1. **Core Module Tests**:
   - test_parameter_resolver.py
   - test_model_interface.py
   - test_config_utils.py
   - test_logger_utils.py

2. **Models Tests**:
   - test_sparse_gfa.py
   - test_standard_gfa.py
   - test_factory.py

3. **Data Tests**:
   - test_qmap_pd.py
   - test_synthetic.py
   - test_preprocessing.py

4. **Optimization Tests**:
   - test_memory_optimizer.py
   - test_mcmc_optimizer.py

### Testing Infrastructure Improvements

1. **Continuous Integration**:
   - Set up GitHub Actions
   - Run tests on every commit
   - Generate coverage reports

2. **Performance Testing**:
   - Add benchmarks for critical functions
   - Track performance over time

3. **Property-Based Testing**:
   - Use hypothesis for property-based tests
   - Test invariants across random inputs

4. **Mutation Testing**:
   - Use mutpy to verify test quality
   - Ensure tests catch real bugs

## Troubleshooting

### Common Issues

**Issue**: ImportError when running tests
```bash
# Solution: Install package in development mode
pip install -e .
```

**Issue**: Tests pass locally but fail in CI
```bash
# Solution: Check for environment differences
pytest --verbose  # See detailed output
```

**Issue**: Mocks not working as expected
```python
# Solution: Check patch path
# Wrong: @patch('my_module.function')
# Right: @patch('module_under_test.function')
```

**Issue**: Fixtures not found
```python
# Solution: Ensure conftest.py is in correct location
# conftest.py should be in tests/ directory
```

## Conclusion

The SGFA test suite provides comprehensive coverage of core functionality, with particular focus on:
- Error handling and recovery
- Component creation and integration
- Clinical validation workflows
- Data loading and preparation
- Model execution

All newly refactored modules have complete test coverage, ensuring maintainability and reliability as the codebase evolves.
