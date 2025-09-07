# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for automated testing, validation, and deployment of the SGFA qMAP-PD research project.

## 🚀 Workflows Overview

### 1. **CI/CD Pipeline** (`ci.yml`)
**Triggers:** Push to main/master/develop, Pull requests, Weekly schedule
- **Multi-platform testing** (Ubuntu, macOS)
- **Multi-Python version** support (3.8-3.11)
- **Code quality checks** (flake8, black, mypy)
- **Comprehensive testing** (unit, integration, data tests)
- **Coverage reporting** with Codecov integration
- **Performance benchmarking**
- **Reproducibility testing**
- **Security scanning**

### 2. **Research Validation** (`research-validation.yml`)
**Triggers:** Push to main/master, Pull requests affecting research code
- **Experimental framework validation**
- **Model implementation testing**
- **Clinical validation experiments**
- **Performance regression detection**
- **Method comparison validation**

### 3. **Release Pipeline** (`release.yml`)
**Triggers:** Version tags (v*), Published releases
- **Comprehensive test suite execution**
- **Performance benchmark generation**
- **Experimental validation reports**
- **Python package building**
- **PyPI deployment** (for tagged releases)
- **Research artifact packaging**
- **GitHub release creation**

### 4. **Dependency Management** (`dependencies.yml`)
**Triggers:** Weekly schedule, Manual dispatch, Dependency file changes
- **Security vulnerability scanning**
- **License compliance checking**
- **Automated dependency updates**
- **Dependency audit reports**

## 📋 Workflow Details

### CI/CD Pipeline Jobs

#### `test` (Matrix Strategy)
- **OS:** Ubuntu Latest, macOS Latest
- **Python:** 3.8, 3.9, 3.10, 3.11
- **Steps:**
  1. Environment setup with dependency caching
  2. System dependencies installation
  3. Code quality checks (linting, formatting, type checking)
  4. Test execution (unit → integration → data validation)
  5. Coverage reporting

#### `performance-tests`
- **Purpose:** Validate performance characteristics
- **Includes:** Memory usage, execution time, scalability
- **Artifacts:** Performance benchmark results

#### `reproducibility-tests`
- **Purpose:** Ensure research reproducibility
- **Tests:** Seed consistency, result stability
- **Critical for:** Scientific validity

#### `documentation`
- **Purpose:** Validate and generate documentation
- **Includes:** README validation, API docs generation
- **Triggers:** Main branch pushes only

#### `security`
- **Purpose:** Security vulnerability detection
- **Tools:** Bandit (code analysis), Safety (dependency scan)
- **Artifacts:** Security scan reports

### Research Validation Jobs

#### `experimental-validation`
- **Purpose:** Validate experimental framework
- **Tests:**
  - Data validation experiments
  - Method comparison experiments
  - Framework integrity checks
- **Timeout:** 45 minutes
- **Artifacts:** Experiment results

#### `model-validation`
- **Purpose:** Validate model implementations
- **Tests:** Model imports, factory patterns, instantiation
- **Focus:** Core SGFA implementations

#### `performance-regression`
- **Purpose:** Detect performance regressions in PRs
- **Method:** Compare execution times against thresholds
- **Threshold:** 30 seconds for standard operations

#### `clinical-validation`
- **Purpose:** Validate clinical research capabilities
- **Triggers:** `[clinical]` in commit message or scheduled runs
- **Tests:** PD subtype classification, biomarker validation

### Release Pipeline Features

#### Comprehensive Validation
- **Full test suite** execution before release
- **Performance benchmarking** with artifact generation
- **Experimental validation** across all modules

#### Artifact Creation
- **Python packages** (wheel and source distributions)
- **Research data packages** (experimental results, benchmarks)
- **Documentation packages** (API docs, changelogs)

#### Multi-target Deployment
- **PyPI** for production releases (non-alpha/beta)
- **GitHub Releases** with comprehensive artifacts
- **Research artifact preservation**

### Dependency Management Features

#### Security Monitoring
- **Safety** for known vulnerability detection
- **Pip-audit** for comprehensive security scanning
- **Regular scheduled scans** (weekly)

#### License Compliance
- **Automated license checking** for all dependencies
- **Research-friendly license validation**
- **Compliance reporting**

#### Automated Updates
- **Weekly dependency updates** via pull requests
- **Smoke testing** of updated dependencies
- **Automated PR creation** with detailed change summaries

## 🔧 Configuration Requirements

### Repository Secrets
```
PYPI_API_TOKEN          # For PyPI deployment
CODECOV_TOKEN           # For coverage reporting (optional)
```

### Branch Protection Rules
Recommended settings for `main`/`master`:
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Required status checks:
  - `test (ubuntu-latest, 3.10)`
  - `test (ubuntu-latest, 3.11)` 
  - `experimental-validation`
  - `model-validation`

### Environment Setup
The workflows assume:
- **Python 3.8+** compatibility
- **Standard scientific stack** (numpy, scipy, matplotlib)
- **JAX/NumPyro** for probabilistic modeling
- **HDF5 support** for large data handling

## 📊 Monitoring and Reporting

### Test Results
- **Status badges** available for README
- **Coverage reports** via Codecov
- **Performance trends** tracked across releases

### Research Validation
- **Experimental results** preserved as artifacts
- **Performance benchmarks** for each release
- **Reproducibility reports** for scientific validation

### Security and Compliance
- **Weekly vulnerability scans**
- **License compliance reports**
- **Dependency audit trails**

## 🚨 Troubleshooting

### Common Issues

#### Test Failures
1. **Import errors:** Check dependency installation
2. **Memory issues:** Reduce test dataset sizes
3. **Timeout errors:** Increase workflow timeouts

#### Performance Tests
1. **Regression detection:** Check recent changes for efficiency
2. **Memory usage:** Monitor peak memory consumption
3. **Benchmark failures:** Validate test data generation

#### Security Scans
1. **False positives:** Review and whitelist if necessary
2. **Dependency vulnerabilities:** Update affected packages
3. **License issues:** Review compliance requirements

### Debugging Workflows
- **Workflow dispatch** enabled for manual testing
- **Artifact preservation** for failed runs
- **Detailed logging** in all steps

## 🔄 Maintenance

### Regular Tasks
- **Weekly:** Review dependency update PRs
- **Monthly:** Analyze performance trends
- **Per release:** Validate experimental results
- **Quarterly:** Review security scan results

### Workflow Updates
- **Test new Python versions** as they're released
- **Update GitHub Actions** to latest versions
- **Expand test coverage** as project grows
- **Optimize CI/CD performance** regularly

## 📚 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [NumPyro Documentation](https://num.pyro.ai/)
- [Scientific Python Ecosystem](https://scientific-python.org/)