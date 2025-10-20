# Codebase Cleanup Complete - October 20, 2025

## Documentation Consolidation

### Removed 11 Redundant Files
**Convergence testing (3 → 1):**
- ~~CONVERGENCE_TESTS_QUICKSTART.md~~
- ~~CONVERGENCE_TESTS_README.md~~
- ~~QUICK_START_CONVERGENCE_TESTS.md~~
- **→ CONVERGENCE_TESTING_GUIDE.md** (consolidated)

**Fix summaries (8 → 1):**
- ~~MODEL_TYPE_FIX_SUMMARY.md~~
- ~~RECENT_FIXES_SUMMARY.md~~
- ~~SEMANTIC_NAMING_GUIDE.md~~
- ~~UNIFIED_SEMANTIC_NAMING_UPDATE.md~~
- ~~POSITION_LOOKUP_SAVE_SUMMARY.md~~
- ~~SUBJECT_OUTLIER_PLOTS_SUMMARY.md~~
- ~~BASELINE_MATCH_VERIFICATION.md~~
- ~~CONVERGENCE_FIXES_SUMMARY.md~~
- **→ FIXES_AND_ENHANCEMENTS.md** (consolidated)

### Final Documentation Structure

**Root directory:**
- `README.md` - Main project documentation

**docs/:**
- `CONVERGENCE_TESTING_GUIDE.md` - How to test convergence ⭐ NEW
- `FIXES_AND_ENHANCEMENTS.md` - All fixes and enhancements ⭐ NEW
- `PCA_CONFIGURATION_GUIDE.md` - PCA usage guide
- `config_quick_reference.md` - Config parameter reference
- `configuration.md` - Full configuration guide
- Architecture docs: ARCHITECTURE.md, CV_ARCHITECTURE.md, UTILS_STRUCTURE.md
- Feature docs: BRAIN_REMAPPING_WITH_VOXEL_FILTERING.md, spatial_remapping_*
- Development docs: TESTING.md, ERROR_HANDLING.md, MCMC_TRACE_PLOTS.md
- Planning docs: REFACTOR_PLAN.md, REMAINING_ISSUES.md

**docs/troubleshooting/:**
- `RECENT_SESSION_SUMMARY.md` - Latest troubleshooting session
- `MCMC_PARAMETER_OVERRIDE_BUG_FIX.md` - Parameter override bugs
- `CONFIG_CHANGE_SMOKING_GUN.md` - Config change analysis
- `CONVERGENCE_ISSUE_ANALYSIS.md` - Convergence issue deep dive

**docs/api/:** (Sphinx-generated API documentation)

---

## Code Fixes Applied

### 1. run_experiments.py
**Line 762:** Fixed target_accept_prob to read from mcmc config
```python
target_accept_prob_value = mcmc_config.get("target_accept_prob") or fs_config.get("target_accept_prob", 0.8)
```

### 2. experiments/robustness_testing.py
**Line 2940:** Fixed hardcoded target_accept_prob
```python
"target_accept_prob": config_dict.get("mcmc", {}).get("target_accept_prob", 0.8),
```

---

## Result

**Before:** 26 markdown files (many redundant)
**After:** 17 markdown files (all unique, well-organized)

**Reduction:** 35% fewer files, 100% better organization

See `docs/FIXES_AND_ENHANCEMENTS.md` for complete fix summary.
See `docs/CONVERGENCE_TESTING_GUIDE.md` for testing guide.
See `docs/troubleshooting/RECENT_SESSION_SUMMARY.md` for session details.
