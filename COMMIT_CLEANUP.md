# Cleanup Commit Guide

## What Happened

The cleanup script:
1. ✅ **Moved** obsolete config files to `archived_configs/` (not tracked by git)
2. ✅ **Deleted** test artifacts from root
3. ✅ **Updated** `.gitignore` to prevent future accumulation
4. ✅ **Created** cleanup tools for future use

## Files to Commit

### Deleted Files (Good - settings now available via flags)
```
deleted:    config_K10_MAD5.0.yaml
deleted:    config_K10_full_voxels.yaml
deleted:    config_K15_full_voxels.yaml
deleted:    config_K20_MAD3.0.yaml
deleted:    config_K20_no_sparsity.yaml
deleted:    config_K2_no_sparsity.yaml
deleted:    config_K50_full_voxels.yaml
deleted:    config_K50_no_sparsity.yaml
deleted:    config_debug.yaml
deleted:    debug_experiments.py
deleted:    test_config_quick.py
deleted:    test_synthetic_config.py
deleted:    test_trace_plots.py
deleted:    test_variance_profile.py
```

### Modified Files
```
modified:   .gitignore (added ignore rules for test artifacts and archives)
```

### New Files (Optional to commit)
```
CLEANUP_SUMMARY.md - Documentation of what was cleaned
cleanup_project.sh - Reusable cleanup script
```

## Recommended Commit Commands

### Option 1: Commit everything (recommended)
```bash
git add -A
git commit -m "chore: remove obsolete config files and test artifacts

- Remove config files now replaceable with CLI flags
  (K variations, MAD thresholds, sparsity settings, etc.)
- Remove test artifacts from root directory
- Update .gitignore to prevent accumulation of:
  - Test outputs and debug files
  - Archive directories
  - Python cache directories
- Add cleanup script and documentation for future maintenance

All removed configs are archived in archived_configs/ (not tracked)
and can be restored if needed."
```

### Option 2: Just commit deletions and .gitignore
```bash
git add .gitignore
git add -u  # Stage all deletions
git commit -m "chore: remove obsolete config files and update gitignore

Config settings now available via CLI flags.
Archived locally in archived_configs/ for reference."
```

### Option 3: Review before commit
```bash
git status
git diff .gitignore
git commit -m "your message"
```

## After Committing

The archived files in `archived_configs/` are:
- ✅ Safe on your local machine
- ✅ Ignored by Git (won't clutter repo)
- ✅ Can be restored anytime: `mv archived_configs/* .`
- ✅ Can be deleted permanently when confident: `rm -rf archived_configs/`

## Using Flags Instead

Old way:
```bash
python run_experiments.py --config config_K20_no_sparsity.yaml
```

New way:
```bash
python run_experiments.py --config config_K50.yaml --K 20 --no-sparsity
```

Much cleaner!
