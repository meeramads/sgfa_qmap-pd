"""Example usage of data loading and preprocessing pipeline."""

import logging
from pathlib import Path

import numpy as np

from core.config_utils import get_data_dir
from core.io_utils import save_csv, save_json, save_numpy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_synthetic_data_generation():
    """Synthetic data generation example."""
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION EXAMPLE")
    print("=" * 60)

    from data import generate_synthetic_data

    # Basic synthetic data generation
    print("🔄 Generating basic synthetic data...")
    basic_data = generate_synthetic_data(num_sources=3, K=5, num_subjects=100)

    print(f"✅ Generated data with {len(basic_data['X_list'])} views:")
    for i, X in enumerate(basic_data["X_list"]):
        print(f"  View {i}: {X.shape} (subjects × features)")

    print(f"True factor matrix Z: {basic_data['Z_true'].shape}")
    print(f"True loading matrices W: {len(basic_data['W_true'])} views")

    # Advanced synthetic data with custom parameters
    print(f"\n🔄 Generating advanced synthetic data...")
    advanced_data = generate_synthetic_data(
        num_sources=4,
        K=8,
        num_subjects=200,
        sparsity_level=0.7,  # 70% sparse
        noise_level=0.1,  # Low noise
        seed=42,
    )

    print(f"✅ Generated advanced data:")
    for i, X in enumerate(advanced_data["X_list"]):
        print(f"  View {i}: {X.shape}")

    # Analyze synthetic data properties
    print(f"\n📊 Analyzing synthetic data properties...")

    # Noise levels
    noise_levels = []
    for i, (X, W_true) in enumerate(
        zip(advanced_data["X_list"], advanced_data["W_true"])
    ):
        # Reconstruct without noise
        X_clean = np.dot(advanced_data["Z_true"], W_true.T)
        noise_var = np.var(X - X_clean)
        signal_var = np.var(X_clean)
        snr = signal_var / noise_var if noise_var > 0 else np.inf

        noise_levels.append(
            {
                "view": i,
                "noise_variance": noise_var,
                "signal_variance": signal_var,
                "snr": snr,
            }
        )
        print(f"  View {i}: SNR = {snr:.2f}")

    # Sparsity analysis
    print(f"\nSparsity analysis:")
    for i, W in enumerate(advanced_data["W_true"]):
        sparsity = np.mean(np.abs(W) < 1e-6)
        print(f"  View {i}: {sparsity * 100:.1f}% sparse")

    return basic_data, advanced_data, noise_levels


def example_qmap_pd_data_loading():
    """qMAP-PD dataset loading example."""
    print("\n" + "=" * 60)
    print("qMAP-PD DATA LOADING EXAMPLE")
    print("=" * 60)

    from data import load_qmap_pd

    # Check if data directory exists
    data_dir = "./qMAP-PD_data"
    if not Path(data_dir).exists():
        print(f"❌ Data directory {data_dir} not found")
        print("   Please ensure qMAP-PD data is available")

        # Show what we would expect to load
        print(f"\n📋 Expected qMAP-PD data structure:")
        print(f"  • Structural MRI data (T1-weighted)")
        print(f"  • Functional connectivity matrices")
        print(f"  • DTI-derived metrics")
        print(f"  • Clinical/demographic variables")
        print(f"  • Multiple subjects and timepoints")

        return None

    try:
        print(f"🔄 Loading qMAP-PD data from {data_dir}...")
        qmap_data = load_qmap_pd(data_dir)

        print(f"✅ Successfully loaded qMAP-PD data!")
        print(f"Data structure: {type(qmap_data)}")

        # Analyze loaded data
        if isinstance(qmap_data, dict):
            print(f"\nData components:")
            for key, value in qmap_data.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                    if value and hasattr(value[0], "shape"):
                        print(f"    First item shape: {value[0].shape}")
                else:
                    print(f"  {key}: {type(value)}")

        return qmap_data

    except Exception as e:
        print(f"❌ Failed to load qMAP-PD data: {e}")
        print(f"   This is normal if the data is not available")
        return None


def example_basic_preprocessing():
    """Basic data preprocessing example."""
    print("\n" + "=" * 60)
    print("BASIC PREPROCESSING EXAMPLE")
    print("=" * 60)

    # Generate test data
    from data import generate_synthetic_data
    from data.preprocessing import DataPreprocessor

    data = generate_synthetic_data(num_sources=2, K=4, num_subjects=80)
    X_list = data["X_list"]

    print(f"🔄 Preprocessing {len(X_list)} data views...")
    for i, X in enumerate(X_list):
        print(f"  Original view {i}: {X.shape}")

    # Create preprocessor with basic settings
    preprocessor = DataPreprocessor(
        standardize=True, handle_missing=True, remove_constant_features=True
    )

    # Preprocess each view
    X_processed = []
    preprocessing_info = {}

    for i, X in enumerate(X_list):
        print(f"\n🔧 Processing view {i}...")

        # Add some missing values for demonstration
        X_with_missing = X.copy()
        n_missing = int(0.05 * X.size)  # 5% missing
        missing_idx = np.random.choice(X.size, n_missing, replace=False)
        flat_X = X_with_missing.flatten()
        flat_X[missing_idx] = np.nan
        X_with_missing = flat_X.reshape(X.shape)

        print(f"  Added {n_missing} missing values ({100 * n_missing / X.size:.1f}%)")

        # Preprocess
        X_proc, info = preprocessor.fit_transform(X_with_missing)
        X_processed.append(X_proc)
        preprocessing_info[f"view_{i}"] = info

        print(f"  ✅ Processed: {X_proc.shape}")
        print(f"     Missing values imputed: {info.get('missing_imputed', 0)}")
        print(f"     Features removed: {info.get('features_removed', 0)}")
        print(f"     Standardization: {info.get('standardized', False)}")

    # Compare before/after statistics
    print(f"\n📊 Preprocessing comparison:")
    print(f"{'View':<6} {'Original':<15} {'Processed':<15} {'Change':<10}")
    print("-" * 50)

    for i, (X_orig, X_proc) in enumerate(zip(X_list, X_processed)):
        orig_std = np.std(X_orig)
        proc_std = np.std(X_proc)
        print(
            f"{i:<6} {orig_std:<15.3f} {proc_std:<15.3f} {proc_std / orig_std:<10.2f}"
        )

    return X_processed, preprocessing_info


def example_advanced_preprocessing():
    """Advanced preprocessing with strategy selection."""
    print("\n" + "=" * 60)
    print("ADVANCED PREPROCESSING EXAMPLE")
    print("=" * 60)

    from data.preprocessing_integration import apply_preprocessing_to_pipeline

    # Load configuration
    try:
        import yaml

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except BaseException:
        # Fallback configuration
        config = {
            "data": {"data_dir": "./qMAP-PD_data"},
            "data_validation": {
                "preprocessing_strategies": {
                    "minimal": {
                        "enable_advanced_preprocessing": False,
                        "imputation_strategy": "mean",
                    },
                    "standard": {
                        "enable_advanced_preprocessing": True,
                        "imputation_strategy": "median",
                        "feature_selection_method": "variance",
                        "variance_threshold": 0.01,
                    },
                    "aggressive": {
                        "enable_advanced_preprocessing": True,
                        "enable_spatial_processing": True,
                        "imputation_strategy": "knn",
                        "feature_selection_method": "mutual_info",
                        "n_top_features": 1000,
                        "spatial_imputation": True,
                        "roi_based_selection": True,
                    },
                }
            },
        }

    # Test different preprocessing strategies
    strategies = ["minimal", "standard", "aggressive"]
    strategy_results = {}

    # Generate test data with more complexity
    from data import generate_synthetic_data

    base_data = generate_synthetic_data(
        num_sources=3, K=6, num_subjects=100, noise_level=0.2
    )

    for strategy in strategies:
        print(f"\n🔄 Testing '{strategy}' preprocessing strategy...")

        try:
            # Apply preprocessing strategy
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config,
                data_dir=get_data_dir(config),
                auto_select_strategy=False,
                preferred_strategy=strategy,
                X_list_override=base_data["X_list"],  # Use synthetic data
            )

            strategy_results[strategy] = {
                "X_list": X_list,
                "info": preprocessing_info,
                "success": True,
            }

            print(f"✅ '{strategy}' strategy completed")
            print(f"   Views processed: {len(X_list)}")
            for i, X in enumerate(X_list):
                print(f"   View {i}: {X.shape}")

            # Analyze preprocessing effects
            if "strategy_selection" in preprocessing_info:
                strategy_info = preprocessing_info["strategy_selection"]
                print(
                    f"   Selected strategy: {
                        strategy_info.get(
                            'selected_strategy',
                            'unknown')}"
                )
                print(f"   Quality score: {strategy_info.get('quality_score', 'N/A')}")

        except Exception as e:
            print(f"❌ '{strategy}' strategy failed: {e}")
            strategy_results[strategy] = {"success": False, "error": str(e)}

    # Compare strategies
    print(f"\n📊 PREPROCESSING STRATEGY COMPARISON")
    print("-" * 60)
    print(f"{'Strategy':<12} {'Success':<8} {'Views':<6} {'Features (View 0)':<16}")
    print("-" * 60)

    for strategy, result in strategy_results.items():
        if result["success"]:
            n_features = result["X_list"][0].shape[1] if result["X_list"] else 0
            print(
                f"{strategy:<12} {'✅':<8} {len(result['X_list']):<6} {n_features:<16}"
            )
        else:
            print(f"{strategy:<12} {'❌':<8} {'N/A':<6} {'N/A':<16}")

    return strategy_results


def example_data_quality_assessment():
    """Data quality assessment and validation example."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT EXAMPLE")
    print("=" * 60)

    import numpy as np

    from data import generate_synthetic_data

    # Generate data with various quality issues
    print("🔄 Generating data with quality issues...")

    # Base data
    data = generate_synthetic_data(num_sources=3, K=4, num_subjects=120)
    X_list = data["X_list"].copy()

    # Introduce quality issues
    quality_issues = {}

    # 1. Missing values
    for i, X in enumerate(X_list):
        n_missing = int(0.03 * X.size)  # 3% missing
        missing_idx = np.random.choice(X.size, n_missing, replace=False)
        flat_X = X.flatten()
        flat_X[missing_idx] = np.nan
        X_list[i] = flat_X.reshape(X.shape)
        quality_issues[f"view_{i}_missing"] = n_missing

    # 2. Outliers
    for i, X in enumerate(X_list):
        n_outliers = int(0.01 * X.size)  # 1% outliers
        outlier_idx = np.random.choice(X.size, n_outliers, replace=False)
        flat_X = X_list[i].flatten()
        flat_X[outlier_idx] = np.random.normal(0, 10, n_outliers)  # Extreme values
        X_list[i] = flat_X.reshape(X.shape)
        quality_issues[f"view_{i}_outliers"] = n_outliers

    # 3. Constant features
    for i, X in enumerate(X_list):
        n_constant = 5  # Make 5 features constant
        constant_features = np.random.choice(X.shape[1], n_constant, replace=False)
        for feat in constant_features:
            X_list[i][:, feat] = np.random.randn()  # Same value for all subjects
        quality_issues[f"view_{i}_constant"] = n_constant

    print(f"✅ Introduced quality issues:")
    for issue, count in quality_issues.items():
        print(f"  {issue}: {count}")

    # Assess data quality
    print(f"\n🔍 Assessing data quality...")

    quality_metrics = {}
    for i, X in enumerate(X_list):
        view_metrics = {}

        # Missing value analysis
        missing_mask = np.isnan(X)
        view_metrics["missing_percentage"] = 100 * np.mean(missing_mask)
        view_metrics["missing_per_subject"] = np.sum(missing_mask, axis=1)
        view_metrics["missing_per_feature"] = np.sum(missing_mask, axis=0)

        # Outlier detection (using IQR method)
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (X < lower_bound) | (X > upper_bound)
        view_metrics["outlier_percentage"] = 100 * np.nanmean(outlier_mask)

        # Constant feature detection
        feature_variance = np.nanvar(X, axis=0)
        constant_features = feature_variance < 1e-10
        view_metrics["constant_features"] = np.sum(constant_features)

        # Distribution properties
        view_metrics["mean_variance"] = np.nanmean(feature_variance)
        view_metrics["mean_skewness"] = float(
            np.nanmean(
                [
                    abs(
                        np.nanmean((X[:, j] - np.nanmean(X[:, j])) ** 3)
                        / np.nanvar(X[:, j]) ** 1.5
                    )
                    for j in range(X.shape[1])
                    if np.nanvar(X[:, j]) > 0
                ]
            )
        )

        quality_metrics[f"view_{i}"] = view_metrics

    # Print quality report
    print(f"\n📋 DATA QUALITY REPORT")
    print("-" * 80)
    print(
        f"{
            'View':<6} {
            'Missing %':<10} {
                'Outliers %':<12} {
                    'Constant':<10} {
                        'Mean Var':<10} {
                            'Skewness':<10}"
    )
    print("-" * 80)

    for i in range(len(X_list)):
        metrics = quality_metrics[f"view_{i}"]
        print(
            f"{
                i:<6} {
                metrics['missing_percentage']:<10.2f} "
            f"{
                metrics['outlier_percentage']:<12.2f} {
                    metrics['constant_features']:<10} "
            f"{
                        metrics['mean_variance']:<10.3f} {
                            metrics['mean_skewness']:<10.3f}"
        )

    # Quality recommendations
    print(f"\n💡 QUALITY RECOMMENDATIONS")
    print("-" * 40)

    for i in range(len(X_list)):
        metrics = quality_metrics[f"view_{i}"]
        print(f"\nView {i}:")

        if metrics["missing_percentage"] > 5:
            print(
                f"  ⚠️  High missing values ({
                    metrics['missing_percentage']:.1f}%) - consider advanced imputation"
            )

        if metrics["outlier_percentage"] > 2:
            print(
                f"  ⚠️  High outlier rate ({
                    metrics['outlier_percentage']:.1f}%) - consider robust methods"
            )

        if metrics["constant_features"] > 0:
            print(
                f"  ⚠️  {
                    metrics['constant_features']} constant features - should be removed"
            )

        if metrics["mean_skewness"] > 2:
            print(
                f"  ⚠️  High skewness ({
                    metrics['mean_skewness']:.2f}) - consider transformation"
            )

        if metrics["mean_variance"] < 0.1:
            print(f"  ⚠️  Low variance features - may need feature selection")

    return X_list, quality_metrics


def example_data_splitting_and_cv():
    """Data splitting and cross-validation setup example."""
    print("\n" + "=" * 60)
    print("DATA SPLITTING AND CROSS-VALIDATION EXAMPLE")
    print("=" * 60)

    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    from data import generate_synthetic_data

    # Generate data with group structure
    print("🔄 Generating data with group/subject structure...")
    data = generate_synthetic_data(num_sources=2, K=3, num_subjects=100)
    X_list = data["X_list"]

    # Simulate subject groupings (e.g., different clinical sites)
    n_subjects = X_list[0].shape[0]
    n_groups = 4
    subject_groups = np.random.randint(0, n_groups, n_subjects)

    # Simulate clinical labels for stratification
    clinical_labels = np.random.choice(
        ["Control", "PD_Mild", "PD_Severe"], size=n_subjects, p=[0.4, 0.35, 0.25]
    )

    print(f"✅ Data structure:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Groups: {n_groups}")
    print(f"  Clinical labels: {np.unique(clinical_labels)}")
    print(
        f"  Label distribution: {dict(zip(*np.unique(clinical_labels, return_counts=True)))}"
    )

    # Example 1: Simple train/validation/test split
    print(f"\n📊 Example 1: Simple train/validation/test split")

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        range(n_subjects), test_size=0.2, stratify=clinical_labels, random_state=42
    )

    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of total
        stratify=clinical_labels[train_val_idx],
        random_state=42,
    )

    print(
        f"  Train: {
            len(train_idx)} subjects ({
            len(train_idx) /
            n_subjects *
            100:.1f}%)"
    )
    print(
        f"  Validation: {
            len(val_idx)} subjects ({
            len(val_idx) /
            n_subjects *
            100:.1f}%)"
    )
    print(f"  Test: {len(test_idx)} subjects ({len(test_idx) / n_subjects * 100:.1f}%)")

    # Example 2: K-fold cross-validation
    print(f"\n📊 Example 2: K-fold cross-validation")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(n_subjects))):
        cv_splits.append((train_idx, val_idx))
        print(f"  Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")

    # Example 3: Stratified K-fold for imbalanced data
    print(f"\n📊 Example 3: Stratified K-fold cross-validation")

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_splits = []

    for fold, (train_idx, val_idx) in enumerate(
        stratified_kfold.split(range(n_subjects), clinical_labels)
    ):
        stratified_splits.append((train_idx, val_idx))

        # Check stratification
        train_labels = clinical_labels[train_idx]
        val_labels = clinical_labels[val_idx]

        train_dist = dict(zip(*np.unique(train_labels, return_counts=True)))
        val_dist = dict(zip(*np.unique(val_labels, return_counts=True)))

        print(f"  Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        print(f"    Train distribution: {train_dist}")
        print(f"    Val distribution: {val_dist}")

    # Example 4: Group-aware splitting (important for multi-site data)
    print(f"\n📊 Example 4: Group-aware splitting")

    from sklearn.model_selection import GroupKFold

    group_kfold = GroupKFold(n_splits=min(4, n_groups))
    group_splits = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(range(n_subjects), groups=subject_groups)
    ):
        group_splits.append((train_idx, val_idx))

        train_groups = set(subject_groups[train_idx])
        val_groups = set(subject_groups[val_idx])

        print(f"  Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        print(f"    Train groups: {sorted(train_groups)}")
        print(f"    Val groups: {sorted(val_groups)}")
        print(f"    No overlap: {len(train_groups & val_groups) == 0}")

    # Utility function for data splitting
    def split_multiview_data(X_list, indices):
        """Split multi-view data using provided indices."""
        return [X[indices] for X in X_list]

    # Demonstrate data splitting
    print(f"\n🔧 Demonstrating data extraction for fold 1:")
    train_idx, val_idx = cv_splits[0]

    X_train = split_multiview_data(X_list, train_idx)
    X_val = split_multiview_data(X_list, val_idx)

    print(f"  Training data:")
    for i, X in enumerate(X_train):
        print(f"    View {i}: {X.shape}")

    print(f"  Validation data:")
    for i, X in enumerate(X_val):
        print(f"    View {i}: {X.shape}")

    return {
        "simple_split": (train_idx, val_idx, test_idx),
        "cv_splits": cv_splits,
        "stratified_splits": stratified_splits,
        "group_splits": group_splits,
        "data": X_list,
        "labels": clinical_labels,
        "groups": subject_groups,
    }


def example_data_export_and_formats():
    """Data export and format conversion example."""
    print("\n" + "=" * 60)
    print("DATA EXPORT AND FORMAT CONVERSION EXAMPLE")
    print("=" * 60)

    from pathlib import Path

    import numpy as np
    import pandas as pd

    from data import generate_synthetic_data

    # Generate example data
    data = generate_synthetic_data(num_sources=2, K=3, num_subjects=50)
    X_list = data["X_list"]
    Z_true = data["Z_true"]
    W_true = data["W_true"]

    # Create export directory
    export_dir = Path("results/data_export_example")
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Exporting data to: {export_dir}")

    # 1. NumPy format (.npy)
    print(f"\n💾 Exporting in NumPy format...")
    for i, X in enumerate(X_list):
        save_numpy(X, export_dir / f"data_view_{i}.npy")
        print(f"  ✅ Saved view {i}: {X.shape}")

    save_numpy(Z_true, export_dir / "true_factors.npy")
    for i, W in enumerate(W_true):
        save_numpy(W, export_dir / f"true_loadings_view_{i}.npy")

    # 2. CSV format (for smaller datasets or inspection)
    print(f"\n📊 Exporting in CSV format...")
    for i, X in enumerate(X_list):
        # Create meaningful column names
        feature_names = [f"Feature_{j + 1}" for j in range(X.shape[1])]
        subject_names = [f"Subject_{j + 1}" for j in range(X.shape[0])]

        df = pd.DataFrame(X, columns=feature_names, index=subject_names)
        save_csv(df, export_dir / f"data_view_{i}.csv")
        print(f"  ✅ Saved view {i} CSV: {df.shape}")

    # Factor scores as CSV
    factor_names = [f"Factor_{j + 1}" for j in range(Z_true.shape[1])]
    subject_names = [f"Subject_{j + 1}" for j in range(Z_true.shape[0])]

    factor_df = pd.DataFrame(Z_true, columns=factor_names, index=subject_names)
    save_csv(factor_df, export_dir / "true_factors.csv")
    print(f"  ✅ Saved factors CSV: {factor_df.shape}")

    # 3. HDF5 format (efficient for large datasets)
    print(f"\n🗜️  Exporting in HDF5 format...")
    try:
        import h5py

        with h5py.File(export_dir / "complete_dataset.h5", "w") as f:
            # Create groups
            data_group = f.create_group("data")
            true_group = f.create_group("ground_truth")
            meta_group = f.create_group("metadata")

            # Store data views
            for i, X in enumerate(X_list):
                data_group.create_dataset(f"view_{i}", data=X, compression="gzip")

            # Store ground truth
            true_group.create_dataset("factors", data=Z_true, compression="gzip")
            for i, W in enumerate(W_true):
                true_group.create_dataset(
                    f"loadings_view_{i}", data=W, compression="gzip"
                )

            # Store metadata
            meta_group.attrs["num_views"] = len(X_list)
            meta_group.attrs["num_subjects"] = X_list[0].shape[0]
            meta_group.attrs["num_factors"] = Z_true.shape[1]
            meta_group.attrs["view_dimensions"] = [X.shape[1] for X in X_list]

        print(f"  ✅ Saved HDF5 format: complete_dataset.h5")

    except ImportError:
        print(f"  ⚠️  h5py not available, skipping HDF5 export")

    # 4. MATLAB format (.mat)
    print(f"\n🔬 Exporting in MATLAB format...")
    try:
        from scipy.io import savemat

        matlab_data = {
            "data_views": X_list,
            "true_factors": Z_true,
            "true_loadings": W_true,
            "metadata": {
                "num_views": len(X_list),
                "num_subjects": X_list[0].shape[0],
                "num_factors": Z_true.shape[1],
            },
        }

        savemat(export_dir / "dataset.mat", matlab_data)
        print(f"  ✅ Saved MATLAB format: dataset.mat")

    except ImportError:
        print(f"  ⚠️  scipy not available, skipping MATLAB export")

    # 5. Create data summary and documentation
    print(f"\n📋 Creating data documentation...")

    data_summary = {
        "dataset_info": {
            "type": "synthetic_multiview",
            "num_subjects": int(X_list[0].shape[0]),
            "num_views": len(X_list),
            "num_factors": int(Z_true.shape[1]),
            "generation_date": pd.Timestamp.now().isoformat(),
        },
        "view_details": [
            {
                "view_id": i,
                "shape": list(X.shape),
                "mean": float(np.mean(X)),
                "std": float(np.std(X)),
                "min": float(np.min(X)),
                "max": float(np.max(X)),
            }
            for i, X in enumerate(X_list)
        ],
        "factor_details": {
            "shape": list(Z_true.shape),
            "mean": float(np.mean(Z_true)),
            "std": float(np.std(Z_true)),
        },
        "files_exported": [
            "NumPy files: data_view_*.npy, true_factors.npy, true_loadings_view_*.npy",
            "CSV files: data_view_*.csv, true_factors.csv",
            "HDF5 file: complete_dataset.h5 (if available)",
            "MATLAB file: dataset.mat (if available)",
        ],
    }

    save_json(data_summary, export_dir / "data_summary.json")

    # Create README
    readme_content = f"""# Exported Dataset

## Overview
This directory contains a synthetic multi-view dataset generated for SGFA analysis.

## Dataset Details
- **Subjects**: {X_list[0].shape[0]}
- **Views**: {len(X_list)}
- **Factors**: {Z_true.shape[1]}
- **Generation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## File Formats

### NumPy Files (.npy)
- `data_view_*.npy`: Raw data matrices for each view
- `true_factors.npy`: True factor scores (Z matrix)
- `true_loadings_view_*.npy`: True factor loadings (W matrices)

### CSV Files (.csv)
- `data_view_*.csv`: Data matrices in CSV format with headers
- `true_factors.csv`: Factor scores in CSV format

### HDF5 File (.h5)
- `complete_dataset.h5`: Complete dataset in hierarchical format

### MATLAB File (.mat)
- `dataset.mat`: All data in MATLAB-compatible format

### Documentation
- `data_summary.json`: Detailed metadata and statistics
- `README.md`: This documentation file

## Usage
```python
# Load NumPy format
import numpy as np
X_view_0 = np.load('data_view_0.npy')
Z_true = np.load('true_factors.npy')

# Load CSV format
import pandas as pd
df = pd.read_csv('data_view_0.csv', index_col=0)

# Load HDF5 format
import h5py
with h5py.File('complete_dataset.h5', 'r') as f:
    X = f['data/view_0'][:]
```
"""

    with open(export_dir / "README.md", "w") as f:
        f.write(readme_content)

    print(f"  ✅ Created documentation: data_summary.json, README.md")

    # List all exported files
    print(f"\n📂 Files exported to {export_dir}:")
    for file_path in sorted(export_dir.iterdir()):
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  {file_path.name} ({file_size:.1f} KB)")

    return export_dir


if __name__ == "__main__":
    print("Data Loading and Preprocessing Examples")
    print("=" * 60)

    # Run all examples
    try:
        # Basic examples
        basic_data, advanced_data, noise_info = example_synthetic_data_generation()
        qmap_data = example_qmap_pd_data_loading()

        # Preprocessing examples
        processed_data, proc_info = example_basic_preprocessing()
        strategy_results = example_advanced_preprocessing()

        # Quality and validation examples
        quality_data, quality_metrics = example_data_quality_assessment()
        cv_data = example_data_splitting_and_cv()
        export_dir = example_data_export_and_formats()

        print("\n" + "=" * 60)
        print("ALL DATA EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nKey takeaways:")
        print(f"• Generated synthetic data with various complexity levels")
        print(f"• Demonstrated {len(strategy_results)} preprocessing strategies")
        print(f"• Assessed data quality across multiple dimensions")
        print(f"• Set up {len(cv_data['cv_splits'])} cross-validation folds")
        print(f"• Exported data in multiple formats to: {export_dir}")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
