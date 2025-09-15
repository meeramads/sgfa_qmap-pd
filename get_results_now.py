import sys
sys.path.insert(0, '.')

print("Starting direct model run...")

# Import and run the analysis directly
from core.run_analysis import main

# Create args namespace
import argparse
args = argparse.Namespace(
    model='sparseGFA',
    K=5,
    num_samples=500,
    num_warmup=500,
    num_chains=1,
    percW=33,
    device='gpu',
    dataset='qmap_pd',
    num_runs=1,
    cv=False,
    save_results=True,
    seed=42
)

print(f"Running SGFA with K={args.K} factors...")
print(f"Using {args.num_samples} samples with {args.num_warmup} warmup")

try:
    main(args)
    print("\nModel completed successfully!")
except Exception as e:
    print(f"Error: {e}")
    # Try alternative approach
    print("\nTrying alternative approach...")
    import subprocess
    subprocess.run([
        "python", "core/run_analysis.py",
        "--model", "sparseGFA",
        "--K", "5",
        "--num_samples", "500",
        "--percW", "33"
    ])

print("\nCheck results in: ../results/qmap_pd/")
