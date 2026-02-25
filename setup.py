# setup.py
import os, sys, math, runpy

# --- Detect correct improved_diffusion path automatically ---
ROOT = os.path.abspath(os.path.dirname(__file__))

def find_repo_root():
    """
    Return the path that contains the improved_diffusion package.
    Works for both:
      benchmark/improved-diffusion/improved_diffusion/
      benchmark/improved_diffusion/improved_diffusion/
    """
    candidates = []
    for root, dirs, _ in os.walk(os.path.join(ROOT, "benchmark")):
        if "improved_diffusion" in dirs:
            candidates.append(os.path.join(root))
    if not candidates:
        raise RuntimeError("Could not locate improved_diffusion/ under benchmark/.")
    # Pick the first valid candidate
    return candidates[0]

BENCHMARK_DIR = find_repo_root()
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)
print(f"[setup] Using improved_diffusion from: {BENCHMARK_DIR}")

# --- Patch logistic schedule and optional NUM_CLASSES ---
def _install_logistic_schedule(num_classes=None):
    from improved_diffusion import gaussian_diffusion as gd, script_util

    # logistic alpha_bar(t)
    def _logistic_alpha_bar(t): return 1.0 / (1.0 + math.exp(10.0 * (t - 0.5)))

    # Inject schedule
    _orig = gd.get_named_beta_schedule
    def _patched(name, steps):
        if name and name.lower() == "logistic":
            return gd.betas_for_alpha_bar(steps, _logistic_alpha_bar)
        return _orig(name, steps)
    gd.get_named_beta_schedule = _patched

    # Optional NUM_CLASSES override
    if num_classes is not None:
        script_util.NUM_CLASSES = num_classes
        print(f"[setup] NUM_CLASSES = {num_classes}")

# --- Main runner ---
def _run_target_with_args():
    """
    Run upstream scripts with the logistic schedule and optional NUM_CLASSES override.

    Example:
      python setup.py scripts.image_train --num_classes 7 -- \
        --data_dir built_data --noise_schedule logistic --class_cond True
    """
    if len(sys.argv) <= 1:
        print("Usage: python setup.py <module> [--num_classes N] -- <args>")
        return

    args = sys.argv[1:]
    num_classes = None
    if "--num_classes" in args:
        i = args.index("--num_classes")
        num_classes = int(args[i + 1])
        del args[i:i + 2]

    if "--" in args:
        dash = args.index("--")
        target, forwarded = args[0], args[dash + 1:]
    else:
        target, forwarded = args[0], args[1:]

    _install_logistic_schedule(num_classes)

    if target.endswith(".py"):
        path = os.path.abspath(target)
        sys.argv = [path] + forwarded
        runpy.run_path(path, run_name="__main__")
    else:
        sys.argv = [target] + forwarded
        runpy.run_module(target, run_name="__main__")

if __name__ == "__main__":
    _run_target_with_args()
else:
    _install_logistic_schedule()
