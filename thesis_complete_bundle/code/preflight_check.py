"""Comprehensive pre-flight check for the DiffMIC-v2 + Quantum pipeline.
Run inside the Docker container before starting the real pipeline."""

import sys, os, traceback

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

errors = []
warnings_list = []

def check(label, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"  {PASS} {label}")
            return True
        else:
            print(f"  {PASS} {label} → {result}")
            return True
    except Exception as e:
        print(f"  {FAIL} {label} → {e}")
        errors.append((label, str(e)))
        return False

def warn_check(label, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"  {PASS} {label}")
        else:
            print(f"  {PASS} {label} → {result}")
    except Exception as e:
        print(f"  {WARN} {label} → {e}")
        warnings_list.append((label, str(e)))

# ─── SECTION 1: Core Python Libraries ───
print("\n[1/8] Core Python Libraries")
check("torch", lambda: __import__('torch'))
check("torchvision", lambda: __import__('torchvision'))
check("numpy", lambda: __import__('numpy'))
check("PIL (Pillow)", lambda: __import__('PIL'))
check("cv2 (OpenCV)", lambda: __import__('cv2'))
check("sklearn", lambda: __import__('sklearn'))
check("scipy", lambda: __import__('scipy'))
check("yaml (PyYAML)", lambda: __import__('yaml'))
check("easydict", lambda: __import__('easydict'))
check("diffusers", lambda: __import__('diffusers'))
check("tqdm", lambda: __import__('tqdm'))
check("einops", lambda: __import__('einops'))
check("matplotlib", lambda: __import__('matplotlib'))
check("seaborn", lambda: __import__('seaborn'))
check("imageio", lambda: __import__('imageio'))
check("timm", lambda: __import__('timm'))
check("skimage (scikit-image)", lambda: __import__('skimage'))
warn_check("SimpleITK", lambda: __import__('SimpleITK'))
warn_check("pytorch_lightning", lambda: __import__('pytorch_lightning'))
check("joblib", lambda: __import__('joblib'))
check("json", lambda: __import__('json'))
check("sqlite3", lambda: __import__('sqlite3'))
check("csv", lambda: __import__('csv'))
check("hashlib", lambda: __import__('hashlib'))
check("argparse", lambda: __import__('argparse'))
check("collections", lambda: __import__('collections'))
check("pathlib", lambda: __import__('pathlib'))

# ─── SECTION 2: Version Compatibility ───
print("\n[2/8] Version Compatibility")
import torch, torchvision, numpy as np
check(f"PyTorch version: {torch.__version__}", lambda: True)
check(f"Torchvision version: {torchvision.__version__}", lambda: True)
check(f"NumPy version: {np.__version__}", lambda: True)

import diffusers
check(f"Diffusers version: {diffusers.__version__}", lambda: True)

import sklearn
check(f"Scikit-learn version: {sklearn.__version__}", lambda: True)

# Check DDIMScheduler API compatibility
def check_ddim_api():
    from diffusers import DDIMScheduler
    s = DDIMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear')
    s.set_timesteps(16)
    ts = s.timesteps
    assert len(ts) == 16, f"Expected 16 timesteps, got {len(ts)}"
    # Check step API
    sample = torch.randn(1, 5, 6, 6)
    noise = torch.randn(1, 5, 6, 6)
    out = s.step(model_output=noise, timestep=ts[0], sample=sample)
    assert hasattr(out, 'prev_sample'), "DDIMScheduler.step must return .prev_sample"
    return f"{len(ts)} timesteps, step API OK"
check("DDIMScheduler API compatibility", check_ddim_api)

# Check sklearn LogisticRegression API
def check_sklearn_api():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    p = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    X = np.random.randn(20, 5)
    y = np.array([0,1,2,3,4]*4)
    p.fit(X, y)
    probs = p.predict_proba(X)
    assert probs.shape == (20, 5), f"Expected (20,5), got {probs.shape}"
    return "Pipeline+LogisticRegression OK"
check("sklearn Pipeline API", check_sklearn_api)

# ─── SECTION 3: CUDA / GPU ───
print("\n[3/8] CUDA / GPU")
check("CUDA available", lambda: torch.cuda.is_available() or (_ for _ in ()).throw(RuntimeError("No CUDA")))
if torch.cuda.is_available():
    check(f"GPU: {torch.cuda.get_device_name(0)}", lambda: True)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    check(f"VRAM: {vram_gb:.1f} GB", lambda: True)
    check(f"CUDA version: {torch.version.cuda}", lambda: True)
    # Test GPU allocation
    def test_gpu_alloc():
        t = torch.randn(1, 10, 5, 6, 6, device='cuda')
        assert t.is_cuda
        del t
        torch.cuda.empty_cache()
        return "Allocation/deallocation OK"
    check("GPU memory allocation", test_gpu_alloc)

# ─── SECTION 4: Project Module Imports ───
print("\n[4/8] Project Module Imports (pipeline code)")
os.chdir('/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019')
sys.path.insert(0, '.')

check("pipeline (SR3Sampler, SR3scheduler)", lambda: __import__('pipeline'))
check("model (ConditionalModel)", lambda: __import__('model'))
check("pretraining.dcg (DCG)", lambda: __import__('pretraining.dcg'))
check("pretraining.tools", lambda: __import__('pretraining.tools'))
check("pretraining.modules", lambda: __import__('pretraining.modules'))
check("aptos_research.data", lambda: __import__('aptos_research.data'))
check("aptos_research.features (FeatureStore)", lambda: __import__('aptos_research.features'))
check("aptos_research.extract", lambda: __import__('aptos_research.extract'))
check("aptos_research.diffmic_adapter", lambda: __import__('aptos_research.diffmic_adapter'))
check("aptos_research.metrics", lambda: __import__('aptos_research.metrics'))
check("aptos_research.models", lambda: __import__('aptos_research.models'))
check("aptos_research.experiments", lambda: __import__('aptos_research.experiments'))
check("aptos_research.gate", lambda: __import__('aptos_research.gate'))
check("aptos_research.run_qrc", lambda: __import__('aptos_research.run_qrc'))
check("aptos_research.trust", lambda: __import__('aptos_research.trust'))
check("aptos_research.trust_eval", lambda: __import__('aptos_research.trust_eval'))
check("aptos_research.explain", lambda: __import__('aptos_research.explain'))
check("aptos_research.xai_eval", lambda: __import__('aptos_research.xai_eval'))
check("aptos_research.cli", lambda: __import__('aptos_research.cli'))
check("aptos_research.registry", lambda: __import__('aptos_research.registry'))
check("aptos_research.checkpoint", lambda: __import__('aptos_research.checkpoint'))
check("aptos_research.train_gru", lambda: __import__('aptos_research.train_gru'))

# ─── SECTION 5: Dataset & Checkpoint Files ───
print("\n[5/8] Dataset & Checkpoint Files")
from pathlib import Path

data_root = '/home/pratyum/diffmic/data/aptos'
check("APTOS data root exists", lambda: Path(data_root).is_dir() or (_ for _ in ()).throw(FileNotFoundError(data_root)))

csv_path = os.path.join(data_root, 'train.csv')
check("train.csv exists", lambda: Path(csv_path).is_file() or (_ for _ in ()).throw(FileNotFoundError(csv_path)))

# Count images
import csv as csv_mod
def count_images():
    with open(csv_path) as f:
        reader = csv_mod.reader(f)
        next(reader)  # skip header
        rows = list(reader)
    img_dir = os.path.join(data_root, 'train_images')
    found = 0
    missing = []
    for row in rows:
        img_id = row[0]
        img_file = os.path.join(img_dir, f'{img_id}.png')
        if os.path.isfile(img_file):
            found += 1
        else:
            if len(missing) < 5:
                missing.append(img_id)
    if missing:
        raise FileNotFoundError(f"{len(rows)-found} images missing, e.g. {missing}")
    return f"{found}/{len(rows)} images found"
check("All APTOS images present", count_images)

ckpt_path = '/home/pratyum/diffmic/checkpoints/v2/ckpts/aptos-epoch117-acc-0.8186-f1-0.6840.ckpt'
check(f"Denoiser checkpoint ({Path(ckpt_path).stat().st_size/(1024**3):.2f} GB)", lambda: Path(ckpt_path).is_file())
aux_ckpt = '/home/pratyum/diffmic/code/DiffMICv2/pretraining/ckpt/aptos_aux_model.pth'
check(f"DCG aux checkpoint ({Path(aux_ckpt).stat().st_size/(1024**2):.0f} MB)", lambda: Path(aux_ckpt).is_file())

splits_path = '/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/outputs/aptos_splits.json'
def check_splits():
    import json
    with open(splits_path) as f:
        data = json.load(f)
    splits = data['splits']
    counts = {k: len(v) for k, v in splits.items()}
    total = sum(counts.values())
    assert total == 3662, f"Expected 3662 total, got {total}"
    return f"train={counts['train']}, val={counts['validation']}, test={counts['test']}, total={total}"
check("Split manifest valid", check_splits)

resnet_cache = os.path.expanduser('~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
check(f"ResNet18 pre-cached", lambda: Path(resnet_cache).is_file() or (_ for _ in ()).throw(FileNotFoundError("Not cached! Will re-download.")))

# Data symlink
data_link = '/home/pratyum/diffmic_anmol/data'
def check_symlink():
    if os.path.islink(data_link):
        target = os.readlink(data_link)
        assert os.path.isdir(data_link), f"Symlink broken: {data_link} → {target}"
        return f"symlink → {target}"
    elif os.path.isdir(data_link):
        return "directory (not symlink)"
    else:
        raise FileNotFoundError(data_link)
check("Data symlink intact", check_symlink)

# ─── SECTION 6: Model Loading & Memory Budget ───
print("\n[6/8] Model Loading & Memory Budget")

def check_model_loading():
    from aptos_research.diffmic_adapter import factory
    sampler, batches = factory()
    # Verify scheduler is set to 16 steps
    ts = list(sampler.scheduler.timesteps)
    assert len(ts) == 16, f"Expected 16 timesteps, got {len(ts)}"
    # Verify checkpoint hashes are set
    assert hasattr(sampler, 'checkpoint_id') and sampler.checkpoint_id, "Missing checkpoint_id"
    return f"Models loaded, {len(ts)} timesteps, ckpt_id set"
check("Factory: load models + verify config", check_model_loading)

def check_gpu_memory_after_load():
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    return f"allocated={alloc:.2f} GB, reserved={reserved:.2f} GB, free={free:.1f} GB"
check("GPU memory after model load", check_gpu_memory_after_load)

# ─── SECTION 7: Single-Batch End-to-End Smoke Test ───
print("\n[7/8] Single-Batch End-to-End Smoke Test")

def smoke_extract():
    """Extract 8 samples and verify the full pipeline path."""
    import subprocess
    r = subprocess.run([
        sys.executable, '-m', 'aptos_research.extract',
        '--factory', 'aptos_research.diffmic_adapter:factory',
        '--checkpoint', ckpt_path,
        '--cache', 'outputs/preflight_smoke.sqlite',
        '--inference-steps', '16',
        '--pooling', 'flatten',
        '--max-samples', '8',
        '--seed', '2026'
    ], capture_output=True, text=True, cwd='/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019')
    if r.returncode != 0:
        raise RuntimeError(f"Extract failed:\n{r.stderr[-500:]}")
    import json
    out = json.loads(r.stdout.strip().split('\n')[-1])
    return f"Extracted {out['new_samples']} samples"
check("Stage 1 smoke: extract 8 samples", smoke_extract)

def smoke_inspect():
    from aptos_research.features import FeatureStore
    s = FeatureStore('outputs/preflight_smoke.sqlite')
    rows = s.load_split('train')
    s.close()
    if not rows:
        raise RuntimeError("No train rows in smoke cache")
    states = rows[0]['states']
    assert np.isfinite(states).all(), "Non-finite states found!"
    K, D = states.shape
    return f"{len(rows)} samples, states shape=({K}, {D}), all finite"
check("Cache integrity: all states finite", smoke_inspect)

def smoke_experiments():
    """We need train+val+test to run experiments. Check that the cache has enough."""
    from aptos_research.features import FeatureStore
    s = FeatureStore('outputs/preflight_smoke.sqlite')
    splits = {}
    for split in ('train', 'validation', 'test'):
        rows = s.load_split(split)
        splits[split] = len(rows)
    s.close()
    return f"Cache splits: {splits}"
warn_check("Cache split coverage (smoke only has train)", smoke_experiments)

# ─── SECTION 8: Disk Space & Estimated Output Size ───
print("\n[8/8] Disk Space & Estimated Output Size")

def check_disk():
    st = os.statvfs('/home/pratyum')
    free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
    if free_gb < 5:
        raise RuntimeError(f"Only {free_gb:.1f} GB free! Pipeline needs ~2-5 GB for outputs")
    return f"{free_gb:.1f} GB free (need ~2-5 GB for full extraction)"
check("Disk space", check_disk)

def estimate_output():
    # 3662 samples × 16 states × (5×6×6=180 floats × 4 bytes) = ~42 MB raw states
    # SQLite overhead ~2x → ~84 MB
    # classical.json ~5 MB, GRU training temporary ~200 MB GPU
    # QRC is CPU-only, small
    raw = 3662 * 16 * 180 * 4
    return f"Estimated trajectory cache: ~{raw/(1024**2):.0f} MB raw, ~{raw*2/(1024**2):.0f} MB SQLite"
check("Estimated output size", estimate_output)

# Cleanup smoke file
try:
    os.remove('outputs/preflight_smoke.sqlite')
except:
    pass

# ─── SUMMARY ───
print("\n" + "="*60)
if errors:
    print(f"{FAIL} PRE-FLIGHT FAILED: {len(errors)} error(s)")
    for label, err in errors:
        print(f"    {FAIL} {label}: {err}")
else:
    print(f"{PASS} PRE-FLIGHT PASSED: All checks OK")

if warnings_list:
    print(f"{WARN} {len(warnings_list)} warning(s) (non-blocking):")
    for label, err in warnings_list:
        print(f"    {WARN} {label}: {err}")

print("="*60)
sys.exit(1 if errors else 0)
