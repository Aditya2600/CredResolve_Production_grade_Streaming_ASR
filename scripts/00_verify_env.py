import sys, subprocess

def fail(msg):
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)

# --- 1. CUDA + bf16 ---
import torch
if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False")
if not torch.cuda.is_bf16_supported():
    fail("bf16 unsupported — do not proceed with bf16-mixed")

props = torch.cuda.get_device_properties(0)
print(f"[ok] GPU        : {props.name}")
print(f"[ok] VRAM       : {props.total_memory / 1024**3:.1f} GiB")
print(f"[ok] Capability : sm_{props.major}{props.minor}")
print(f"[ok] torch      : {torch.__version__}")
print(f"[ok] torch CUDA : {torch.version.cuda}")
print(f"[ok] bf16       : supported")

# --- 2. NeMo import ---
try:
    import nemo
    import nemo.collections.asr as nemo_asr  # noqa: F401
except Exception as e:
    fail(f"NeMo import failed: {e}")
print(f"[ok] NeMo       : {nemo.__version__}")

# --- 3. THE REAL TEST: RNNT loss fwd+bwd on GPU ---
from nemo.collections.asr.losses.rnnt import RNNTLoss

V = 129
loss_fn = RNNTLoss(num_classes=V - 1).cuda()

B, T, U = 2, 50, 30
logits  = torch.randn(B, T, U + 1, V, device="cuda", requires_grad=True)
targets = torch.randint(1, V, (B, U), dtype=torch.int32, device="cuda")
t_len   = torch.full((B,), T, dtype=torch.int32, device="cuda")
u_len   = torch.full((B,), U, dtype=torch.int32, device="cuda")

try:
    loss = loss_fn(log_probs=logits, targets=targets,
                   input_lengths=t_len, target_lengths=u_len)
    loss.backward()
except Exception as e:
    fail(f"RNNT loss fwd/bwd failed — warprnnt_numba is broken: {e}")

if logits.grad is None or not torch.isfinite(logits.grad).all():
    fail("RNNT produced null or non-finite gradients")

print(f"[ok] RNNT loss  : {loss.item():.4f}, grads finite")

# --- 4. Numba CUDA visibility ---
try:
    from numba import cuda as numba_cuda
    if not numba_cuda.is_available():
        fail("numba.cuda.is_available() is False")
    print(f"[ok] numba cuda : available")
except ImportError:
    fail("numba not installed")

# --- 5. /dev/shm ---
out = subprocess.run(["df", "-BG", "--output=avail", "/dev/shm"],
                     capture_output=True, text=True).stdout
shm = int(out.strip().split("\n")[-1].strip().rstrip("G"))
if shm < 32:
    fail(f"/dev/shm is {shm}G, need >=32G. Relaunch with --shm-size=64g")
print(f"[ok] /dev/shm   : {shm}G")

# --- 6. ffmpeg codecs ---
r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                   capture_output=True, text=True)
for enc in ["pcm_mulaw", "libgsm", "libopencore_amrnb"]:
    if enc not in r.stdout:
        print(f"[warn] ffmpeg encoder missing: {enc}")
    else:
        print(f"[ok] ffmpeg     : {enc}")

print("\n=== Stage 0 PASSED ===")
