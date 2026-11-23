"""
galaxy_ddpm_v2.py  --  Galaxy10 DECaLS 32x32: DDPM (paper method) vs DCGAN baseline, CPU-friendly.

Changes vs v1:
  * EMA fix: two EMAs (0.999 / 0.9995) with warm-up; the better-suited one is picked automatically
  * all figures are saved upscaled (bicubic) so they look clean
  * FID (pytorch-fid) on 2100 samples per model, incl. a real-vs-real floor
  * DCGAN baseline trained for a fixed time budget
  * physical-property metrics (SGD-style) with more samples, for DDPM, DCGAN and the real-vs-real baseline
  * nearest-neighbour grids, blind real-or-fake test (+ answer key), inpainting with an error number
  * everything resumable; results go to ./results_v2 (v1 results are not touched)

Run (PowerShell, in the folder that contains galaxies_32.npy):
    python -u galaxy_ddpm_v2.py --quick     # ~3-5 min smoke test
    python -u galaxy_ddpm_v2.py             # full run
"""
import os, sys, time, math, json, re, argparse, subprocess
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from scipy.stats import wasserstein_distance

# ---------------- config ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results_v2")
ap.add_argument("--train-min", type=float, default=30, help="DDPM training budget (minutes)")
ap.add_argument("--gan-min", type=float, default=10, help="DCGAN training budget (minutes)")
ap.add_argument("--n-gen", type=int, default=2100, help="samples per model (FID needs > 2048)")
ap.add_argument("--n-ref", type=int, default=5000, help="real reference images for FID/metrics")
ap.add_argument("--threads", type=int, default=20)
ap.add_argument("--retrain", action="store_true", help="ignore a saved DDPM checkpoint and train again")
ap.add_argument("--skip-gan", action="store_true")
ap.add_argument("--load-ema", default=None, help="reuse v1 weights (e.g. results/ema.pt) instead of training")
ap.add_argument("--quick", action="store_true", help="tiny budgets, to check that nothing crashes")
A = ap.parse_args()
if A.quick:
    A.train_min, A.gan_min, A.n_gen, A.n_ref, A.out = 0.3, 0.2, 64, 200, "results_quick"

IMG, T, BATCH = 32, 500, 64
OUT = A.out; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(A.threads, os.cpu_count() or A.threads))
torch.manual_seed(0); np.random.seed(0)
print("device:", dev, "| threads:", torch.get_num_threads(), "| output folder:", OUT, flush=True)

RES_P = f"{OUT}/results.json"
R = json.load(open(RES_P)) if os.path.exists(RES_P) else {}
def save_res(): json.dump(R, open(RES_P, "w"), indent=2)
R["config"] = {"img": IMG, "T": T, "train_min": A.train_min, "gan_min": A.gan_min,
               "n_gen": A.n_gen, "n_ref": A.n_ref}

# ---------------- data ----------------
H5, CACHE = "Galaxy10_DECals.h5", f"galaxies_{IMG}.npy"
URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1"
if not os.path.exists(CACHE):
    import h5py, urllib.request
    if not os.path.exists(H5):
        print("downloading Galaxy10 DECaLS (~2.7GB)...")
        urllib.request.urlretrieve(URL, H5)
    with h5py.File(H5, "r") as f:
        imgs = f["images"]; chunks = []
        for i in range(0, len(imgs), 500):
            x = torch.from_numpy(imgs[i:i + 500]).permute(0, 3, 1, 2).float() / 255.
            chunks.append(F.interpolate(x, size=IMG, mode="area").numpy())
    np.save(CACHE, np.concatenate(chunks))
data_cpu = torch.from_numpy(np.load(CACHE)) * 2 - 1          # [-1, 1], CPU copy
data = data_cpu.to(dev)
print("data:", tuple(data.shape), flush=True)

# ---------------- model (same small U-Net as v1) ----------------
class TimeEmb(nn.Module):
    def __init__(s, d):
        super().__init__(); s.d = d
        s.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d * 4))
    def forward(s, t):
        h = s.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
        a = t[:, None].float() * f[None]
        return s.mlp(torch.cat([a.sin(), a.cos()], 1))

class Block(nn.Module):
    def __init__(s, i, o, td):
        super().__init__()
        s.n1 = nn.GroupNorm(8, i); s.c1 = nn.Conv2d(i, o, 3, padding=1)
        s.n2 = nn.GroupNorm(8, o); s.c2 = nn.Conv2d(o, o, 3, padding=1)
        s.t = nn.Linear(td, o)
        s.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x, t):
        h = s.c1(F.silu(s.n1(x))) + s.t(t)[:, :, None, None]
        h = s.c2(F.silu(s.n2(h)))
        return h + s.skip(x)

class UNet(nn.Module):
    def __init__(s, C=32):
        super().__init__()
        td = C * 4
        s.temb = TimeEmb(C)
        s.inc = nn.Conv2d(3, C, 3, padding=1)
        s.d1 = Block(C, C, td); s.d2 = Block(C, 2 * C, td); s.d3 = Block(2 * C, 4 * C, td)
        s.mid = Block(4 * C, 4 * C, td)
        s.u2 = Block(4 * C + 2 * C, 2 * C, td); s.u1 = Block(2 * C + C, C, td)
        s.out = nn.Conv2d(C, 3, 1)
    def forward(s, x, t):
        t = s.temb(t)
        x0 = s.inc(x)
        h1 = s.d1(x0, t)
        h2 = s.d2(F.avg_pool2d(h1, 2), t)
        h3 = s.d3(F.avg_pool2d(h2, 2), t)
        m = s.mid(h3, t)
        u = s.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), t)
        u = s.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), t)
        return s.out(u)

betas = torch.linspace(2e-4, 0.04, T, device=dev)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)

def augment(x):
    if np.random.rand() < .5: x = x.flip(3)
    return torch.rot90(x, np.random.randint(4), (2, 3))

# ---------------- 1. train DDPM (paper Eq. 7) ----------------
CKP = f"{OUT}/ddpm.pt"
ema = UNet().to(dev)
if A.load_ema:
    ema.load_state_dict(torch.load(A.load_ema, map_location=dev)); losses = []
    R["ddpm_training"] = {"note": f"loaded existing EMA weights from {A.load_ema}"}; save_res()
    print("loaded existing EMA weights:", A.load_ema, flush=True)
elif os.path.exists(CKP) and not A.retrain:
    c = torch.load(CKP, map_location=dev)
    ema.load_state_dict(c["ema"]); losses = c["losses"]
    print(f"loaded DDPM checkpoint ({c['step']} steps, EMA {c['ema_decay']})", flush=True)
else:
    model = UNet().to(dev)
    EMA_DECAYS = (0.999, 0.9995)
    emas = {d: UNet().to(dev) for d in EMA_DECAYS}
    for e in emas.values(): e.load_state_dict(model.state_dict())
    mp = list(model.parameters()); ep = {d: list(e.parameters()) for d, e in emas.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    print("U-Net params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2), flush=True)
    losses, step, t0 = [], 0, time.time()
    while time.time() - t0 < A.train_min * 60:
        x0 = augment(data[torch.randint(0, len(data), (BATCH,), device=dev)])
        t = torch.randint(0, T, (BATCH,), device=dev)
        z = torch.randn_like(x0)
        xt = abar[t].sqrt()[:, None, None, None] * x0 + (1 - abar[t]).sqrt()[:, None, None, None] * z
        loss = F.mse_loss(model(xt, t), z)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for d, lst in ep.items():
                dd = min(d, (1 + step) / (10 + step))          # warm-up
                torch._foreach_mul_(lst, dd); torch._foreach_add_(lst, mp, alpha=1 - dd)
        losses.append(loss.item()); step += 1
        if step % 500 == 0:
            print(f"[ddpm] step {step}  loss {np.mean(losses[-500:]):.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    chosen = 0.9995 if step >= 30000 else 0.999     # a longer EMA horizon needs enough steps
    ema.load_state_dict(emas[chosen].state_dict())
    print(f"[ddpm] finished: {step} steps, using EMA {chosen}", flush=True)
    torch.save({"ema": ema.state_dict(), "step": step, "losses": losses, "ema_decay": chosen}, CKP)
    R["ddpm_training"] = {"steps": step, "ema_decay": chosen, "minutes": A.train_min}; save_res()
ema.eval()
if len(losses) > 1:
    sm = np.convolve(losses, np.ones(100) / 100, mode="valid") if len(losses) > 100 else np.array(losses)
    plt.figure(); plt.plot(sm); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("MSE loss")
    plt.title("DDPM training loss"); plt.savefig(f"{OUT}/loss.png", dpi=150); plt.close()

# ---------------- sampling (paper Eq. 8) + guided inpainting (Sec. 4.2) ----------------
@torch.inference_mode()
def sample(n, known=None, mask=None):
    x = torch.randn(n, 3, IMG, IMG, device=dev)
    for t in reversed(range(T)):
        eps = ema(x, torch.full((n,), t, device=dev))
        x = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0: x = x + betas[t].sqrt() * torch.randn_like(x)
        if known is not None:
            kn = known if t == 0 else abar[t - 1].sqrt() * known + (1 - abar[t - 1]).sqrt() * torch.randn_like(known)
            x = mask * kn + (1 - mask) * x
    return x.clamp(-1, 1)

def gen_ddpm():
    fp, pp = f"{OUT}/fake_ddpm.pt", f"{OUT}/fake_ddpm_part.pt"
    if os.path.exists(fp): return torch.load(fp)
    part = torch.load(pp) if os.path.exists(pp) else torch.empty(0, 3, IMG, IMG)
    n0, t0 = len(part), time.time()
    while len(part) < A.n_gen:
        part = torch.cat([part, sample(min(100, A.n_gen - len(part))).cpu()]); torch.save(part, pp)
        el, done = time.time() - t0, len(part) - n0
        print(f"[ddpm sampling] {len(part)}/{A.n_gen}  elapsed {el / 60:.1f} min  "
              f"eta {el / done * (A.n_gen - len(part)) / 60:.1f} min", flush=True)
    torch.save(part, fp)
    R["ddpm_sampling"] = {"sampler": f"ancestral, {T} steps", "n": A.n_gen, "minutes": round((time.time() - t0) / 60, 1)}
    save_res(); return part

# ---------------- 2. DCGAN baseline ----------------
class Gen(nn.Module):
    def __init__(s, nz=100, f=64):
        super().__init__()
        s.net = nn.Sequential(
            nn.ConvTranspose2d(nz, f * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(f * 8), nn.ReLU(True),      # 4
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 4), nn.ReLU(True),   # 8
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 2), nn.ReLU(True),   # 16
            nn.ConvTranspose2d(f * 2, 3, 4, 2, 1, bias=False), nn.Tanh())                                  # 32
    def forward(s, z): return s.net(z)

class Disc(nn.Module):
    def __init__(s, f=64):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(3, f, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                                 # 16
            nn.Conv2d(f, f * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 2), nn.LeakyReLU(0.2, True),      # 8
            nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 4), nn.LeakyReLU(0.2, True),  # 4
            nn.Conv2d(f * 4, 1, 4, 1, 0, bias=False))
    def forward(s, x): return s.net(x).view(-1)

def init_w(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.normal_(m.weight, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d): nn.init.normal_(m.weight, 1, 0.02); nn.init.zeros_(m.bias)

def gen_gan():
    fp, gp = f"{OUT}/fake_gan.pt", f"{OUT}/dcgan.pt"
    if os.path.exists(fp): return torch.load(fp)
    G, D = Gen().to(dev), Disc().to(dev); G.apply(init_w); D.apply(init_w)
    if os.path.exists(gp): G.load_state_dict(torch.load(gp, map_location=dev))
    else:
        oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
        oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
        bce, bs, step, t0 = nn.BCEWithLogitsLoss(), 128, 0, time.time()
        real_l, fake_l = torch.full((bs,), 0.9, device=dev), torch.zeros(bs, device=dev)   # one-sided label smoothing
        while time.time() - t0 < A.gan_min * 60:
            x = augment(data[torch.randint(0, len(data), (bs,), device=dev)])
            fake = G(torch.randn(bs, 100, 1, 1, device=dev))
            oD.zero_grad(); dl = bce(D(x), real_l) + bce(D(fake.detach()), fake_l); dl.backward(); oD.step()
            oG.zero_grad(); gl = bce(D(fake), torch.ones(bs, device=dev)); gl.backward(); oG.step()
            step += 1
            if step % 500 == 0:
                print(f"[gan] step {step}  D {dl.item():.3f}  G {gl.item():.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save(G.state_dict(), gp)
        R["dcgan_training"] = {"steps": step, "minutes": A.gan_min}; save_res()
    with torch.no_grad():     # BatchNorm left in train mode (batch statistics), as is common for DCGAN sampling
        out = torch.cat([G(torch.randn(100, 100, 1, 1, device=dev)).cpu() for _ in range(math.ceil(A.n_gen / 100))])[:A.n_gen]
    torch.save(out, fp); return out

# ---------------- run sampling ----------------
fake_ddpm = gen_ddpm()
fake_gan = None if A.skip_gan else gen_gan()

# ---------------- real subsets (disjoint): "floor" = real vs real baseline, "ref" = reference ----------------
perm = torch.randperm(len(data_cpu), generator=torch.Generator().manual_seed(0))
floor_idx, ref_idx = perm[:A.n_gen], perm[A.n_gen:A.n_gen + A.n_ref]
real_floor, real_ref = data_cpu[floor_idx], data_cpu[ref_idx]
sets = {"real_floor": real_floor, "ddpm": fake_ddpm}
if fake_gan is not None: sets["gan"] = fake_gan

# ---------------- figures (upscaled so they look clean) ----------------
def tile(x, nrow, scale=4):
    x = ((x.cpu() + 1) / 2).clamp(0, 1)
    x = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)
    return make_grid(x, nrow=nrow, padding=4, pad_value=1.0)

def grid(x, path, nrow=8): save_image(tile(x, nrow), path)

grid(real_ref[:64], f"{OUT}/real.png"); grid(fake_ddpm[:64], f"{OUT}/generated_ddpm.png")
if fake_gan is not None: grid(fake_gan[:64], f"{OUT}/generated_dcgan.png")
panels = [("Real", real_ref[:36]), ("DDPM", fake_ddpm[:36])] + ([("DCGAN", fake_gan[:36])] if fake_gan is not None else [])
fig, ax = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.3))
for a, (name, x) in zip(np.atleast_1d(ax), panels):
    a.imshow(tile(x, 6).permute(1, 2, 0).numpy()); a.set_title(name, fontsize=16); a.axis("off")
plt.savefig(f"{OUT}/comparison.png", dpi=150, bbox_inches="tight"); plt.close()

def nn_grid(fake, path, n=8, k=4):                          # memorisation check (paper Fig. 3)
    idx = torch.cdist(fake[:n].flatten(1), data_cpu.flatten(1)).topk(k, largest=False).indices
    rows = torch.cat([torch.cat([fake[i:i + 1], data_cpu[idx[i]]]) for i in range(n)])
    grid(rows, path, nrow=k + 1)
nn_grid(fake_ddpm, f"{OUT}/nearest_neighbours_ddpm.png")
if fake_gan is not None: nn_grid(fake_gan, f"{OUT}/nearest_neighbours_dcgan.png")

mix = torch.cat([real_ref[:32], fake_ddpm[:32]])           # blind real-or-fake test (paper Fig. 2)
order = torch.randperm(64, generator=torch.Generator().manual_seed(1))
grid(mix[order], f"{OUT}/blind_test.png")
open(f"{OUT}/blind_test_ANSWER_KEY.txt", "w").write(
    "numbering: row-major, 8 per row, starting at 0\n" +
    "\n".join(f"{j}: {'FAKE' if int(o) >= 32 else 'real'}" for j, o in enumerate(order)))

# ---------------- FID ----------------
def dump(x, folder):
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder): os.remove(os.path.join(folder, f))
    for i, im in enumerate(x): save_image(im * 0.5 + 0.5, f"{folder}/{i:05d}.png")

def fid_score(ref, d):
    cmd = [sys.executable, "-m", "pytorch_fid", ref, d, "--device", dev, "--batch-size", "64", "--num-workers", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"FID:\s*([0-9.eE+-]+)", r.stdout)
    if not m: raise RuntimeError("pytorch-fid failed:\n" + r.stdout[-1500:] + r.stderr[-1500:])
    return float(m.group(1))

if A.n_gen > 2048:
    R.setdefault("fid", {}); FID_DIR = f"{OUT}/fid"
    if not os.path.isdir(f"{FID_DIR}/real_ref") or len(os.listdir(f"{FID_DIR}/real_ref")) != len(real_ref):
        dump(real_ref, f"{FID_DIR}/real_ref")
    for name, x in sets.items():
        if name in R["fid"]: continue
        dump(x, f"{FID_DIR}/{name}"); t0 = time.time()
        R["fid"][name] = fid_score(f"{FID_DIR}/real_ref", f"{FID_DIR}/{name}"); save_res()
        print(f"[fid] {name} vs real_ref: {R['fid'][name]:.2f}  ({time.time() - t0:.0f}s)", flush=True)
else:
    print("FID skipped (needs more than 2048 samples).")

# ---------------- physical-property metrics (SGD-style, paper Eq. 11 / Table 1 / Fig. 4) ----------------
yy, xx = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")
rad = ((yy - IMG / 2).pow(2) + (xx - IMG / 2).pow(2)).sqrt().flatten(); order_r = rad.argsort()

def props(x):
    x = ((x.cpu() + 1) / 2).clamp(0, 1); gray = x.mean(1)
    flux = gray.sum((1, 2))
    cs = gray.flatten(1)[:, order_r].cumsum(1)
    re = rad[order_r][(cs < 0.5 * cs[:, -1:]).sum(1).clamp(max=IMG * IMG - 1)]    # half-light radius
    return {"total flux": flux, "half-light radius": re,
            "colour (R-G)": x[:, 0].mean((1, 2)) - x[:, 1].mean((1, 2)),
            "colour (G-B)": x[:, 1].mean((1, 2)) - x[:, 2].mean((1, 2))}

def cohens_d(a, b):
    a, b = a.numpy(), b.numpy()
    return float(abs(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

pref = props(real_ref); pset = {k: props(v) for k, v in sets.items()}
lines = ["property," + ",".join(f"W1_{k}" for k in sets) + "," + ",".join(f"d_{k}" for k in sets if k != "real_floor")]
tot = {k: 0.0 for k in sets}
for p in pref:
    w = {k: float(wasserstein_distance(pset[k][p].numpy(), pref[p].numpy())) for k in sets}
    ds = {k: cohens_d(pset[k][p], pref[p]) for k in sets if k != "real_floor"}
    for k in sets: tot[k] += w[k]
    lines.append(f"{p}," + ",".join(f"{w[k]:.4f}" for k in sets) + "," + ",".join(f"{ds[k]:.3f}" for k in ds))
lines.append("SGD-style sum," + ",".join(f"{tot[k]:.4f}" for k in sets) + "," + "," * (len(sets) - 2))
open(f"{OUT}/metrics.csv", "w").write("\n".join(lines))
R["sgd_style_sum"] = {k: round(v, 4) for k, v in tot.items()}; save_res()
print("\n".join(lines), flush=True)

fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
for a, p in zip(ax, pref):
    a.hist(pref[p].numpy(), 30, alpha=.55, density=True, label="real")
    for k in ("ddpm", "gan"):
        if k in pset: a.hist(pset[k][p].numpy(), 30, alpha=.45, density=True, label=k.upper())
    a.set_title(p)
ax[0].legend(); plt.savefig(f"{OUT}/property_histograms.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------------- inpainting a fake satellite trail (paper Sec. 4.2) ----------------
orig = data[1000:1008]
gy, gx = torch.meshgrid(torch.arange(IMG, device=dev), torch.arange(IMG, device=dev), indexing="ij")
trail = ((gy - gx * 0.7 - 5).abs() < 1.5).float()
mask = (1 - trail)[None, None].expand(8, 3, -1, -1)
defaced = orig * mask + trail[None, None] * 1.0
rec = sample(8, known=orig, mask=mask)
tr3 = ((1 - mask) > 0.5)
mae_rec = float(((rec - orig).abs()[tr3]).mean() / 2)
fill = orig.flatten(2).median(2).values[:, :, None, None].expand_as(orig)            # naive baseline: per-channel median
mae_fill = float(((fill - orig).abs()[tr3]).mean() / 2)
R["inpainting_MAE_on_trail_pixels"] = {"ddpm_inpainting": round(mae_rec, 4), "naive_median_fill": round(mae_fill, 4)}; save_res()
print(f"[inpaint] MAE on trail pixels (0-1 scale): DDPM {mae_rec:.4f} vs median-fill {mae_fill:.4f}", flush=True)
fig, ax = plt.subplots(3, 8, figsize=(16, 6.4))
for j in range(8):
    for r, im in enumerate([orig, defaced, rec]):
        ax[r, j].imshow(((im[j].cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0), interpolation="bicubic"); ax[r, j].axis("off")
ax[0, 0].set_title("original"); ax[1, 0].set_title("defaced"); ax[2, 0].set_title("inpainted")
plt.savefig(f"{OUT}/inpainting.png", dpi=150, bbox_inches="tight"); plt.close()

print("\nDONE. Everything is saved in:", os.path.abspath(OUT))
print(json.dumps({k: R[k] for k in ("fid", "sgd_style_sum") if k in R}, indent=2))
