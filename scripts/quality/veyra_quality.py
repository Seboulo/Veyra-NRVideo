#!/usr/bin/env python3
"""Veyra image-quality regression tests (held-out frame method).

Principle (Baker et al., Middlebury, IJCV 2011; same as Vimeo-90K triplets): from a real
capture, keep one real game frame out of two, let Veyra generate the missing one, then
compare the generated frame with the real frame that was removed.

  prepare  capture.mp4 -> clips/<name>/ {ref.mkv, input.mp4, manifest.json}
           ref.mkv   = every distinct game frame (Elgato duplicates removed), lossless FFV1
           input.mp4 = real frames 0, 2, 4, ... only (half rate), what Veyra must export at 2X
  score    clips/<name> veyra_out.mp4 -> clips/<name>/results/<label>/{result.json, worst_*.png}
  gate     result.json baseline.json  -> exit code 1 if quality regressed
  suite    clips/ --label X           -> score + gate every clip, summary.md
  selftest                            -> synthetic check of the tool itself (used by CI)

Metrics on each generated frame (only frames whose source cadence was regular):
  PSNR / SSIM (luma, BT.709), NVIDIA FLIP (Andersson et al., HPG 2020: visible differences
  when flipping between images) mean and 99th percentile, the same FLIP restricted to the
  character zone ("head"), the static HUD zone and motion edges (where halos appear),
  and the same numbers for a plain 50/50 blend of the neighbours (the floor any frame
  generation must beat). Optional LPIPS (Zhang et al., CVPR 2018) with --lpips.

Requires: Python 3.10+, numpy, opencv-python, flip-evaluator, ffmpeg + ffprobe on PATH.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("opencv manquant : pip install -r requirements.txt")

TOOL_VERSION = 1
THUMB_W = 320                       # width of the low-res copies used for dedup/alignment
DEFAULT_HEAD = [0.36, 0.28, 0.64, 0.92]   # x0, y0, x1, y1 (fractions): 3rd-person character
# Metrics compared by the gate: (path, higher_is_better, tolerance, tolerance kind)
GATE_KEYS = [
    ("full.flip_mean", False, 0.03, "rel"),
    ("full.flip_p99_worst5", False, 0.05, "rel"),
    ("head.flip_mean", False, 0.04, "rel"),
    ("head.flip_worst5", False, 0.06, "rel"),
    ("head.static_flip_mean", False, 0.06, "rel"),
    ("head.static_flip_worst5", False, 0.08, "rel"),
    ("static.flip_mean", False, 0.06, "rel"),
    ("motion.flip_mean", False, 0.04, "rel"),
    ("hud.flip_mean", False, 0.05, "rel"),
    ("full.psnr", True, 0.25, "abs"),
    ("full.ssim", True, 0.004, "abs"),
]
LABELS = {
    "full.flip_mean": "FLIP moyen (image entière)",
    "full.flip_p99_worst5": "FLIP p99, 5 % pires images",
    "head.flip_mean": "FLIP zone personnage",
    "head.flip_worst5": "FLIP personnage, 5 % pires",
    "head.static_flip_mean": "FLIP personnage immobile",
    "head.static_flip_worst5": "FLIP personnage immobile, 5 % pires",
    "static.flip_mean": "FLIP zones immobiles",
    "motion.flip_mean": "FLIP bords en mouvement (halos)",
    "hud.flip_mean": "FLIP HUD",
    "full.psnr": "PSNR (dB)",
    "full.ssim": "SSIM",
}


# --------------------------------------------------------------------------- video I/O
def need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        sys.exit(f"{tool} introuvable. Windows : winget install Gyan.FFmpeg, puis rouvrir la fenêtre.")
    return path


def probe(path: Path) -> dict:
    out = subprocess.run([need("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,color_space,color_range,nb_frames",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    if fps <= 0:
        num, den = s["r_frame_rate"].split("/")
        fps = float(num) / float(den)
    return dict(w=int(s["width"]), h=int(s["height"]), fps=fps,
                space=s.get("color_space", "unknown"), range=s.get("color_range", "unknown"))


class Reader:
    """Sequential RGB frame reader through an ffmpeg pipe, with random access forward only."""

    def __init__(self, path: Path, width: int | None = None, height: int | None = None,
                 start: float = 0.0, duration: float = 0.0):
        info = probe(path)
        self.w, self.h = width or info["w"], height or info["h"]
        vf = []
        # Untagged files would be decoded as BT.601 by default: Elgato/PS5 HD is BT.709.
        if info["space"] in ("unknown", "", None):
            vf.append("scale=in_color_matrix=bt709:in_range=tv")
        if (self.w, self.h) != (info["w"], info["h"]):
            vf.append(f"scale={self.w}:{self.h}:flags=area")
        cmd = [need("ffmpeg"), "-v", "error", "-nostdin"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(path)]
        if duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-map", "0:v:0", "-vsync", "passthrough"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        # stderr muted: closing the pipe early makes ffmpeg print harmless 'Broken pipe'.
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.path = path
        self.size = self.w * self.h * 3
        self.index = -1
        self.cache: dict[int, np.ndarray] = {}

    def next(self) -> np.ndarray | None:
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return None
        self.index += 1
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def get(self, i: int) -> np.ndarray | None:
        if i in self.cache:
            return self.cache[i]
        if i < self.index:
            raise RuntimeError(f"frame {i} déjà dépassée")
        frame = None
        while self.index < i:
            frame = self.next()
            if frame is None:
                return None
        self.cache[i] = frame
        for k in [k for k in self.cache if k < i - 3]:
            del self.cache[k]
        return frame

    def __iter__(self):
        while (f := self.next()) is not None:
            yield f

    def close(self):
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.wait()
        if self.index < 0:
            sys.exit(f"Impossible de lire {self.path} (fichier absent, vide ou codec non pris en charge)")


class Writer:
    def __init__(self, path: Path, w: int, h: int, fps: float, codec: str):
        enc = {
            # Lossless reference (FFV1 v3), BT.709 limited like the PS5/Elgato chain.
            "ffv1": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"],
            # Near-lossless but decodable by any hardware decoder (Veyra's file source).
            "x264": ["-c:v", "libx264", "-preset", "slow", "-crf", "10", "-profile:v", "high",
                     "-pix_fmt", "yuv420p", "-g", "30", "-bf", "0"],
        }[codec]
        cmd = [need("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv", *enc,
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-color_range", "tv", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            sys.exit("ffmpeg a échoué pendant l'encodage")


# --------------------------------------------------------------------------- image metrics
def luma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def thumb(rgb: np.ndarray) -> np.ndarray:
    h = max(2, round(rgb.shape[0] * THUMB_W / rgb.shape[1]))
    return cv2.resize(luma(rgb), (THUMB_W, h), interpolation=cv2.INTER_AREA)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    mse = float(d[mask].mean()) if mask is not None else float(d.mean())
    return 99.0 if mse <= 1e-10 else 10 * math.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian SSIM on luma, Wang et al. 2004 constants (K1=0.01, K2=0.03, 11x11, sigma 1.5)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a, b = a.astype(np.float32), b.astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean())


class Flip:
    def __init__(self, ppd: float):
        try:
            import flip_evaluator
        except ImportError:  # pragma: no cover
            sys.exit("flip-evaluator manquant : pip install -r requirements.txt")
        self.mod, self.params = flip_evaluator, {"ppd": ppd}

    def map(self, ref: np.ndarray, test: np.ndarray) -> np.ndarray:
        r = (ref.astype(np.float32) / 255.0)
        t = (test.astype(np.float32) / 255.0)
        m, _, _ = self.mod.evaluate(r, t, "LDR", inputsRGB=True, applyMagma=False,
                                    computeMeanError=False, parameters=dict(self.params))
        return m[..., 0] if m.ndim == 3 else m


class Lpips:
    def __init__(self):
        try:
            import lpips, torch
        except ImportError:
            sys.exit("--lpips demande : pip install torch lpips")
        self.torch, self.net = torch, lpips.LPIPS(net="alex", verbose=False)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = lambda x: self.torch.from_numpy(cv2.resize(x, (960, 540), interpolation=cv2.INTER_AREA)
                                            ).permute(2, 0, 1)[None].float() / 127.5 - 1
        with self.torch.no_grad():
            return float(self.net(t(a), t(b)).item())


def rect_mask(h: int, w: int, rect) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x0, y0, x1, y1 = rect
    m[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = True
    return m


def motion_mask(prev: np.ndarray, nxt: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Where the two real neighbours differ strongly, widened: object edges in motion,
    where frame generation leaves halos, double edges and disocclusion holes."""
    d = np.abs(luma(prev) - luma(nxt))
    d = cv2.GaussianBlur(d, (0, 0), 1.0) > 12.0
    k = max(3, (prev.shape[1] // 640) * 4 + 1)
    d = cv2.dilate(d.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return d & ~exclude


def static_mask(prev: np.ndarray, nxt: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pixels that do not move at all between the two real neighbours and the real middle
    frame. A 50/50 blend is already perfect there, so any error is damage done by frame
    generation: a third-person character standing still while the camera orbits (the
    "vanishing head"), or HUD text dragged by the background."""
    yp, yn, yt = luma(prev), luma(nxt), luma(truth)
    still = (cv2.GaussianBlur(np.abs(yp - yn), (0, 0), 1.0) < 3.0) & \
            (cv2.GaussianBlur(np.abs(yt - yp), (0, 0), 1.0) < 3.0)
    return cv2.erode(still.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


# --------------------------------------------------------------------------- prepare
def cmd_prepare(a):
    src = Path(a.capture)
    out = Path(a.out) / a.name
    if out.exists() and not a.force:
        sys.exit(f"{out} existe déjà (ajoute --force pour le refaire)")
    out.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    print(f"Source : {info['w']}x{info['h']} à {info['fps']:.2f} i/s, couleur={info['space']}/{info['range']}")

    # Pass 1: low-res luma of every captured frame -> which frames are new game frames.
    rd = Reader(src, start=a.start, duration=a.duration)
    thumbs = [thumb(f) for f in rd]
    rd.close()
    if len(thumbs) < 12:
        sys.exit("Clip trop court (au moins 12 images)")
    keep, diffs = [0], []
    for i in range(1, len(thumbs)):
        d = float(np.abs(thumbs[i] - thumbs[keep[-1]]).mean())
        diffs.append(d)
        if d > a.dup_threshold:
            keep.append(i)
    gaps = np.diff(keep)
    content_fps = info["fps"] * len(keep) / len(thumbs)
    hist = {int(g): int((gaps == g).sum()) for g in np.unique(gaps)}
    print(f"{len(thumbs)} images capturées -> {len(keep)} vraies images du jeu ({content_fps:.2f} i/s). "
          f"Écarts (en images de capture) : {hist}")

    # A held-out triplet (c[2k], c[2k+1], c[2k+2]) is a fair test only when the missing frame
    # sits halfway in time and there is no scene cut.
    triplets = []
    for k in range((len(keep) - 1) // 2):
        i0, i1, i2 = keep[2 * k], keep[2 * k + 1], keep[2 * k + 2]
        cut = float(np.abs(thumbs[i2] - thumbs[i0]).mean())
        reason = "ok"
        if i1 - i0 != i2 - i1:
            reason = "cadence"     # PS5 dropped a frame here: truth is not at t=0.5
        elif cut > a.cut_threshold:
            reason = "scene-cut"   # Veyra resets on cuts, nothing to judge
        triplets.append(dict(k=k, capture=[i0, i1, i2], valid=reason == "ok", reason=reason,
                             change=round(cut, 2)))

    # HUD = pixels that never change over the clip while the image around them moves,
    # and that carry edges (text, minimap). Saved at thumbnail size.
    stack = np.stack([thumbs[i] for i in keep])
    std = stack.std(axis=0)
    mean = stack.mean(axis=0)
    grad = cv2.magnitude(cv2.Sobel(mean, cv2.CV_32F, 1, 0), cv2.Sobel(mean, cv2.CV_32F, 0, 1))
    hud = (std < 1.5) & (cv2.dilate((grad > 40).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    hud = cv2.dilate(hud.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    moving = float((std > 4).mean())
    hud_note = "auto"
    if hud.mean() > 0.2 or moving < 0.3:
        hud[:] = False
        hud_note = "désactivé : la caméra bouge trop peu dans ce clip pour séparer le HUD"
    cv2.imwrite(str(out / "hud_mask.png"), hud.astype(np.uint8) * 255)

    # Pass 2: write the reference (all game frames) and Veyra's input (even game frames).
    rd = Reader(src, start=a.start, duration=a.duration)
    ref = Writer(out / "ref.mkv", rd.w, rd.h, content_fps, "ffv1")
    inp = Writer(out / "input.mp4", rd.w, rd.h, content_fps / 2, "x264")
    keep_set = {c: j for j, c in enumerate(keep)}
    last_even = (len(keep) - 1) // 2 * 2
    for i, f in enumerate(rd):
        j = keep_set.get(i)
        if j is None:
            continue
        ref.write(f)
        if j % 2 == 0 and j <= last_even:
            inp.write(f)
    rd.close(); ref.close(); inp.close()

    manifest = dict(tool_version=TOOL_VERSION, name=a.name, source=str(src.resolve()), start=a.start,
                    duration=a.duration, width=info["w"], height=info["h"], capture_fps=info["fps"],
                    content_fps=round(content_fps, 4), input_fps=round(content_fps / 2, 4),
                    content_frames=len(keep), input_frames=last_even // 2 + 1,
                    capture_index=keep, cadence=hist, dup_threshold=a.dup_threshold,
                    triplets=triplets, head_rect=a.head or DEFAULT_HEAD, hud_rects=[],
                    hud_mask=hud_note, tags=a.tag or [])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    valid = sum(t["valid"] for t in triplets)
    print(f"OK : {out}\n  input.mp4 = {manifest['input_frames']} images à {content_fps / 2:.2f} i/s "
          f"(à exporter avec Veyra en 2X)\n  {valid}/{len(triplets)} images testables "
          f"({len(triplets) - valid} écartées : cadence irrégulière ou changement de plan)\n"
          f"  HUD : {hud_note}. Zone personnage : {manifest['head_rect']} (modifiable dans manifest.json)")


# --------------------------------------------------------------------------- alignment
def align(out_thumbs: list[np.ndarray], in_thumbs: list[np.ndarray]) -> list[int | None]:
    """For each input frame k, the output frame that carries it (or None). Tolerates an
    offset at the start, repeated frames (holds) and dropped frames in the export."""
    def score(a, b):
        return -float(np.abs(a - b).mean())

    # Global offset from the first frames, assuming 2 output frames per input frame.
    best, off = -1e9, 0
    for o in range(0, min(8, len(out_thumbs))):
        n = min(20, (len(out_thumbs) - o) // 2, len(in_thumbs))
        if n < 3:
            continue
        s = np.mean([score(out_thumbs[o + 2 * k], in_thumbs[k]) for k in range(n)])
        if s > best:
            best, off = s, o
    anchors: list[int | None] = []
    prev = off - 2
    for k in range(len(in_thumbs)):
        guess = prev + 2
        cands = [j for j in range(guess - 1, guess + 3) if prev < j < len(out_thumbs)]
        if not cands:
            anchors.append(None); continue
        j = max(cands, key=lambda j: score(out_thumbs[j], in_thumbs[k]))
        if -score(out_thumbs[j], in_thumbs[k]) > 6.0:   # no output frame looks like it
            anchors.append(None); continue
        anchors.append(j); prev = j
    return anchors


# --------------------------------------------------------------------------- score
def summarize(values: list[float]) -> dict:
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return {}
    worst = np.sort(v)[::-1][:max(1, int(math.ceil(v.size * 0.05)))]
    return dict(mean=float(v.mean()), worst5=float(worst.mean()), max=float(v.max()))


def cmd_score(a):
    clip = Path(a.clip)
    man = json.loads((clip / "manifest.json").read_text(encoding="utf-8"))
    out_path = Path(a.output)
    label = a.label or out_path.stem
    res_dir = clip / "results" / label
    res_dir.mkdir(parents=True, exist_ok=True)
    W, H = man["width"], man["height"]
    if a.scale != 1.0:
        W, H = int(round(W * a.scale / 2) * 2), int(round(H * a.scale / 2) * 2)
    oinfo = probe(out_path)
    ratio = oinfo["fps"] / man["input_fps"] if man["input_fps"] else 0
    print(f"Sortie Veyra : {oinfo['w']}x{oinfo['h']} à {oinfo['fps']:.2f} i/s (x{ratio:.2f} de l'entrée)")
    if abs(ratio - 2) > 0.15:
        print("  Attention : la sortie devrait être à 2X l'entrée. Vérifie que la génération d'images 2X était active.")
    if (oinfo["w"], oinfo["h"]) != (man["width"], man["height"]):
        print("  Attention : résolution différente de la source (super-résolution active ?). "
              "La sortie est remise à l'échelle, les scores seront moins justes.")

    # Pass 1: thumbnails for alignment.
    t0 = time.time()
    in_thumbs = [thumb(f) for f in Reader(clip / "input.mp4")]
    out_thumbs = [thumb(f) for f in Reader(out_path, man["width"], man["height"])]
    anchors = align(out_thumbs, in_thumbs)
    found = sum(x is not None for x in anchors)
    print(f"Alignement : {found}/{len(in_thumbs)} vraies images retrouvées dans la sortie ({time.time() - t0:.0f} s)")
    if found < len(in_thumbs) * 0.8:
        sys.exit("Sortie impossible à aligner : vérifier que c'est bien l'export de input.mp4, "
                 "sans super-résolution, débruitage ni filtre couleur.")

    head_rect = man.get("head_rect") or DEFAULT_HEAD
    head = rect_mask(H, W, head_rect)
    hud = cv2.resize(cv2.imread(str(clip / "hud_mask.png"), cv2.IMREAD_GRAYSCALE), (W, H),
                     interpolation=cv2.INTER_NEAREST) > 0 if (clip / "hud_mask.png").exists() else np.zeros((H, W), bool)
    for r in man.get("hud_rects", []):
        hud |= rect_mask(H, W, r)
    head &= ~hud
    flip = Flip(a.ppd * a.scale)
    lp = Lpips() if a.lpips else None

    # Blend baseline does not depend on the Veyra version: cache it in the clip folder.
    cache_path = clip / f"blend_cache_v{TOOL_VERSION}_{W}x{H}_ppd{a.ppd:g}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    tri = [t for t in man["triplets"]][::a.every]
    if a.max_frames:
        tri = tri[:a.max_frames]
    rd_ref, rd_in, rd_out = Reader(clip / "ref.mkv", W, H), Reader(clip / "input.mp4", W, H), Reader(out_path, W, H)
    rows, worst = [], []
    t0 = time.time()
    for n, t in enumerate(tri):
        k = t["k"]
        if not t["valid"]:
            rows.append(dict(k=k, valid=False, reason=t["reason"])); continue
        if k + 1 >= len(anchors) or anchors[k] is None or anchors[k + 1] is None:
            rows.append(dict(k=k, valid=False, reason="non-aligné")); continue
        j0, j1 = anchors[k], anchors[k + 1]
        truth = rd_ref.get(2 * k + 1)
        prev, nxt = rd_in.get(k), rd_in.get(k + 1)
        out_real = rd_out.get(j0)
        if j1 - j0 != 2:
            # Export repeated or dropped something here: no single generated frame.
            rows.append(dict(k=k, valid=False, reason=f"écart sortie {j1 - j0}")); continue
        gen = rd_out.get(j0 + 1)
        if truth is None or prev is None or nxt is None or gen is None:
            break
        row = dict(k=k, valid=t["valid"], reason=t["reason"], out_frame=j0 + 1)
        yg, yt, yp, yn = luma(gen), luma(truth), luma(prev), luma(nxt)
        hold = min(psnr(yg, yp), psnr(yg, yn)) > 48
        row["hold"] = bool(hold)          # Veyra repeated an image instead of generating one
        mmask = motion_mask(prev, nxt, hud)
        smask = static_mask(prev, nxt, truth)
        hs = head & smask
        fm = flip.map(truth, gen)
        row.update(psnr=psnr(yg, yt), ssim=ssim(yg, yt), flip_mean=float(fm.mean()),
                   flip_p99=float(np.percentile(fm, 99)),
                   head_flip=float(fm[head].mean()), head_psnr=psnr(yg, yt, head),
                   hud_flip=float(fm[hud].mean()) if hud.any() else None,
                   motion_flip=float(fm[mmask].mean()) if mmask.mean() > 0.002 else None,
                   motion_area=float(mmask.mean()),
                   static_flip=float(fm[smask].mean()) if smask.mean() > 0.002 else None,
                   head_static_flip=float(fm[hs].mean()) if hs.sum() > 200 else None,
                   floor_psnr=psnr(luma(out_real), yp))   # codec/settings loss on real frames
        if n % 5 == 0:
            # Export encoder noise, measured on a real frame of the output against Veyra's input.
            fr = flip.map(prev, out_real)
            row["floor_flip"] = float(fr.mean())
            row["floor_head_static_flip"] = float(fr[hs].mean()) if hs.sum() > 200 else None
        if lp:
            row["lpips"] = lp(truth, gen)
        key = str(k)
        if key not in cache:
            blend = ((prev.astype(np.uint16) + nxt) // 2).astype(np.uint8)
            bm = flip.map(truth, blend)
            cache[key] = dict(psnr=psnr(luma(blend), yt), ssim=ssim(luma(blend), yt),
                              flip_mean=float(bm.mean()), head_flip=float(bm[head].mean()),
                              motion_flip=float(bm[mmask].mean()) if mmask.mean() > 0.002 else None,
                              static_flip=float(bm[smask].mean()) if smask.mean() > 0.002 else None,
                              head_static_flip=float(bm[hs].mean()) if hs.sum() > 200 else None,
                              repeat_psnr=psnr(yp, yt))
        row["blend"] = cache[key]
        rows.append(row)
        if row["valid"]:
            worst.append((row["head_flip"] + row["flip_mean"] + (row["head_static_flip"] or 0), k, truth.copy(), gen.copy(), fm.copy()))
            worst = sorted(worst, key=lambda x: -x[0])[:a.worst]
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  {n + 1}/{len(tri)} images ({el:.0f} s, reste ~{el / (n + 1) * (len(tri) - n - 1):.0f} s)", flush=True)
    for r in (rd_ref, rd_in, rd_out):
        r.close()
    cache_path.write_text(json.dumps(cache))

    ok = [r for r in rows if r.get("valid") and "psnr" in r]
    if not ok:
        sys.exit("Aucune image testable : clip trop court ou export non aligné.")
    col = lambda key, rs=ok: [r[key] for r in rs if r.get(key) is not None]
    bcol = lambda key: [r["blend"][key] for r in ok if r["blend"].get(key) is not None]
    agg = dict(
        full=dict(psnr=float(np.mean(col("psnr"))), ssim=float(np.mean(col("ssim"))),
                  flip_mean=float(np.mean(col("flip_mean"))),
                  flip_p99_worst5=summarize(col("flip_p99"))["worst5"]),
        head=dict(flip_mean=float(np.mean(col("head_flip"))), flip_worst5=summarize(col("head_flip"))["worst5"],
                  psnr=float(np.mean(col("head_psnr"))),
                  static_flip_mean=float(np.mean(col("head_static_flip"))) if col("head_static_flip") else None,
                  static_flip_worst5=summarize(col("head_static_flip")).get("worst5")),
        static=dict(flip_mean=float(np.mean(col("static_flip"))) if col("static_flip") else None),
        motion=dict(flip_mean=float(np.mean(col("motion_flip"))) if col("motion_flip") else None),
        hud=dict(flip_mean=float(np.mean(col("hud_flip"))) if col("hud_flip") else None),
        blend=dict(psnr=float(np.mean(bcol("psnr"))), ssim=float(np.mean(bcol("ssim"))),
                   flip_mean=float(np.mean(bcol("flip_mean"))), head_flip=float(np.mean(bcol("head_flip"))),
                   motion_flip=float(np.mean(bcol("motion_flip"))) if bcol("motion_flip") else None,
                   static_flip=float(np.mean(bcol("static_flip"))) if bcol("static_flip") else None,
                   head_static_flip=float(np.mean(bcol("head_static_flip"))) if bcol("head_static_flip") else None,
                   repeat_psnr=float(np.mean(bcol("repeat_psnr")))),
        holds=int(sum(r.get("hold", False) for r in ok)),
        codec_floor_psnr=float(np.mean(col("floor_psnr"))),
    )
    if lp:
        agg["full"]["lpips"] = float(np.mean(col("lpips")))
    # Share of frames where the still part of the character is damaged: clearly worse than a
    # plain blend, which is near perfect on still pixels (the "vanishing head" symptom).
    # The export encoder adds its own noise (measured on real frames), the blend has none.
    floor_hs = float(np.mean(col("floor_head_static_flip"))) if col("floor_head_static_flip") else 0.0
    agg["codec_floor_flip"] = float(np.mean(col("floor_flip"))) if col("floor_flip") else None
    hs_rows = [r for r in ok if r.get("head_static_flip") is not None and r["blend"].get("head_static_flip") is not None]
    agg["head"]["damaged_pct"] = 100.0 * float(np.mean(
        [r["head_static_flip"] > max(r["blend"]["head_static_flip"], floor_hs) * 1.5 + 0.02 for r in hs_rows])) if hs_rows else None

    for i, (_, k, truth, gen, fm) in enumerate(worst):
        save_worst(res_dir / f"worst_{i + 1}_k{k}.png", truth, gen, fm, head)
    result = dict(tool_version=TOOL_VERSION, clip=man["name"], tags=man.get("tags", []), label=label,
                  output=str(out_path.resolve()), resolution=[W, H], ppd=a.ppd * a.scale,
                  samples=len(rows), valid=len(ok), head_rect=head_rect,
                  hud_pixels_pct=round(100 * float(hud.mean()), 2), metrics=agg, frames=rows,
                  created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (res_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print_result(result)
    if agg["codec_floor_psnr"] < 34:
        print(f"\n  Attention : les vraies images de la sortie s'écartent de l'entrée "
              f"(PSNR {agg['codec_floor_psnr']:.1f} dB). Un réglage modifie l'image (couleur, SR, débruitage, "
              f"netteté RCAS) ou le débit d'export est trop bas : les scores mélangent ces effets.")
    print(f"\nRésultat : {res_dir / 'result.json'}")
    return result


def save_worst(path: Path, truth, gen, fm, head):
    """Real frame | generated frame | FLIP map (magma-like), character zone outlined."""
    heat = cv2.applyColorMap(np.clip(fm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    tiles = [cv2.cvtColor(truth, cv2.COLOR_RGB2BGR), cv2.cvtColor(gen, cv2.COLOR_RGB2BGR), heat]
    ys, xs = np.where(head)
    for t, name in zip(tiles, ["vraie image", "image generee", "erreur FLIP"]):
        if ys.size:
            cv2.rectangle(t, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
        cv2.putText(t, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    w = 1280
    tiles = [cv2.resize(t, (w, round(t.shape[0] * w / t.shape[1])), interpolation=cv2.INTER_AREA) for t in tiles]
    cv2.imwrite(str(path), np.vstack(tiles))


def get(d: dict, path: str):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def print_result(r: dict):
    m = r["metrics"]
    b = m["blend"]
    print(f"\n== {r['clip']} / {r['label']} : {r['valid']} images générées testées ==")
    print(f"  PSNR {m['full']['psnr']:.2f} dB (mélange 50/50 : {b['psnr']:.2f}, répétition : {b['repeat_psnr']:.2f})")
    print(f"  SSIM {m['full']['ssim']:.4f} (mélange : {b['ssim']:.4f})")
    print(f"  FLIP moyen {m['full']['flip_mean']:.4f} (mélange : {b['flip_mean']:.4f}) | 5 % pires p99 {m['full']['flip_p99_worst5']:.3f}")
    print(f"  Zone personnage : FLIP {m['head']['flip_mean']:.4f} (mélange : {b['head_flip']:.4f})")
    if m["head"]["static_flip_mean"] is not None:
        print(f"  Personnage immobile (tête qui disparaît) : FLIP {m['head']['static_flip_mean']:.4f} "
              f"(mélange : {b['head_static_flip']:.4f}), abîmé sur {m['head']['damaged_pct']:.0f} % des images")
    if m["static"]["flip_mean"] is not None:
        print(f"  Toutes zones immobiles : FLIP {m['static']['flip_mean']:.4f} (mélange : {b['static_flip']:.4f})")
    if m["motion"]["flip_mean"] is not None:
        print(f"  Bords en mouvement (halos) : FLIP {m['motion']['flip_mean']:.4f} (mélange : {b['motion_flip']:.4f})")
    if m["hud"]["flip_mean"] is not None:
        print(f"  HUD : FLIP {m['hud']['flip_mean']:.4f}")
    print(f"  Images répétées au lieu d'être générées : {m['holds']} | perte codec sur vraies images : PSNR {m['codec_floor_psnr']:.1f} dB")
    if "lpips" in m["full"]:
        print(f"  LPIPS {m['full']['lpips']:.4f}")


# --------------------------------------------------------------------------- gate
def compare(result: dict, base: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for key, higher, tol, kind in GATE_KEYS:
        cur, ref = get(result["metrics"], key), get(base["metrics"], key)
        if cur is None or ref is None:
            continue
        delta = cur - ref
        limit = tol if kind == "abs" else tol * abs(ref)
        worse = (-delta if higher else delta) > limit
        better = (delta if higher else -delta) > limit
        ok &= not worse
        mark = "RÉGRESSION" if worse else ("mieux" if better else "stable")
        pct = f" ({delta / ref * 100:+.1f} %)" if kind == "rel" and ref else ""
        lines.append(f"| {LABELS[key]} | {ref:.4f} | {cur:.4f} | {delta:+.4f}{pct} | {mark} |")
    return ok, lines


def cmd_gate(a):
    result = json.loads(Path(a.result).read_text(encoding="utf-8"))
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    if result["clip"] != base["clip"] or result["resolution"] != base["resolution"]:
        sys.exit(f"Comparaison impossible : clip {result['clip']} {result['resolution']} contre "
                 f"{base['clip']} {base['resolution']} (il faut le même clip, à la même échelle)")
    ok, lines = compare(result, base)
    print(f"{result['clip']} : {base['label']} -> {result['label']}")
    print("| Mesure | Référence | Nouvelle | Écart | Verdict |\n|---|---|---|---|---|")
    print("\n".join(lines))
    print("\nVERDICT :", "OK, pas de régression" if ok else "RÉGRESSION de qualité")
    return 0 if ok else 1


def cmd_suite(a):
    root = Path(a.clips)
    clips = sorted(p for p in root.iterdir() if (p / "manifest.json").exists())
    summary = [f"# Qualité Veyra : {a.label} ({time.strftime('%Y-%m-%d %H:%M')})", "",
               "Méthode : une vraie image sur deux retirée, l'image générée par Veyra comparée à la vraie. "
               "FLIP : plus bas = mieux. PSNR/SSIM : plus haut = mieux.", ""]
    failed, missing = [], []
    for clip in clips:
        out = clip / a.output_name
        if not out.exists():
            missing.append(clip.name); continue
        ns = argparse.Namespace(clip=str(clip), output=str(out), label=a.label, scale=a.scale, ppd=a.ppd,
                                every=a.every, max_frames=a.max_frames, worst=a.worst, lpips=a.lpips)
        r = cmd_score(ns)
        m = r["metrics"]
        summary += [f"## {clip.name} ({', '.join(r['tags']) or 'sans tag'})", "",
                    f"PSNR {m['full']['psnr']:.2f} dB, SSIM {m['full']['ssim']:.4f}, FLIP {m['full']['flip_mean']:.4f} "
                    f"(mélange 50/50 : {m['blend']['flip_mean']:.4f}), personnage {m['head']['flip_mean']:.4f}, "
                    f"personnage immobile abîmé sur {m['head']['damaged_pct'] or 0:.0f} % des images, "
                    f"{m['holds']} images répétées.", ""]
        base_path = clip / "baseline.json"
        if base_path.exists() and not a.update_baseline:
            base = json.loads(base_path.read_text(encoding="utf-8"))
            ok, lines = compare(r, base)
            summary += [f"Comparé à la référence **{base['label']}** : " + ("OK" if ok else "**RÉGRESSION**"), "",
                        "| Mesure | Référence | Nouvelle | Écart | Verdict |", "|---|---|---|---|---|", *lines, ""]
            if not ok:
                failed.append(clip.name)
        else:
            shutil.copyfile(clip / "results" / a.label / "result.json", base_path)
            summary += [f"Enregistré comme nouvelle référence ({base_path.name}).", ""]
    if missing:
        summary += [f"Clips sans fichier `{a.output_name}` (pas encore exportés) : {', '.join(missing)}", ""]
    summary += ["**VERDICT : " + (f"RÉGRESSION sur {', '.join(failed)}" if failed else "OK") + "**"]
    path = root / f"summary_{a.label}.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    print("\n" + "\n".join(summary) + f"\n\nRésumé : {path}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- selftest
def synth_scene(t: float, w: int, h: int, hud=True, char=True) -> np.ndarray:
    """Textured background panning right + a 'character' that stays still in the middle
    (third-person camera orbit) + a static HUD block. t in frame units at 60 Hz."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx + 6.0 * t
    bg = (128 + 60 * np.sin(x / 9.0) * np.cos(yy / 13.0) + 40 * np.sin((x + yy) / 31.0))
    img = np.stack([bg, bg * 0.8 + 30, 255 - bg * 0.9], -1)
    cx, cy = w * 0.5, h * 0.6
    body = ((xx - cx) / (w * 0.06)) ** 2 + ((yy - cy) / (h * 0.22)) ** 2 < 1
    if char:
        img[body] = [200, 60, 50]
    if hud:
        img[int(h * 0.82):int(h * 0.95), int(w * 0.04):int(w * 0.2)] = [240, 240, 240]
        img[int(h * 0.86):int(h * 0.91), int(w * 0.06):int(w * 0.18)] = [20, 20, 20]
    return np.clip(img, 0, 255).astype(np.uint8)


def cmd_selftest(a):
    """End to end on synthetic video: prepare a 30 fps 'game' captured at 60 Hz, then score
    three fake 'exports': perfect (true middle frame), blend and a broken one where the
    character disappears. The tool must rank them and the gate must flag the broken one."""
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    w, h, n = 320, 180, 96
    cap = Writer(work / "capture.mp4", w, h, 60, "x264")
    for i in range(n):
        cap.write(synth_scene(float(i // 2 * 2), w, h))   # each 30 fps game frame shown twice
    cap.close()
    ns = argparse.Namespace(capture=str(work / "capture.mp4"), out=str(work / "clips"), name="synth", force=True,
                            start=0.0, duration=0.0, dup_threshold=0.6, cut_threshold=45.0, head=None,
                            tag=["synthetique"])
    cmd_prepare(ns)
    clip = work / "clips" / "synth"
    man = json.loads((clip / "manifest.json").read_text())
    assert abs(man["content_fps"] - 30) < 0.5, man["content_fps"]
    frames = [synth_scene(float(i * 2), w, h) for i in range(man["content_frames"])]
    results = {}
    for kind in ("perfect", "blend", "broken"):
        wr = Writer(clip / f"{kind}.mp4", w, h, man["input_fps"] * 2, "x264")
        wr.write(frames[0])  # one leading extra frame: exercises the alignment offset
        for k in range(man["input_frames"]):
            a0 = frames[2 * k]
            wr.write(a0)
            if k + 1 < man["input_frames"]:
                if kind == "perfect":
                    g = frames[2 * k + 1]
                elif kind == "blend":
                    g = ((a0.astype(np.uint16) + frames[2 * k + 2]) // 2).astype(np.uint8)
                else:
                    g = synth_scene(float((2 * k + 1) * 2), w, h, char=False)  # the head "vanishes"
                wr.write(g)
        wr.close()
        sa = argparse.Namespace(clip=str(clip), output=str(clip / f"{kind}.mp4"), label=kind, scale=1.0, ppd=67.0,
                                every=1, max_frames=0, worst=2, lpips=False)
        results[kind] = cmd_score(sa)
    p, b, x = (results[k]["metrics"] for k in ("perfect", "blend", "broken"))
    assert p["full"]["flip_mean"] < b["full"]["flip_mean"], "parfait doit battre le mélange"
    assert p["head"]["flip_mean"] < x["head"]["flip_mean"], "tête cassée doit être détectée"
    assert x["head"]["damaged_pct"] > 50, x["head"]["damaged_pct"]
    assert p["head"]["damaged_pct"] < 10, p["head"]["damaged_pct"]
    assert x["head"]["static_flip_mean"] > 3 * p["head"]["static_flip_mean"]
    assert p["full"]["psnr"] > b["full"]["psnr"]
    assert results["perfect"]["valid"] >= 10
    ok, _ = compare(results["broken"], results["perfect"])
    assert not ok, "le gate doit signaler la régression"
    ok, _ = compare(results["perfect"], results["perfect"])
    assert ok
    print("\nSELFTEST OK")
    return 0


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="capture Elgato -> clip de test")
    s.add_argument("capture"); s.add_argument("--name", required=True)
    s.add_argument("--out", default="clips"); s.add_argument("--force", action="store_true")
    s.add_argument("--start", type=float, default=0.0, help="début en secondes")
    s.add_argument("--duration", type=float, default=12.0, help="durée en secondes (0 = tout)")
    s.add_argument("--dup-threshold", type=float, default=0.6,
                   help="écart moyen (0-255, basse résolution) sous lequel une image est un doublon")
    s.add_argument("--cut-threshold", type=float, default=45.0)
    s.add_argument("--head", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="zone personnage en fractions de l'image")
    s.add_argument("--tag", action="append", help="ex. tete, hud, feuillage, nuit")

    def scoring(s):
        s.add_argument("--scale", type=float, default=1.0, help="0.5 = deux fois plus rapide, moins précis")
        s.add_argument("--ppd", type=float, default=67.0,
                       help="pixels par degré de vision pour FLIP (67 = défaut NVIDIA ; ~45 pour un 27\" 1440p à 60 cm)")
        s.add_argument("--every", type=int, default=1, help="ne tester qu'une image générée sur N")
        s.add_argument("--max-frames", type=int, default=0)
        s.add_argument("--worst", type=int, default=3, help="nombre d'images pires enregistrées en PNG")
        s.add_argument("--lpips", action="store_true", help="ajoute LPIPS (demande torch + lpips)")

    s = sub.add_parser("score", help="note un export Veyra d'un clip"); s.add_argument("clip"); s.add_argument("output")
    s.add_argument("--label"); scoring(s)
    s = sub.add_parser("gate", help="compare à la référence"); s.add_argument("result"); s.add_argument("baseline")
    s = sub.add_parser("suite", help="tous les clips d'un dossier")
    s.add_argument("clips"); s.add_argument("--label", required=True)
    s.add_argument("--output-name", default="veyra_out.mp4")
    s.add_argument("--update-baseline", action="store_true"); scoring(s)
    s = sub.add_parser("selftest"); s.add_argument("--work", default="quality-selftest")

    a = p.parse_args(argv)
    if a.cmd == "prepare":
        return cmd_prepare(a) or 0
    if a.cmd == "score":
        cmd_score(a); return 0
    if a.cmd == "gate":
        return cmd_gate(a)
    if a.cmd == "suite":
        return cmd_suite(a)
    return cmd_selftest(a)


if __name__ == "__main__":
    sys.exit(main())
