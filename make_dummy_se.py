#!/usr/bin/env python
import numpy as np, soundfile as sf, argparse, pathlib, random, math, os

def sine_sweep(length, sr, f0=300, f1=3000):
    t = np.linspace(0, length, int(sr*length), False)
    k = (f1/f0)**(1/len(t))
    phase = 2*np.pi*f0*(k**np.arange(len(t))-1)/np.log(k)
    return np.sin(phase)

def white_noise(length, sr):
    return np.random.randn(int(sr*length))

def write_pair(idx, split, out_root, sr=16000, dur=2.0):
    clean = sine_sweep(dur, sr)
    rms = np.sqrt(np.mean(clean**2))
    noisy = clean + 0.1 * rms * white_noise(dur, sr)
    for kind, sig in [('clean', clean), ('noisy', noisy)]:
        fn = f'sine_{idx:04d}.wav'
        d = pathlib.Path(out_root, split, kind)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / fn, sig, sr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_files", type=int, default=30,
                    help="total files; 80% train, 10% valid, 10% test")
    args = ap.parse_args()

    splits = (["train"]*math.ceil(.8*args.n_files) +
              ["valid"]*math.ceil(.1*args.n_files) +
              ["test"]*math.ceil(.1*args.n_files))
    random.seed(0); random.shuffle(splits)

    for i, split in enumerate(splits):
        write_pair(i, split, args.out_dir)
    print(f"✓ Saved {args.n_files} clean/noisy pairs to {args.out_dir}")
