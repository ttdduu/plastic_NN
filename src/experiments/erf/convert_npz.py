#!/usr/bin/env python3
import os, json, time
import numpy as np

def convert_npz_to_packed_per_channel(npz_path, out_dir, channels=None):
    os.makedirs(out_dir, exist_ok=True)
    with np.load(npz_path) as z:
        all_keys = [k for k in z.files if k.startswith("patch_")]
        if channels is None:
            channels = sorted({int(k.split("_")[1]) for k in all_keys})
        base = os.path.splitext(os.path.basename(npz_path))[0]

        for ch in channels:
            ch_keys = [k for k in all_keys if k.startswith(f"patch_{ch}_")]
            if not ch_keys:
                print(f"Channel {ch}: no patches", flush=True)
                continue

            # Sort (optional): y,x ascending for reproducibility
            ch_keys.sort(key=lambda k: (int(k.split("_")[2]), int(k.split("_")[3])))
            total_keys = len(ch_keys)
            print(f"Channel {ch}: {total_keys} patches – indexing shapes and coords...", flush=True)
            t0 = time.perf_counter()

            # Pre-compute total elements to allocate contiguous buffer
            shapes = []
            coords = []
            total_elems = 0
            for idx_k, k in enumerate(ch_keys, start=1):
                _, _, ys, xs = k.split("_")
                arr = z[k]
                h, w = arr.shape
                shapes.append((h, w))
                coords.append((int(ys), int(xs)))
                total_elems += h * w
                if idx_k % 5000 == 0 or idx_k == total_keys:
                    dt = time.perf_counter() - t0
                    rate = idx_k / dt if dt > 0 else 0.0
                    print(f"  Indexed {idx_k}/{total_keys} (elapsed {dt:.1f}s, {rate:.0f} patches/s)", flush=True)

            data = np.empty(total_elems, dtype=np.float32)
            starts = np.empty(len(ch_keys), dtype=np.int64)
            lengths = np.empty(len(ch_keys), dtype=np.int64)
            hs = np.empty(len(ch_keys), dtype=np.int32)
            ws = np.empty(len(ch_keys), dtype=np.int32)
            ys = np.empty(len(ch_keys), dtype=np.int32)
            xs = np.empty(len(ch_keys), dtype=np.int32)

            cursor = 0
            print(f"Channel {ch}: packing data buffer of {total_elems} floats...", flush=True)
            t1 = time.perf_counter()
            for i, k in enumerate(ch_keys, start=1):
                arr = z[k].astype(np.float32, copy=False)
                h, w = arr.shape
                n = h * w
                data[cursor:cursor+n] = arr.ravel()
                starts[i] = cursor
                lengths[i] = n
                hs[i] = h; ws[i] = w
                y, x = coords[i]
                ys[i] = y; xs[i] = x
                cursor += n
                if i % 5000 == 0 or i == total_keys:
                    dt = time.perf_counter() - t1
                    rate = i / dt if dt > 0 else 0.0
                    print(f"  Packed {i}/{total_keys} (elapsed {dt:.1f}s, {rate:.0f} patches/s)", flush=True)

            out_data = os.path.join(out_dir, f"{base}_ch{ch}_data.npy")
            out_index = os.path.join(out_dir, f"{base}_ch{ch}_index.npz")
            print(f"Channel {ch}: writing outputs...", flush=True)
            np.save(out_data, data)
            np.savez_compressed(out_index,
                                start=starts, length=lengths,
                                h=hs, w=ws, y=ys, x=xs)

            meta = {
                "source_npz": npz_path,
                "channel": ch,
                "num_patches": int(len(ch_keys)),
                "total_elems": int(total_elems),
                "unique_shapes": sorted({tuple(s) for s in shapes}),
            }
            with open(out_data.replace("_data.npy", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Channel {ch}: wrote {out_data} and {out_index} ({len(ch_keys)} patches)", flush=True)

if __name__ == "__main__":
    npz_path = "/home/tomasdu/repos/trained_models/02yo20f3/gabor_tuning_model_stages_0_0_dwconv_results.npz"
    out_dir = os.path.join(os.path.dirname(npz_path), "per_channel_packed")
    convert_npz_to_packed_per_channel(npz_path, out_dir, channels=None)
