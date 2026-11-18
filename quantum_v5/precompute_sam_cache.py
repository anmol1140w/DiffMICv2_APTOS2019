"""Precompute EfficientSAM image embeddings for the entire APTOS-2019 dataset.
Extracts 256-dimensional pooled foundation embeddings for all 3,662 images once.
Saves to quantum_v5/outputs/efficient_sam_aptos_cache.pt (~3.7 MB).
Reduces training epoch time from 7 minutes -> ~20 seconds!
"""
import os, sys, time, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aptos_research.data import load_split_manifest, AptosImageDataset
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f" >>> PRECOMPUTING EFFICIENT-SAM FOUNDATION EMBEDDINGS")
    print(f" >>> Device: {device}")
    print(f"=======================================================\n")

    output_path = PROJECT_ROOT / 'quantum_v5/outputs/efficient_sam_aptos_cache.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[*] Cache file already exists at {output_path} ({output_path.stat().st_size / 1024:.1f} KB).")
        print("    Loading existing cache to verify...")
        cache = torch.load(output_path, map_location='cpu', weights_only=False)
        print(f"    Verified: {len(cache)} image embeddings cached!")
        return

    # Load EfficientSAM ViT-S
    sam_ckpt = PROJECT_ROOT / 'EfficientSAM/weights/efficient_sam_vits.pt'
    sam = build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=str(sam_ckpt)
    ).to(device)
    for p in sam.parameters():
        p.requires_grad = False
    sam.eval()

    # Load splits manifest
    manifest_path = PROJECT_ROOT / 'outputs/aptos_splits.json'
    manifest = load_split_manifest(manifest_path)

    all_records = []
    for split_name in ['train', 'validation', 'test']:
        all_records.extend(manifest['splits'][split_name])

    dataset = AptosImageDataset(all_records, 'train', size=224, training=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    cache = {}
    t0 = time.time()
    print(f"--> Extracting embeddings for {len(dataset)} retinal images (batch size 16)...")

    with torch.no_grad():
        for b_idx, (imgs, labels, img_ids, splits) in enumerate(loader):
            imgs = imgs.to(device)
            # Process in chunks of 8 to avoid MIG allocator issues
            embs = []
            for chunk in imgs.split(8):
                embs.append(sam.get_image_embeddings(chunk))
            full_emb = torch.cat(embs, dim=0) # (B, 256, 64, 64)
            pooled = F.adaptive_avg_pool2d(full_emb, (1, 1)).flatten(1).cpu() # (B, 256)
            
            for i, img_id in enumerate(img_ids):
                cache[img_id] = pooled[i]

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
                elapsed = time.time() - t0
                pct = (b_idx + 1) / len(loader) * 100.0
                rate = (b_idx + 1) * 16 / elapsed
                print(f"    Progress: [{b_idx+1:03d}/{len(loader):03d}] ({pct:.1f}%) - {rate:.1f} imgs/s - Elapsed: {elapsed:.1f}s")

    torch.save(cache, output_path)
    print(f"\n[*] Precomputation COMPLETE! Saved {len(cache)} embeddings to {output_path} ({output_path.stat().st_size / 1024:.1f} KB).")
    print(f"    Total extraction time: {time.time() - t0:.1f}s\n")

if __name__ == '__main__':
    main()
