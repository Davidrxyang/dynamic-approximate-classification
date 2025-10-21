#!/usr/bin/env python3
import argparse, time, torch
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

def parse_args():
    p = argparse.ArgumentParser(description="CPU test for MobileNetV3-Small (ImageNet-1k).")
    p.add_argument("--image", type=str, required=False, help="Path to an image file (any common format).")
    p.add_argument("--runs", type=int, default=100, help="Timed runs for latency avg.")
    p.add_argument("--warmup", type=int, default=20, help="Warmup runs (not timed).")
    p.add_argument("--num-threads", type=int, default=4, help="torch.set_num_threads for CPU.")
    p.add_argument("--img-size", type=int, default=224, help="Resize/crop target.")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    # Load pretrained weights and model (downloads once, then cached)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).eval()  # CPU by default

    # Build preprocessing matching the weights (resize, crop, normalize)
    # Using the weights’ canonical transform keeps accuracy correct.
    tform = weights.transforms(crop_size=args.img_size)

    # Prepare input
    if args.image:
        img = Image.open(args.image).convert("RGB")
        x = tform(img).unsqueeze(0)  # [1,3,H,W]
    else:
        # Fallback: random tensor if no image is provided (for quick smoke test)
        x = torch.randn(1, 3, args.img_size, args.img_size)
        # Still normalize like real images
        norm_only = transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"])
        x = norm_only(x)

    # Warm-up (JIT / kernel caches / allocator warming)
    for _ in range(args.warmup):
        _ = model(x)

    # Timed runs (latency)
    start = time.perf_counter()
    for _ in range(args.runs):
        out = model(x)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / args.runs) * 1000.0

    # Top-5 predictions (only meaningful if a real image is provided)
    probs = torch.softmax(out, dim=1)[0]
    top5 = torch.topk(probs, k=5)
    ids = top5.indices.tolist()
    vals = top5.values.tolist()
    labels = weights.meta.get("categories", [str(i) for i in range(out.shape[1])])

    print("\n=== MobileNetV3-Small (CPU) ===")
    print(f"Threads: {args.num_threads} | Image size: {args.img_size} | Runs: {args.runs} | Warmup: {args.warmup}")
    print(f"Average latency (batch=1): {avg_ms:.2f} ms")
    print("\nTop-5:")
    for i, (cls_id, p) in enumerate(zip(ids, vals), 1):
        print(f"{i}. {labels[cls_id]} — {p*100:.2f}%")

if __name__ == "__main__":
    main()
