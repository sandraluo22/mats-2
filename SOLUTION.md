# Solution: Get Dataset Working

## The Problem
- `/root` has only 502MB free (98% full)
- The processed dataset cache is missing
- Cache can't be moved because `/root` is full

## The Solution

**Delete the cache and download fresh to `/workspace`:**

```bash
# 1. Delete the old cache to free up space
rm -rf /root/.cache/huggingface

# 2. Run with environment variables pointing to /workspace
cd /root/mats-2
HF_HOME=/workspace/.cache/huggingface \
HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
TMPDIR=/workspace/tmp \
python3 download_10_problems.py
```

This will:
- Free up 19GB on /root
- Download the dataset fresh to /workspace (which has plenty of space)
- Process the dataset and cache it in /workspace
- Work because /workspace has 355TB free

## Alternative: If you want to keep the raw cache

If the download takes too long and you want to try keeping the raw cache, you could try using `rsync` or `cp` instead of `mv`, but that still requires temporary space.

The simplest solution is to delete and re-download to /workspace.

