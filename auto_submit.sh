#!/bin/zsh
# Wait for the rgb_s1024 training run to finish, then predict on test and submit to Kaggle.
cd "$(dirname "$0")"
while pgrep -f "train.py --kind rgb" >/dev/null; do sleep 60; done
EP=$(tail -n1 runs/rgb_s1024/results.csv | cut -d, -f1)
VAL=$(sort -t, -k9 -g runs/rgb_s1024/results.csv | tail -1 | cut -d, -f9)
echo "[$(date)] training finished at epoch $EP, best val mAP50-95=$VAL"
python3 predict.py --weights runs/rgb_s1024/weights/best.pt --kind rgb --imgsz 1024 --out submission_rgb_final.csv 2>&1 | grep -E "^wrote|Traceback|Error"
kaggle competitions submit -c hyperspectral-object-detection-challenge-2026 -f submission_rgb_final.csv \
  -m "YOLO11s pseudo-RGB imgsz1024, full 50-epoch run best.pt (epoch $EP reached), val mAP50-95 $VAL" 2>&1 | tail -2
echo "[$(date)] submitted"
