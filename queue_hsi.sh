#!/bin/zsh
# After the RGB run has been auto-submitted: train YOLO11m on 16-band input and score it on the
# hold-out val. No ensembling (competition rule: single model only). Submission is decided manually.
cd "$(dirname "$0")"
LOG=logs/queue_hsi.log
while pgrep -f auto_submit.sh >/dev/null; do sleep 60; done
echo "[$(date)] RGB run done, starting HSI yolo11m" >> $LOG

python3 train.py --kind hsi --model yolo11m.pt --imgsz 1024 --epochs 40 --batch 6 --name hsi_m1024 \
  --extra scale=0.7 > logs/hsi_m1024.log 2>&1 &
TP=$!; caffeinate -i -s -w $TP & wait $TP
echo "[$(date)] HSI training exited ($?)" >> $LOG

python3 predict.py --weights runs/hsi_m1024/weights/best.pt --kind hsi --imgsz 1024 --split val --out val_pred_hsi_m.csv >> $LOG 2>&1
score() { python3 eval_local.py --pred $1 2>/dev/null | grep -E "0.50:0.95 \| area=   all" | grep -oE "[0-9.]+$"; }
echo "[$(date)] val mAP50-95: hsi_m=$(score val_pred_hsi_m.csv)  (rgb ep40 = 0.693)" >> $LOG
