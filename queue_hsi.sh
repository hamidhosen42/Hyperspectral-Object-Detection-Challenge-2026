#!/bin/zsh
# After the RGB run has been auto-submitted: train YOLO11m on 16-band input, then validate,
# ensemble with the RGB model and submit whichever is best on the hold-out val.
cd "$(dirname "$0")"
LOG=logs/queue_hsi.log
while pgrep -f auto_submit.sh >/dev/null; do sleep 60; done
echo "[$(date)] RGB run done, starting HSI yolo11m" >> $LOG

python3 train.py --kind hsi --model yolo11m.pt --imgsz 1024 --epochs 40 --batch 6 --name hsi_m1024 \
  --extra scale=0.7 > logs/hsi_m1024.log 2>&1 &
TP=$!; caffeinate -i -s -w $TP & wait $TP
echo "[$(date)] HSI training exited ($?)" >> $LOG

python3 predict.py --weights runs/hsi_m1024/weights/best.pt --kind hsi --imgsz 1024 --split val --out val_pred_hsi_m.csv >> $LOG 2>&1
python3 predict.py --weights runs/rgb_s1024/weights/best.pt --kind rgb --imgsz 1024 --split val --out val_pred_rgb_final.csv >> $LOG 2>&1
python3 ensemble.py --preds val_pred_rgb_final.csv val_pred_hsi_m.csv --split val --out val_pred_fused.csv >> $LOG 2>&1
score() { python3 eval_local.py --pred $1 2>/dev/null | grep -E "0.50:0.95 \| area=   all" | grep -oE "[0-9.]+$"; }
S_RGB=$(score val_pred_rgb_final.csv); S_HSI=$(score val_pred_hsi_m.csv); S_FUSE=$(score val_pred_fused.csv)
echo "[$(date)] val mAP50-95: rgb=$S_RGB hsi=$S_HSI fused=$S_FUSE" >> $LOG

BEST=rgb; BS=$S_RGB
[ "$(echo "$S_HSI > $BS" | bc)" = 1 ] && { BEST=hsi; BS=$S_HSI; }
[ "$(echo "$S_FUSE > $BS" | bc)" = 1 ] && { BEST=fused; BS=$S_FUSE; }
echo "[$(date)] best on val: $BEST ($BS)" >> $LOG
if [ "$BEST" = rgb ]; then echo "RGB already submitted; nothing new to submit" >> $LOG; exit 0; fi

python3 predict.py --weights runs/hsi_m1024/weights/best.pt --kind hsi --imgsz 1024 --out submission_hsi_m.csv >> $LOG 2>&1
if [ "$BEST" = fused ]; then
  python3 ensemble.py --preds submission_rgb_final.csv submission_hsi_m.csv --out submission_fused.csv >> $LOG 2>&1; F=submission_fused.csv
else F=submission_hsi_m.csv; fi
kaggle competitions submit -c hyperspectral-object-detection-challenge-2026 -f $F -m "$BEST: YOLO11m 16-band (+WBF with RGB s) val mAP50-95 $BS" >> $LOG 2>&1
echo "[$(date)] submitted $F" >> $LOG
