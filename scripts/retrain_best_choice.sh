# Auto running for retraining best choice searched by the random search

if [ ! -d "./logdir" ]; then
  mkdir ./logdir
fi

LogName=log_spos_c10_retrain_best_choice
CUDA_VISIBLE_DEVICES=0 nohup python -u retrain_best_choice.py --cutout --auto_aug \
  > logdir/${LogName} 2>&1 &

sleep 3s

tail -f logdir/${LogName}