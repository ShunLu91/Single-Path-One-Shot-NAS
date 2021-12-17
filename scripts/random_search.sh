# Auto running for random search using SPOS

if [ ! -d "./logdir" ]; then
  mkdir ./logdir
fi

LogName=log_spos_c10_random_search
CUDA_VISIBLE_DEVICES=0 nohup python -u random_search.py > logdir/${LogName} 2>&1 &

sleep 3s

tail -f logdir/${LogName}