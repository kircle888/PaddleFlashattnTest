casenums=$(python get_launch_config.py 0)
gpu=$(python get_launch_config.py 1)
log_file=$(python get_launch_config.py 2)
for i in $casenums; do
    source use_std_paddle.sh
    python run_flashattn.py std $i $gpu $log_file
    source use_dev_paddle.sh
    python run_flashattn.py dev $i $gpu $log_file
done
