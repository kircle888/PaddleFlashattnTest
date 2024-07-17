casenums=$(python get_launch_config.py 0)
gpu=$(python get_launch_config.py 1)
log_file=$(python get_launch_config.py 2)
for i in $casenums; do
    python time_unpad.py $i $gpu $log_file
    python time_varlen.py $i $gpu $log_file
    # python time_flashmask.py $i $gpu $log_file
done
