casenums=$(python get_launch_config.py 0)
gpu=$(python get_launch_config.py 1)
log_file=$(python get_launch_config.py 2)
for i in $casenums; do
    python time_qkvpacked.py qkvpacked $i $gpu $log_file
    python time_qkvpacked.py notpacked $i $gpu $log_file
done
