source use_std_paddle.sh
python time_flashattn.py gpu:5 std_fa.log
source use_dev_paddle.sh
python time_flashattn.py gpu:5 dev_fa.log
python check.py