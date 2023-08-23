python3 train.py --lamda 0.5 --T 0.5 --save checkpoint_0.50._5 &
python3 train.py --lamda 0.5 --T 1 --save checkpoint_0.5_1 &
python3 train.py --lamda 0.5 --T 5 --save checkpoint_0.5_5 &
python3 train.py --lamda 0.5 --T 10 --save checkpoint_0.5_10 &

python3 train.py --lamda 0.1 --T 5 --save checkpoint_0.1_5 &
python3 train.py --lamda 0.9 --T 5 --save checkpoint_0.9_5 &
