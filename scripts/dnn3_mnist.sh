# train
python3 train.py --config configs/coslu/mnist/dnn3.yaml
python3 train.py --config configs/default/mnist/dnn3.yaml
python3 train.py --config configs/delu/mnist/dnn3.yaml
python3 train.py --config configs/helu/mnist/dnn3.yaml
python3 train.py --config configs/lincomb/mnist/dnn3.yaml
python3 train.py --config configs/normlincomb/mnist/dnn3.yaml
python3 train.py --config configs/relun/mnist/dnn3.yaml
python3 train.py --config configs/scaledsoftsign/mnist/dnn3.yaml
python3 train.py --config configs/shilu/mnist/dnn3.yaml
python3 train.py --config configs/sinlu/mnist/dnn3.yaml

# test
python3 test.py --config configs/coslu/mnist/dnn3.yaml
python3 test.py --config configs/default/mnist/dnn3.yaml
python3 test.py --config configs/delu/mnist/dnn3.yaml
python3 test.py --config configs/helu/mnist/dnn3.yaml
python3 test.py --config configs/lincomb/mnist/dnn3.yaml
python3 test.py --config configs/normlincomb/mnist/dnn3.yaml
python3 test.py --config configs/relun/mnist/dnn3.yaml
python3 test.py --config configs/scaledsoftsign/mnist/dnn3.yaml
python3 test.py --config configs/shilu/mnist/dnn3.yaml
python3 test.py --config configs/sinlu/mnist/dnn3.yaml