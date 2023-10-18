# train
python3 train.py --config configs/coslu/mnist/dnn5.yaml
python3 train.py --config configs/default/mnist/dnn5.yaml
python3 train.py --config configs/delu/mnist/dnn5.yaml
python3 train.py --config configs/helu/mnist/dnn5.yaml
python3 train.py --config configs/lincomb/mnist/dnn5.yaml
python3 train.py --config configs/normlincomb/mnist/dnn5.yaml
python3 train.py --config configs/relun/mnist/dnn5.yaml
python3 train.py --config configs/scaledsoftsign/mnist/dnn5.yaml
python3 train.py --config configs/shilu/mnist/dnn5.yaml
python3 train.py --config configs/sinlu/mnist/dnn5.yaml

# test
python3 test.py --config configs/coslu/mnist/dnn5.yaml
python3 test.py --config configs/default/mnist/dnn5.yaml
python3 test.py --config configs/delu/mnist/dnn5.yaml
python3 test.py --config configs/helu/mnist/dnn5.yaml
python3 test.py --config configs/lincomb/mnist/dnn5.yaml
python3 test.py --config configs/normlincomb/mnist/dnn5.yaml
python3 test.py --config configs/relun/mnist/dnn5.yaml
python3 test.py --config configs/scaledsoftsign/mnist/dnn5.yaml
python3 test.py --config configs/shilu/mnist/dnn5.yaml
python3 test.py --config configs/sinlu/mnist/dnn5.yaml