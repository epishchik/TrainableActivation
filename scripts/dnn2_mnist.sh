# train
python3 train.py --config configs/coslu/mnist/dnn2.yaml
python3 train.py --config configs/default/mnist/dnn2.yaml
python3 train.py --config configs/delu/mnist/dnn2.yaml
python3 train.py --config configs/helu/mnist/dnn2.yaml
python3 train.py --config configs/lincomb/mnist/dnn2.yaml
python3 train.py --config configs/normlincomb/mnist/dnn2.yaml
python3 train.py --config configs/relun/mnist/dnn2.yaml
python3 train.py --config configs/scaledsoftsign/mnist/dnn2.yaml
python3 train.py --config configs/shilu/mnist/dnn2.yaml
python3 train.py --config configs/sinlu/mnist/dnn2.yaml

# test
python3 test.py --config configs/coslu/mnist/dnn2.yaml
python3 test.py --config configs/default/mnist/dnn2.yaml
python3 test.py --config configs/delu/mnist/dnn2.yaml
python3 test.py --config configs/helu/mnist/dnn2.yaml
python3 test.py --config configs/lincomb/mnist/dnn2.yaml
python3 test.py --config configs/normlincomb/mnist/dnn2.yaml
python3 test.py --config configs/relun/mnist/dnn2.yaml
python3 test.py --config configs/scaledsoftsign/mnist/dnn2.yaml
python3 test.py --config configs/shilu/mnist/dnn2.yaml
python3 test.py --config configs/sinlu/mnist/dnn2.yaml