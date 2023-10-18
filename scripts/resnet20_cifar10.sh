# train
python3 train.py --config configs/default/cifar10/resnet20.yaml
python3 train.py --config configs/lincomb/cifar10/resnet20.yaml
python3 train.py --config configs/shilu/cifar10/resnet20.yaml
python3 train.py --config configs/scaledsoftsign/cifar10/resnet20.yaml
python3 train.py --config configs/relun/cifar10/resnet20.yaml
python3 train.py --config configs/helu/cifar10/resnet20.yaml
python3 train.py --config configs/delu/cifar10/resnet20.yaml
python3 train.py --config configs/sinlu/cifar10/resnet20.yaml
python3 train.py --config configs/coslu/cifar10/resnet20.yaml
python3 train.py --config configs/normlincomb/cifar10/resnet20.yaml

# test
python3 test.py --config configs/default/cifar10/resnet20.yaml
python3 test.py --config configs/lincomb/cifar10/resnet20.yaml
python3 test.py --config configs/shilu/cifar10/resnet20.yaml
python3 test.py --config configs/scaledsoftsign/cifar10/resnet20.yaml
python3 test.py --config configs/relun/cifar10/resnet20.yaml
python3 test.py --config configs/helu/cifar10/resnet20.yaml
python3 test.py --config configs/delu/cifar10/resnet20.yaml
python3 test.py --config configs/sinlu/cifar10/resnet20.yaml
python3 test.py --config configs/coslu/cifar10/resnet20.yaml
python3 test.py --config configs/normlincomb/cifar10/resnet20.yaml