
# python fedrolex.py -c example_ResNet_noiid2.yml &> outputs/Resnet18/noiid_0.6.txt 
# python fedrolex.py -c example_ResNet_noiid3.yml &> outputs/Resnet18/noiid_0.9.txt 
python fedrolex.py -c fedrolex/Resnet34_CIFAR10/iid.yml &> outputs/iid.txt 
python fedrolex.py -c fedrolex/Resnet34_CIFAR10/noiid_0.6.yml &> outputs/noiid_0.6.txt 