# python FedPRC.py -c FedPRC_ResNet_CIFAR10.yml &> outputs/iid.txt
# python FedPRC.py -c FedPRC_ResNet_CIFAR10_noiid1.yml &> outputs/noiid1.txt
# python FedPRC.py -c FedPRC_ResNet_CIFAR10_noiid2.yml &> outputs/noiid2.txt
# python FedPRC.py -c FedPRC_ResNet_CIFAR10_noiid3.yml &> outputs/noiid3.txt
# python FedPRC.py -c config/new_3.yml &> outputs/new_3.txt
# python FedPRC.py -c config/new_4.yml &> outputs/new_4.txt
# python FedPRC.py -c config/new_6.yml &> outputs/new_6.txt
# python FedPRC.py -c config/new_7.yml &> outputs/new_7.txt
python FedPRC.py -c config/hsn.yml &> outputs/iid_monent_0.9.txt 
python FedPRC.py -c config/hsn_noiid0.6.yml &> outputs/noiid_0.6_monent_0.9.txt 
python FedPRC.py -c config/unhsn_noiid0.6.yml &> outputs/noiid_0.6_monent_0.9.txt 