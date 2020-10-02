import numpy as np
import random
import torch

class Config():
    def __init__(self):
        self.experiment_seed = 0
        self.train_data = './data/computers_train_xlarge.json'
        self.test_data = './data/testset_1500_with_labels.json'
        self.model_path = './model/'
        self.result_path = './result/'
        self.model = 'bert-base-uncased'
        self.exist_model = ''
        self.target_list = ['bert-large-uncased', 'roberta-base', 'roberta-large']
        self.batch_size = 32
        self.lr = 2e-5 #learning rate
        self.epochs = 2
        self.log_interval = 50
        self.device = self.get_device()

    def get_device(self):
        device = None
        # If there’s a GPU available
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU
            device = torch.device('cuda:0')
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('Using GPU: ', torch.cuda.get_device_name(0))
        # If not, use cpu
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device('cpu')
        return device

    # For reproducibility set all random seed in the model
    def set_seed(self):
        np.random.seed(self.experiment_seed)
        random.seed(self.experiment_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(self.experiment_seed)
        torch.cuda.manual_seed_all(self.experiment_seed)
        torch.manual_seed(self.experiment_seed)
        torch.random.manual_seed(self.experiment_seed)
