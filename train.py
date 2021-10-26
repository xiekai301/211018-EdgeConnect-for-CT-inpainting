import os
import torch
from src.utils import load_config
from src.edge_connect import EdgeConnect


def main(mode=None):
    """starts the model
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # build the model and initialize
    model = EdgeConnect(config)
    model.load()

    # model training
    config.print()
    print('\nstart training...\n')
    model.train()


if __name__ == "__main__":
    main()
