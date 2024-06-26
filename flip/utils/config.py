from dataclasses import dataclass

@dataclass
class dataset:
    mean: int = 0.0
    std: int = 1.0
    xrange: tuple = (0.,1.)
    num_workers: int = 0
    path: str = '../../datasets/'
    name: str = 'MNIST'
    download: bool = False
    batch_size: int = 32


@dataclass
class model_attributes:
    name: str = 'CNN'
    path: str = '../../weights/'
    file_name: str = 'cnn.pt'
    sizes: list = None
    act_fun: str = 'ReLU'
    
@dataclass
class cfg:
    data: dataset
    model: model_attributes
    device: str = 'cpu'