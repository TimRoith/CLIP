from dataclasses import dataclass

@dataclass
class dataset:
    mean: int = 0.0
    std: int = 1.0
    xrange: tuple = (0.,1.)
    num_workers: int = 0
    path: str = '../../datasets/'
    download: bool = False
    batch_size: int = 32


@dataclass
class model:
    name: str = 'CNN'
    path: str = '../../weights/'
    
@dataclass
class cfg:
    data: dataset
    model: model
    device: str = 'cpu'