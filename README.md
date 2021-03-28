# CLIP
Implementation of the *CLIP* algorithm for Lipschitz regularization of neural networks, proposed in **CLIP: Cheap Lipschitz Training of Neuronal Networks** [[1]](#1).
Feel free to use it and please refer to our paper when doing so.
```
@misc{bungert2021clip,
      title={CLIP: Cheap Lipschitz Training of Neural Networks}, 
      author={Leon Bungert and René Raab and Tim Roith and Leo Schwinn and Daniel Tenbrinck},
      year={2021},
      eprint={2103.12531},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Method Description
The CLIP Algorithm proposes a regularization for controlling the Lipschitz constant of a neural network. For a neural network 
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;f_\theta:\mathcal{X}\rightarrow\mathcal{Y}" title="net"/> 
</p>

parametrized by weights <img src="https://latex.codecogs.com/svg.latex?\theta" title="weights"/> the goal is to minimize a loss term of the form

<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{|\mathcal{T}|}\sum_{(x,y)\in\mathcal{T}}l(f_{\theta}(x),y)+\lambda~\mathrm{Lip}(f_\theta)," title="Lipschitz Loss" />
</p>

where
* <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}=\{(x_i,y_i)\}_{i=1}^N\subset\mathcal{X}\times\mathcal{Y}" title="training set"/> denotes the training set, 
* <img src="https://latex.codecogs.com/svg.latex?l(\cdot,\cdot)" title="loss"/> denotes a loss function.

The Lipschitz constant of the net w.r.t. to the input space variable is defined as
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Lip}(f_\theta)=\sup_{x,x^\prime\in\mathcal{X}}\frac{|f_\theta(x)-f_\theta(x^\prime)|}{|x-x^\prime|}." title="Lipschitz Constant" />
</p>

In the algorithm we approximate this constant via difference quotients on a finite subset
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{X}_{\mathrm{Lip}}\subset\mathcal{X}\times\mathcal{X}." title="Lipschitz Constant" />
</p>

In order to ensure that the tuples in this set correspond to point with that realize a high difference qoutient the set is updated iteratively 
via a gradient ascent scheme.



## Usage
Our code is implemented Python and utilizes PyTorch [[2]](#2). An example how to use the code is provided in the file ```main.py```. 
Therein the dictionary ```conf_arg``` specifies the cofiguration of a run.
### CUDA Settings
* ```conf_arg['use_cude']```: Boolean that specifies wether the model should be trained on the CPU, default: ```False```.

### Datasets
The example loads the data via helper methods which then call the standard dataloaders provided by PyTorch.
* ```conf_arg['data_file']```: Specifies the path to the dataset you wish to use. 
* ```conf_arg['data_set']:```: Name of the dataset. 
    * ```"MNIST"``` (Default, [[3]](#3))
    * ```"Fashion-MNIST"``` ([[4]](#4)). 
The dataloaders are then created by  
```
train_loader, valid_loader, test_loader = get_data_set(conf.dataset, conf.data_file, conf.batch_size)
```
Note: The ```download``` flags for the torchvision dataset methods are all set to ```False```. You can easily subsitute this by your other dataloaders as long 
the three loaders ```train_loader, valid_loader, test_loader``` are specified.

### Model
The example loads a simple fully connected net from the file ```model.py```.
* ```conf_arg['model']```: Specifies the model that should be loaded from ```model.py```.
    * ```"fc"``` (Dafault, fully connected model, currently only possibility).
* ```conf_arg['activation_function']```: Specifies the activation function for the net.
    * ```"ReLU"``` (Default),
    * ```"sigmoid"```.

The model is then loaded via
```
model = models.fully_connected([784, 400, 200, 10], conf.activation_function)
model.to(conf.device)
```
where ```[784, 400, 200, 10]``` denotes the layer dimesnions. Alternatively, you can use an arbitrary PyTorch model, i.e., a subclass of ```nn.Module```.
### Regularization
You can specify the following options for the regularizer:
* ```conf_arg['regularization']```: Specifies which regularization to use:
    * ```"global_lipschitz"```: activates the CLIP reguarization as described above (Default),
    * ```"none"```: decativates any kind of additional regularization.
* ```conf_arg['reg_iters']```: The number of gradient ascent steps for the Lipschitz set update, default: ```reg_iters=1```.
* ```conf_arg['reg_lr']```: Step size for the gradient ascent scheme, default: ```reg_lr=1.0```.
* ```conf_arg['reg_interval']```: Specifies in which interval the regularization is applied, default: ```reg_interval=1```.
* ```conf_arg['reg_max']```: Specifies the maximum value of a Lipschitz constant that is allowed to enter the backprop. Note, that large Lipschitz regularization terms yield numerical instabilities, default: ```reg_max=5e3```.
* ```conf_arg['reg_init']```: Specifies the initialization strategy of the Lipschitz set.
    * ```"plain"```: Splits a batch of the loader equally and assigns each half to ```u``` and ```v``` (Default).
    * ```"partial_random"```: Assigns ```u``` to a batch of the loader and sets ```v = u + delta```, where ```delta``` is sampled from gaussian distribution.
    * ```"noise"```: Chooses ```u,v``` as samples from the normal distribution.
* ```conf_arg['goal_accuracy']```:
* ```conf_arg['alpha']```:
* ```conf_arg['alpha_incremental']```:

### Adverserial Attack
General:
* ```gauss_attack(nl=1.0)```:

* ```fgsm(model, loss, epsilon=0.3)```:

* ```pgd(model, loss, epsilon=None, alpha=None, alpha_mul=1.0, restarts=1, attack_iters=7, norm_type="l2")```:

## References
<a id="1">[1]</a> Leon Bungert, René Raab, Tim Roith, Leo Schwinn, Daniel Tenbrinck. "CLIP: Cheap Lipschitz Training of Neuronal Networks." arXiv preprint arXiv:2103.12531 (2021). https://arxiv.org/abs/2103.12531

<a id="2">[2]</a> The Pytorch website https://pytorch.org/, see also their git https://github.com/pytorch/pytorch

<a id="3">[3]</a> The MNIST dataset http://yann.lecun.com/exdb/mnist/

<a id="4">[4]</a> The Fashion-MNIST dataset  https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/


