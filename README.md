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
      <img src="https://latex.codecogs.com/svg.latex?f_\theta:\mathcal{X}\rightarrow\mathcal{Y}" title="net"/> 
</p>
parametrized by weights <img src="https://latex.codecogs.com/svg.latex?\theta" title="net"/> the goal is to minimize a loss term of the form

<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{|\mathcal{T}|}\sum_{(x,y)\in\mathcal{T}}l(f_{\theta}(x),y)+\mathrm{Lip}(f_\theta)" title="Lipschitz Loss" />
</p>

where <img src="https://latex.codecogs.com/svg.latex?\mathcal{T}=\{(x_i,y_i)\}_{i=1}^N\subset\mathcal{X}\times\mathcal{Y}" title="training set"/> denotes the training set, 
<img src="https://latex.codecogs.com/svg.latex?l(\cdot,\cdot)" title="loss"/> a loss function and 
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathrm{Lip}(f_\theta)=\sup_{x,x^\prime\in\mathcal{X}}\frac{|f_\theta(x)-f_\theta(x^\prime)|}{|x-x^\prime|}" title="Lipschitz Constant" />
</p>
the Lipschitz constant of the net w.r.t. to the input space variable.


## Prerequistes
Our code is implemented python and utilizes PyTorch https://pytorch.org/ (see also their git https://github.com/pytorch/pytorch). 

## References
<a id="1">[1]</a> Leon Bungert, René Raab, Tim Roith, Leo Schwinn, Daniel Tenbrinck. "CLIP: Cheap Lipschitz Training of Neuronal Networks." arXiv preprint arXiv:2103.12531 (2021). https://arxiv.org/abs/2103.12531


