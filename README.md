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
      <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{|\mathcal{T}|}\sum_{(x,y)\in\mathcal{T}}l(f_{\theta}(x),y)+\mathrm{Lip}(f_\theta)," title="Lipschitz Loss" />
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



## Prerequistes
Our code is implemented python and utilizes PyTorch[[2]](#2). 

## References
<a id="1">[1]</a> Leon Bungert, René Raab, Tim Roith, Leo Schwinn, Daniel Tenbrinck. "CLIP: Cheap Lipschitz Training of Neuronal Networks." arXiv preprint arXiv:2103.12531 (2021). https://arxiv.org/abs/2103.12531
<a id="2">[2]</a> https://pytorch.org/, see also their git https://github.com/pytorch/pytorch.


