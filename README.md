# StructureNet: Hierarchical Graph Networks for 3D Shape Generation

![Overview](https://github.com/daerduoCarey/structurenet/blob/master/images/teaser.png)

**Figure 1.** StructureNet is a hierarchical graph network that produces a unified latent space to encode structured models with both continuous geometric and discrete structural variations. In this example, we projected an un-annotated point cloud (left) and un-annotated image (right) into the learned latent space yielding semantically segmented point clouds structured as a hierarchy of graphs. The shape interpolation in the latent space also produces structured point clouds (top) including their corresponding graphs (bottom). Edges correspond to specific part relationships that are modeled by our approach. For simplicity, here we only show the graphs without the hierarchy. Note how the base of the chair morphs via functionally plausible intermediate configurations, or the chair back transitions from a plain back to a back with arm-rests.

## Introduction

The ability to generate novel, diverse, and realistic 3D shapes along with associated part semantics and structure is central to many applications requiring high-quality 3D assets or large volumes of realistic training data. A key challenge towards this goal is how to accommodate diverse shape variations, including both continuous deformations of parts as well as structural or discrete alterations which add to, remove from, or modify the shape constituents and compositional structure. Such object structure can typically be organized into a hierarchy of constituent object parts and relationships, represented as a hierarchy of n-ary graphs. We introduce StructureNet, a hierarchical graph network which (i) can directly encode shapes represented as such n-ary graphs, (ii) can be robustly trained on large and complex shape families, and (iii) be used to generate a great diversity of realistic structured shape geometries. Technically, we accomplish this by drawing inspiration from recent advances in graph neural networks to propose an order-invariant encoding of n-ary graphs, considering jointly both part geometry and inter-part relations during network training. We extensively evaluate the quality of the learned latent spaces for various shape families and show significant advantages over baseline and competing methods. The learned latent spaces enable several structure-aware geometry processing applications, including shape generation and interpolation, shape editing, or shape structure discovery directly from un-annotated images, point clouds, or partial scans.

## About the paper

Our team: 
[Kaichun Mo](https://cs.stanford.edu/~kaichun),
[Paul Guerrero](http://paulguerrero.net/),
[Li Yi](https://cs.stanford.edu/~ericyi/),
[Hao Su](http://cseweb.ucsd.edu/~haosu/),
[Peter Wonka](http://peterwonka.net/),
[Niloy Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/),
and [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) 
from 
Stanford University, University College London (UCL), University of California San Diego (UCSD), King Abdullah University of Science and Technology (KAUST).

Arxiv Version: xxx

Project Page: xxx

## Citations
xxx

## About this repository

This repository provides data and code as follows.


```
    data/                   # contains data, models, results, logs
    exps/                   # contains code and scripts
         # please follow `exps/README.md` to run the codes 
    stats/                  # contains helper statistics
```

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## License

MIT Licence

## Updates

* [July xxx, 2019] Data and Code released.

