# StructureNet: Hierarchical Graph Networks for 3D Shape Generation

![Overview](https://github.com/daerduoCarey/structurenet/blob/master/images/teaser.png)

**Figure 1.** StructureNet is a hierarchical graph network that produces a unified latent space to encode structured models with both continuous geometric and discrete structural variations. In this example, we projected an un-annotated point cloud (left) and un-annotated image (right) into the learned latent space yielding semantically segmented point clouds structured as a hierarchy of graphs. The shape interpolation in the latent space also produces structured point clouds (top) including their corresponding graphs (bottom). Edges correspond to specific part relationships that are modeled by our approach. For simplicity, here we only show the graphs without the hierarchy. Note how the base of the chair morphs via functionally plausible intermediate configurations, or the chair back transitions from a plain back to a back with arm-rests.

## Introduction
We introduce a hierarchical graph network for learning structure-aware shape generation which (i) can directly encode shape parts represented as such n-ary graphs; (ii) can be robustly trained on large and complex shape families such as PartNet; and (iii) can be used to generate a great diversity of realistic structured shape geometries with both both continuous geometric and discrete structural variations.


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
Stanford University, University College London (UCL), University of California San Diego (UCSD), King Abdullah University of Science and Technology (KAUST), Adobe Research and Facebook AI Research.

Arxiv Version: https://arxiv.org/abs/1908.00575

Accepted by [Siggraph Asia 2019](https://sa2019.siggraph.org/). See you at Brisbane, Australia in November!

Project Page: https://cs.stanford.edu/~kaichun/structurenet/

## Citations
    @article{mo2019structurenet,
          title={StructureNet: Hierarchical Graph Networks for 3D Shape Generation},
          author={Mo, Kaichun and Guerrero, Paul and Yi, Li and Su, Hao and Wonka, Peter and Mitra, Niloy and Guibas, Leonidas},
          journal={ACM Transactions on Graphics (TOG), Siggraph Asia 2019},
          volume={38},
          number={6},
          pages={Article 242},
          year={2019},
          publisher={ACM}
    }

## About this repository

This repository provides data and code as follows.


```
    data/                   # contains data, models, results, logs
    code/                   # contains code and scripts
         # please follow `code/README.md` to run the code
    stats/                  # contains helper statistics
```

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## License

MIT Licence

## Updates

* [July 27, 2019] Data and Code released.
* [May 13, 2020] Release the blender code for rendering figures in the paper.

