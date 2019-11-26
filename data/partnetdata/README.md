Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSecwZKRr8is2lTLy8idmrwSMTAB0w65QpSH5BGhLD_v2p5mIw/viewform?usp=sf_link) to download the data.

This zip file provides the processed PartNet hierarchy of graphs data for six object categories used in the paper: bed, chair, storage furniture, table, trashcan, vase.

## About this repository

```
    chair_hier/                         # hierarchy of graphs data for chairs
            [PartNet_anno_id].json      # storing the tree structure, detected sibling edges, 
                                        # and all part oriented bounding box parameters) for a chair
            [train/test/val].txt        # PartNet train/test/val split
            [train/test/val]_no_other_less_than_10_parts.txt    
                                        # Subsets of data used in StructureNet where all parts are labeled 
                                        # and no more than 10 parts per parent node
                                        # We use this subset for StructureNet
    chair_geo/                          # part geometry point clouds for chairs
            [PartNet_anno_id].npz       # storing all part geometry point clouds for a chair

``` 

## Cite

Please cite both [PartNet](https://cs.stanford.edu/~kaichun/partnet/) and StructureNet if you use this data.
    
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

    @InProceedings{Mo_2019_CVPR,
        author = {Mo, Kaichun and Zhu, Shilin and Chang, Angel X. and Yi, Li and Tripathi, Subarna and Guibas, Leonidas J. and Su, Hao},
        title = {{PartNet}: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level {3D} Object Understanding},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }

