Please download the processed PartNet hierarchy of graphs data and unzip the files here.

```
    chair_hier/                         # hierarchy of graphs data for chairs
            [PartNet_anno_id].json      # storing the tree structure, detected sibling edges, 
                                        # and all part oriented bounding box parameters) for a chair
            [train/test/val].txt        # PartNet train/test/val split
            [train/test/val]_no_other_less_than_10_parts.txt    
                                        # Subsets of data used in StructureNet where all parts are labeled 
                                        # and no more than 10 parts per parent node
                                        # We use this subset for StructureNet
            semantics.txt               # PartNet defined part semantics
            semantic_colors.txt         # colors assigned to each part semantic for visualization
    chair_geo/                          # part geometry point clouds for chairs
            [PartNet_anno_id].npz       # storing all part geometry point clouds for a chair

``` 

