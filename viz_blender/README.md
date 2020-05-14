# Blender Render Code

This code uses Blender 2.79.

## Render Part PC
Check `render_one_pc.sh` for rendering point cloud shape, with different colors for different semantic parts and different part instances.

We need three input files: `example_pc.pts` listing all point coordinates, `example_pc-sem.label` assigning a semantic label for each point, and `example_pc-ins.label` assigning an instance label for each point. 
The semantic labels match the ones defined in `semantics.txt`. 
The semantic colormaps are defined in `semantic_colors.txt`
Different part instances with the same semantics will use the color offsets defined in `instance_color_offsets.txt` to render with slightly different colors.

![examplepc](https://github.com/daerduoCarey/structurenet/blob/master/viz_blender/example_pc.png)

## Render Shape PC
Check `render_one_pc_same_color.sh` for rendering the whole point cloud shape with the same color. We only need an input `example_pc.pts` listing all point coordinates.

![examplepconecolor](https://github.com/daerduoCarey/structurenet/blob/master/viz_blender/example_pc_same_color.png)


## Render Part Boxes
Check `render_one_boxshape.sh` for rendering a box-shape defined in a StructureNet-compatible JSON format.


![exampleboxshape](https://github.com/daerduoCarey/structurenet/blob/master/viz_blender/example_boxshape.png)


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


