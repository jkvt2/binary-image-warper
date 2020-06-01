# binary-image-warper
Estimate the warp mesh required to transform one binary image to another.
The warp mesh this algorithm finds considers both the error term between the desired output and the given starting image, as well as the degree of local warping. Set a balance between these terms with the mesh_regulariser_weight variable.

![Toy example input and desired output](https://github.com/jkvt2/binary-image-warper/figures/toy_example.png)
![Snapshots of the warping across iterations](https://github.com/jkvt2/binary-image-warper/figures/progress.png)
![Graph of error against iterations](https://github.com/jkvt2/binary-image-warper/figures/error_over_iterations.png)
