# Point Cloud Registration Based on Rotation invariant Feature

This is a point cloud registration algorithm using rotation invariant feature. The feature abstract network is modified from [PVCNN](https://github.com/mit-han-lab/pvcnn). We modify it to be invariant to rotation. For each point $\mathbf{p}$,  we abstract the Point Pair Features(PPF) between $\mathbf{p}$ and its k nearest neighbors  $\mathbf{p}_i$ and fuse them as the local feature of $\mathbf{p}$. The PPF is a hand-craft rotation invariant feature, whose definition is shown as following:
$$
\left(\left<\mathbf{n},\mathbf{n}_{1}\right>,\left<\mathbf{n},\mathbf{d}\right>,\left<\mathbf{n}_{1},\mathbf{d}\right>,|\mathbf{d}|\right)
$$
where $\mathbf{n}, \mathbf{n}_{1}$ are the normals of $\mathbf{p}$ and $\mathbf{p}_1$ respectively，$\mathbf{d}=\mathbf{p}-\mathbf{p}_1$，$\left< \cdot \right> $ is a function giving the angle between two vectors.

We fuse them using MLP, thus the local features are rotation invariant as well. We also create a local reference frame to rotate the point cloud, which can mitigate the rotation impact. As an illustration shown below, the local reference frame is defined as follows. 

<img src="assets\LRF.png" style="zoom: 33%;" />
$$
LRF = \left[\mathbf{Op_{\max}},\mathbf{{Op_{\max}}_{\bot}}, \mathbf{Op_{\max}}\times\mathbf{{Op_{\max}}_{\bot}}\right]\\
\mathbf{{Op_{\max}}_{\bot}} = \mathbf{Op_{\min}}-\left( \mathbf{Op_{\min}}\bullet \mathbf{Op_{\max}} \right) \mathbf{Op_{\max}}
$$
Where $\mathbf{O}$ is the point cloud's center, $\mathbf{p_{\max}}$ is the point furthest from the center, and $\mathbf{p_{\min}}$ is the point closest to the center. We use the LRF to rotate the point cloud, then concatenate the coordinates with the local features to create initial features for the later network.

The features of the points are no longer susceptible to rotation after these two processes. PVCNN's voxelization component, on the other hand, is cube voxelization, whose voxelization outputs are closely related to 3D coordinates. As a result, the same point will be assigned to various voxels in different coordinate frames. We transform it into a spherical voxel. That is, we convert from a Cartesian to a spherical coordinate system. We also employ DGCNN instead of the PointNet used by the original PVCNN to improve the feature's ability to describe in detail. To speed up the network, we created two Pytorch extensions that calculate the PPF feature in CUDA and perform CUDA-based spherical voxelization. Furthermore, instead of grouping, we use the neighborhood knowledge provided by voxelization to perform DGCNN, which speeds up the network even more. The network's detailed architecture is presented below.

![Rotation Invariant Feature Extracter](assets\旋转不变特征提取器网络结构图.jpg)



To verify the network, we use it to classify the random rotated modelnet40 dataset. The following diagram depicts the classification network architecture:

![Classification Network.The blue blocks are voxel-dgcnn block, in which $(64, 64, 32)$ indicate that there are two cascading single layer voxel-dgcnn block which ouputs $64$ channels and the resolution of whose voxelization is $32*32*32$.Green blocks are original MLP block, in which $(64, 64)$ indicate that there are two cascading single MLP layer which ouputs $64$ channels. Gray block is the score vector which gives the predicting probability that the input belong to corresponding class](assets\pvcnn_classify.jpg)

We compare the results to several state-of-the-art networks (until 2020).

| Method                                                       | w/o random rotation(%) | random rotation(%) |
| ------------------------------------------------------------ | ---------------------- | ------------------ |
| PointNet                                                     | 89.2                   | 75.5               |
| PointNet++                                                   | 91.8                   | 77.4               |
| [SO-Net](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf) | **92.6**               | 80.2               |
| DGCNN                                                        | 92.2                   | 81.1               |
| [Spherical CNN](https://openreview.net/pdf?id=Hkbd5xZRb)     | 88.9                   | 86.9               |
| [ClusterNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_ClusterNet_Deep_Hierarchical_Cluster_Network_With_Rigorously_Rotation-Invariant_Representation_for_CVPR_2019_paper.pdf) | 87.1                   | 87.1               |
| Ours(cu-pt)                                                  | 92.0                   | **88.8**           |
| Ours(cu-dg)                                                  | 91.9                   | **89.3**           |
| Ours(sph-pt)                                                 | 91.9                   | **86.9**           |
| Ours(sph-dg)                                                 | 91.9                   | **89.7**           |

Then we use the model trained in classifying task as the feature extractor to register point cloud in Modelnet40. The registration pipeline is shown below. We follow the classical two-stage registration pipeline. First, extract and match points' feature between the source point cloud and target point cloud to get correspondences, then use robust pose estimator such as RANSAC and TEASER to recover pose from noisy correspondences.

![](assets\pvcnn_registration.jpg)

[DeepGMR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500715.pdf)'s ModelNet40-Noisy, ICL-NUIM, and ModelNet40-Noisy-Partial datasets are used to evaluate registration performance. The results are shown below. Note that our model was trained on classification tasks without any finetuning in the registration datasets, whereas DeepGMR was trained on them. It can be seen that our method has good generalization ability.

The registration results on ModelNet40-Noisy

|         | RRE($\degree$) | RTE(m) | RMSE   |
| ------- | -------------- | ------ | ------ |
| DeepGMR | 1.74           | 0.0065 | 0.0124 |
| Ours    | 2.892          | 0.0118 | 0.0175 |

The registration results on ICL-NUIM

|         | RRE($\degree$) | RTE(m)      | RMSE   |
| ------- | -------------- | ----------- | ------ |
| DeepGMR | 0.60           | 0.0243      | 0.0077 |
| Ours    | **0.5156**     | **0.02112** | 0.0085 |

The registration results on ModelNet40-Noisy-Partial

|         | RRE($\degree$) | RTE(m)     | RMSE       |
| ------- | -------------- | ---------- | ---------- |
| DeepGMR | 59.19          | 0.2013     | 0.3848     |
| Ours    | **31.0813**    | **0.1177** | **0.2044** |

Here we give some qualitative results:

| Before     | <img src="assets\desk_nb.png" style="zoom:25%;" /> | <img src="assets\cap_nb.png" style="zoom:25%;" /> | <img src="assets\basin_nb.png" style="zoom:25%;" /> | <img src="assets\partial_chair_nb.png" style="zoom:25%;" /> |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **After**  | <img src="assets\desk_na.png" style="zoom:25%;" /> | <img src="assets\cap_na.png" style="zoom:25%;" /> | <img src="assets\basin_na.png" style="zoom:25%;" /> | <img src="assets\partial_chair_na.png" style="zoom:25%;" /> |
| **Before** | <img src="assets\partial_plane_nb.png" style="zoom:25%;" /> | <img src="assets\basin_ib.png" style="zoom:25%;" /> | <img src="assets\desk_ib.png" style="zoom:25%;" /> | <img src="assets\plane_ib.png" style="zoom:25%;" /> |
| **After**  | <img src="assets\partial_plane_ia.png" style="zoom:25%;" /> | <img src="assets\basin_ia.png" style="zoom:25%;" /> | <img src="assets\desk_ia.png" style="zoom:25%;" /> | <img src="assets\plane_ig.png" style="zoom:25%;" /> |
