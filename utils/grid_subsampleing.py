import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    """
    CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param grid_size: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: sub_sampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
    elif labels is None:
        return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
    elif features is None:
        return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)