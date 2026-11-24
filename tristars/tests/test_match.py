def generate_catalogs(seed=1, N1=50, N2=10, tr=[40, 50], rot=0.1, err=0.1):
    """
    Generate test catalogs for matching
    """
    import numpy as np
    from skimage.transform import SimilarityTransform

    np.random.seed(seed)

    V1 = np.random.rand(N1, 2) * 1000 - 500

    # Transform and add noise to second catalog, drawn from the first
    draw_ix = np.unique((np.random.rand(N2) * N1).astype(int))
    N2 = len(draw_ix)

    tf = SimilarityTransform(matrix=None, scale=1.0, rotation=rot, translation=tr)

    V2 = tf(V1[draw_ix, :]) + np.random.normal(size=(N2, 2)) * err

    return V1, V2, tf


def test_match(
    seed=1,
    N1=50,
    N2=10,
    tr=[40, 50],
    rot=0.1,
    err=0.1,
    make_figure=True,
    Nextra2=10,
    size_limit=[5, 800],
    ba_max=0.9,
):
    """
    Demo of the matching script
    """
    import numpy as np

    from skimage.transform import SimilarityTransform

    try:
        from .. import match
    except:
        from tristars import match

    # Length of parent catalog
    # N = 50
    # tr = [10, 20]
    # rot = 1./180*np.pi #

    # Random draws
    # Ndraw = 10
    # tol = 5.e-5
    # err = 0.1
    # Nkeep = 5

    V1, V2, tf = generate_catalogs(seed=seed, N1=N1, N2=N2, tr=tr, rot=rot, err=err)

    if Nextra2 > 0:
        xV1, xV2, tf = generate_catalogs(
            seed=seed * 2, N1=N1, N2=Nextra2, tr=tr, rot=rot, err=err
        )
        V2 = np.vstack([V2, xV2])

    ## Run the matching
    # pair_ix = match_catalog_quads(V1, V2, maxKeep=2, normed_triangles=False)
    #
    # V1 = V1[pair_ix[:,0],:]
    # V2 = V2[pair_ix[:,1],:]

    pair_ix = match.match_catalog_tri(
        V1,
        V2,
        maxKeep=4,
        auto_keep=3,
        ignore_rot=False,
        ba_max=ba_max,
        size_limit=size_limit,
    )  # , ignore_rot=True, ignore_scale=True, ba_max=0.98, size_limit=[1,1000])

    ## Check output transform
    tfo, dx, rms = match.get_transform(
        V1, V2, pair_ix, transform=SimilarityTransform, use_ransac=True
    )
    # tfo = SimilarityTransform() #

    ## Make diagnostic plot
    if make_figure:
        fig = match.match_diagnostic_plot(V1, V2, pair_ix, tf=tfo, new_figure=True)
    else:
        fig = None

    assert np.allclose(tf.translation, tfo.translation, rtol=1.0e-2)
    assert np.allclose(tf.rotation, tfo.rotation, rtol=1.0e-2)

    print(
        " Input transform: translation=[{0:.2f}, {1:.2f}] rotation={2:.4f}".format(
            tf.translation[0], tf.translation[1], tf.rotation
        )
    )

    print(
        "Output transform: translation=[{0:.2f}, {1:.2f}] rotation={2:.4f}".format(
            tfo.translation[0], tfo.translation[1], tfo.rotation
        )
    )

    return fig
