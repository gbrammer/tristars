
def generate_catalogs(seed=1, N1=50, N2=10, tr=[40,50], rot=0.1, err=0.1):
    """
    Generate test catalogs for matching
    """
    import numpy as np    
    from skimage.transform import SimilarityTransform
    
    np.random.seed(seed)
    
    V1 = np.random.rand(N1,2)*1000-500

    # Transform and add noise to second catalog, drawn from the first  
    draw_ix = np.unique(np.cast[int](np.random.rand(N2)*N1))
    N2 = len(draw_ix)
    
    tf = SimilarityTransform(matrix=None, scale=1., rotation=rot,
                             translation=tr)
    
    V2 = tf(V1[draw_ix,:])+np.random.normal(size=(N2,2))*err
    
    return V1, V2, tf
    
def test(seed=1, N1=50, N2=10, tr=[40,50], rot=0.1, err=0.1, make_figure=True, Nextra2=10, size_limit=[5,800], ba_max=0.9):
    """
    Demo of the matching script
    """
    import numpy as np
    
    from skimage.transform import SimilarityTransform

    try:
        from . import match
        from . import test
    except:
        from tristars import match 
        from tristars import test 
        
    # Length of parent catalog
    #N = 50
    #tr = [10, 20]
    #rot = 1./180*np.pi # 
    
    # Random draws
    #Ndraw = 10
    #tol = 5.e-5
    #err = 0.1
    #Nkeep = 5
    
    V1, V2, tf = test.generate_catalogs(seed=seed, N1=N1, N2=N2, tr=tr, rot=rot, err=err)
    
    if Nextra2 > 0:
        xV1, xV2, tf = test.generate_catalogs(seed=seed*2, N1=N1, N2=Nextra2, tr=tr, rot=rot, err=err)
        V2 = np.vstack([V2, xV2])
        
    # Test2 j230822-021134
    if False:
        import astropy.io.fits as pyfits
        import astropy.wcs as pywcs

        from skimage.transform import SimilarityTransform
    
        root = 'aco-2537-ir-czg-gd-050.0-f160w'
        im = pyfits.open('{0}_drz_sci.fits'.format(root))
        wcs = pywcs.WCS(im[0].header)
    
        xref = [im[0].header['CRPIX1'], im[0].header['CRPIX2']]
    
        drz = utils.read_catalog('{0}.cat.fits'.format(root))
    
        cut1 = (drz['MAG_AUTO'] > 18.5) & (drz['MAG_AUTO'] < 22)
        V1 = np.vstack([drz['X_IMAGE']-xref[0], drz['Y_IMAGE']-xref[1]]).T[cut1,:]#[:50,:]
    
        ps1 = utils.read_catalog('aco-2537-ir-czg-gd-050.0-f160w_ps1.radec')
        x, y = wcs.all_world2pix(ps1['ra'], ps1['dec'], 1)
        cut2 = (x > 0) & (x < wcs._naxis1) & (y > 0) & (y < wcs._naxis2)
        V2 = np.vstack([x-xref[0], y-xref[1]]).T[cut2,:]#[:50,:]
    
        # Cut drizzled catalog at 2x surface density of PS1
        icut = np.minimum(len(drz), int(1.5*cut2.sum()))
        cut1 = np.argsort(drz['MAG_AUTO'])[:icut]
        V1 = np.vstack([drz['X_IMAGE']-xref[0], drz['Y_IMAGE']-xref[1]]).T[cut1,:]#[:50,:]
    
    ## Run the matching
    # pair_ix = match_catalog_quads(V1, V2, maxKeep=2, normed_triangles=False)
    # 
    # V1 = V1[pair_ix[:,0],:]
    # V2 = V2[pair_ix[:,1],:]
    
    pair_ix = match.match_catalog_tri(V1, V2, maxKeep=4, auto_keep=3, ignore_rot=False, ba_max=ba_max, size_limit=size_limit)#, ignore_rot=True, ignore_scale=True, ba_max=0.98, size_limit=[1,1000])
    
    ## Check output transform
    tfo, dx, rms = match.get_transform(V1, V2, pair_ix, transform=SimilarityTransform, use_ransac=True)
    #tfo = SimilarityTransform() #
    
    ## Make diagnostic plot
    if make_figure:
        fig = match.match_diagnostic_plot(V1, V2, pair_ix, tf=tfo,
                                          new_figure=True)
    else:
        fig = None
        
    print(' Input transform: translation=[{0:.2f}, {1:.2f}] rotation={2:.4f}'.format(tf.translation[0], tf.translation[1], tf.rotation))

    print('Output transform: translation=[{0:.2f}, {1:.2f}] rotation={2:.4f}'.format(tfo.translation[0], tfo.translation[1], tfo.rotation))
    
    return fig
    