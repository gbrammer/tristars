"""
Matching functions.  

Only triangles are reasonably robust for now!
"""

import numpy as np
import os

def match_catalog_tri(V1, V2, maxKeep=10, auto_keep=False,
                      auto_transform=None, auto_limit=3, 
                      size_limit=[5,800], ignore_rot=True, ignore_scale=True, 
                      ba_max=0.9):
    """
    Match to position lists together using quads similar to Heyl (2013).
    
    Parameters
    ----------
    V1 : [N1,2] array
        Coordinate set #1.
    
    V2 : [N2,2] array
        Coordinate set #2.
        
    maxKeep : int
        Max number of triangles to keep, sorted by the tree match separation.
    
    auto_keep : bool/int
        If specified, then first compute an `auto_transform` transformation
        with the best `auto_keep` matched triangles. Then compute the
        residuals of *all* pairs with this tranform and finally return pairs
        where the residuals are less than `auto_limit` pixels.
    
    auto_limit : float
        See `auto_keep`.
    
    size_limit, ignore_rot, ignore_scale, ba_max : see `cat_tri_hash`
    
        
    Returns
    -------
    pair_ix : [M,2] array
        Indices of derived matched pairs.

    """
    from scipy import spatial
    import skimage.transform
    
    if auto_transform is None:
        auto_transform = skimage.transform.SimilarityTransform
    
    tri_1, ix_1 = cat_tri_hash(V1, size_limit=size_limit,
                               ignore_scale=ignore_scale, 
                               ignore_rot=ignore_rot, ba_max=ba_max)

    tri_2, ix_2 = cat_tri_hash(V2, size_limit=size_limit, 
                               ignore_scale=ignore_scale, 
                               ignore_rot=ignore_rot, ba_max=ba_max)
                
    # Use scipy.spatial.cKDTree
    tree = spatial.cKDTree(tri_1)
    dist, ix = tree.query(tri_2)
    
    if auto_keep:
        # Fit a tranform from the best triangle        
        clip = np.argsort(dist)[:auto_keep*1]
        tri_ix = np.vstack([ix, np.arange(tri_2.shape[0])]).T[clip,:]

        # Matched point pairs
        pair_ix = _get_tri_pairs(V1, ix_1, V2, ix_2, tri_ix)
        
        tfo, dx, rms = get_transform(V1, V2, pair_ix,
                                     transform=auto_transform,
                                     use_ransac=True)
        
        # Full list of pairs
        transV1 = tfo(V1)
        tri_ix = np.vstack([ix, np.arange(tri_2.shape[0])]).T
        pair_ix = _get_tri_pairs(V1, ix_1, V2, ix_2, tri_ix)
        resid = transV1[pair_ix[:,0],:] - V2[pair_ix[:,1],:]
        
        # Small residuals
        clip = resid < 1
        pair_ix = pair_ix[np.sqrt((resid**2).sum(axis=1)) < 3,:]
    
    else:  
        # Use Nkeep best matches
        Nkeep = np.array([(dist/dist.min() < 10).sum(), V2.shape[0],
                          maxKeep]).min()

        clip = np.argsort(dist)[:Nkeep]        
        tri_ix = np.vstack([ix, np.arange(tri_2.shape[0])]).T[clip,:]
        # Matched point pairs
        pair_ix = _get_tri_pairs(V1, ix_1, V2, ix_2, tri_ix)
    
    return pair_ix
    
def cat_tri_hash(V, ba_max=0.8, size_limit=[5,800], ignore_scale=True,
                 ignore_rot=True):
    """
    Compute triangle hash values
    
    Parameters
    ----------
    V : `~numpy.ndarray` with dimensions [N,2]
        Coordinate list.  
        
    .. warning::
    The number of triangles grows rapidly with `N`, with 
          Ntri =  19600 (N=50)
          Ntri = 161700 (N=100)
          Ntri =   1.3M (N=200)
    
    ba_max : float (0.5, 1)
        Maximum allowed ratio of the second-longest to longest side of the 
        test triangles.  Set this to something less than one when you want 
        to use the position angle of the longest side as a constraint
        (`ignore_rot=True`).
    
    size_limit : float, float
        Consider triangles with the longest edge within the specified size 
        limit.
    
    ignore_scale : bool
        Use the absolute size of the sides of the triangles as constraints, 
        for when scale differences between catalogs are negligible.
    
    ignore_rot : bool
        Use the position angle of the longest edge of the triangle as a 
        constraint, for when rotation differences are negligible.
     
    .. warning::
       The matching doesn't seem to work well using only the scale-free 
       sides of the triangles (ignore_scale = ignore_rot = False).
       

    Returns
    -------
    tri_hash : [Nt, M] array
        Triangle hash values, where `Nt` is the number of available triangles
        and `M` depends on the constraints used given the "ignore" parameters
        above.
    
    tri_indices : [Nt, 3]
        Array indices of `V` of the available triangles.
        
    """
    tri = parse_triangles(V)
    clip = (tri['b']/tri['a'] < ba_max) & (tri['a'] > size_limit[0])
    clip &= (tri['a'] < size_limit[1])
    
    if ignore_scale:
        pars = [tri['a'], tri['b'], tri['c']]
    else:
        pars = [tri['b']/tri['a'], tri['c']/tri['a']]
    
    if ignore_rot:
        pars.append(tri['theta_a']/np.pi*180)
        
    tri_hash = np.vstack(pars).T[clip,:]
    
    return tri_hash, tri['indices'][clip,:]
    
def parse_triangles(V):
    """
    Parse triangles derived from a coordinate list
    
    Parameters
    ----------
    V : [N,2] array
        Input coordinate list.
    
    Returns
    -------
    params : dict
        Triangle parameters 'centroid', 'area', 'a', 'b', 'c', 'theta_a', and 
        'indices'.  Paramters 'a','b','c' are the sorted lengths of the 
        triangle sides and 'theta_a' is the position angle of the long side, 
        'a'.
    
    """
    from collections import OrderedDict
    import itertools
    
    params = OrderedDict()
    
    sh = V.shape

    # split into triangles
    indices = np.array(list(itertools.combinations(range(sh[0]), 3)))
    tsh = indices.shape
    
    triV = V[indices,:]
    
    # Centroid 
    cx = triV.mean(axis=1)
    dx = triV*1.
    for i in range(3):
        dx[:,i,:] = triV[:,i,:] - cx
    
    params['centroid'] = cx
    
    # Angle of points relative to centroid
    theta = np.arctan2(dx[:,:,0], dx[:,:,1])
    so = np.argsort(theta, axis=1)
    
    # Sorted indices
    indices = np.vstack([indices[range(tsh[0]), so[:,i]] for i in range(3)]).T
    triV = V[indices,:]
        
    # Sides of the triangle
    trix = [(0, 1), (1, 2), (2, 0)]
    
    sides = triV[:,trix,:]
    dx = sides[:,:,1,:] - sides[:,:,0,:]
    side_len = np.sqrt((dx**2).sum(axis=2))
    
    # Area
    v1 = triV[:,1,:] - triV[:,0,:]
    v2 = triV[:,2,:] - triV[:,0,:]
    area = np.abs(0.5*np.cross(v1, v2))
    params['area'] = area
    
    # Sorted sides
    so = np.argsort(side_len, axis=1)
    a = side_len[np.arange(tsh[0]), so[:,2]]
    b = side_len[np.arange(tsh[0]), so[:,1]]
    c = side_len[np.arange(tsh[0]), so[:,0]]
    params['a'], params['b'], params['c'] = a, b, c
    
    # Position angle of long edge
    long_edge = sides[np.arange(tsh[0]), so[:,2], :,:]
    dx = long_edge[:,1,:] - long_edge[:,0,:]
    theta_a = np.arctan2(dx[:,0], dx[:,1])
    params['theta_a'] = theta_a
    
    # Re-sort indices so long edge last
    for i in [0,1]:
        r = np.roll(indices, 2-i, axis=1)
        ix = so[:,2] == i
        indices[ix,:] = r[ix,:]
    
    params['indices'] = indices
    triV = V[indices,:]
    
    if False:
        ii = 100
        # show a triangle
        pl = plt.plot(triV[ii,:,0], triV[ii,:,1], alpha=0.5)
        plt.scatter(triV[ii,0,0], triV[ii,0,1], marker='o',
                    color=pl[0].get_color())
        pl = plt.plot(long_edge[ii,:,0], long_edge[ii,:,1], alpha=0.5,
                      linestyle='--', color=pl[0].get_color())
    
    return params
    
def _get_tri_pairs(V1, ix_1, V2, ix_2, ix):
    """
    Compute matched pair indices.
    
    Parameters
    ----------
    V1 : [N1,2] array
        Coordinate set #1.
    
    q1 : [M, 3] array
        Triangle indices for set #1.
    
    V2 : [N1,2] array
        Coordinate set #2.
    
    q2 : [M, 3] array
        Triangle indices for set #2.   
    
    ix : [Q, 2] array
        Indices of triangles in `q1` and `q2` matched from the KDTree.
        
    Returns
    -------
    pair_ix : [P,2]
        Indices of unique matched pairs in `V1` and `V2`.
        
    """
    # Triangles already sorted by side length in both lists
    all_pairs = np.vstack([ix_1[ix[:,0],:].flatten(),
                           ix_2[ix[:,1],:].flatten()])
    pair_ix = np.unique(all_pairs, axis=1).T
    return pair_ix

def match_catalog_quads(V1, V2, normed_triangles=False, maxKeep=8,
                        size_limit=[5,800]):
    """
    Match to position lists together using quads similar to Heyl (2013).
    
    Parameters
    ----------
    V1 : [N1,2] array
        Coordinate set #1.
    
    V2 : [N2,2] array
        Coordinate set #2.
    
    normed_triangles : bool
        If True, then match on the second and third largest triangles of the 
        quad scaled by the largest triangle.  If False, then match on the 
        absolute sizes of the first three largest triangles.  The latter 
        appears to be more robust, though assumes that scale differences
        are negligible.
    
    maxKeep : int
        Max number of quads to keep, sorted by the tree match separation.
    
    size_limit : [min, max]
        Only keep quads with rough linear dimensions between min/max.
        
    Returns
    -------
    pair_ix : [M,2] array
        Indices of derived matches.
    
    .. warning::
    Note: number of quads grows rapidly with `N`, with 
          Nquad =  230300 (N=50)
          Nquad = 3921225 (N=100)


    """
    from scipy import spatial
    
    a_1, ba_1, ca_1, quads_1 = cat_quad_hash(V1)
    ok_1 = np.isfinite(a_1+ba_1+ca_1) & (a_1 > size_limit[0]**2/2) 
    ok_1 &= (a_1 < size_limit[1]**2/2.)
    
    a_2, ba_2, ca_2, quads_2 = cat_quad_hash(V2)
    ok_2 = np.isfinite(a_2+ba_2+ca_2) & (a_2 > size_limit[0]**2/2)
    ok_2 &= (a_2 < size_limit[1]**2/2.)
        
    if normed_triangles:
        # More flexibility to account for scale differences
        inp = np.vstack([ba_1, ca_1]).T[ok_1]
        out = np.vstack([ba_2, ca_2]).T[ok_2]
    else:
        # Use natural size of all three triangles, assumes scale correct
        inp = np.vstack([ba_1*a_1, ca_1*a_1, a_1]).T[ok_1]
        out = np.vstack([ba_2*a_2, ca_2*a_2, a_2]).T[ok_2]
    
    # Use scipy.spatial.cKDTree
    tree = spatial.cKDTree(inp)
    dist, ix = tree.query(out)

    # Use Nkeep best matches
    Nkeep = np.array([(dist/dist.min() < 3).sum(), V2.shape[0], maxKeep]).min()
    
    clip = np.argsort(dist)[:Nkeep]
        
    quad_ix = np.vstack([ix, np.arange(out.shape[0])]).T[clip,:]
    
    # Matched point pairs
    pair_ix = _get_quad_pairs(V1, quads_1[ok_1,:], V2, quads_2[ok_2,:], quad_ix)
    
    return pair_ix
    
def cat_quad_hash(V):
    """
    Compute the quad hash values following Heyl (2013)
    
    Parameters
    ----------
    V : `~numpy.ndarray` with dimensions [N,2]
        Coordinate list.  
        
        Note: number of quads grows rapidly with `N`, with 
              Nquad =  230300 (N=50)
              Nquad = 3921225 (N=100)
            
    Returns
    -------
    a : array
        Area of the largest triangle 
    
    ba, ca : array
        Fractional areas of the next two largest triangles 
        (fourth is not independent).
    
    quad_indices : array
        Indices of all the quads generated from the parent coordinate list
    """
    import itertools
    
    sh = V.shape
    
    # quad combinations
    quad_indices = np.array(list(itertools.combinations(range(sh[0]), 4)))
    quadV = V[quad_indices,:]

    # 4 triangles in a quad
    #trix = list(itertools.combinations(range(4), 3))
    trix = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    
    triangles = quadV[:,trix,:]
    v1 = triangles[:,:,1,:] - triangles[:,:,0,:]
    v2 = triangles[:,:,2,:] - triangles[:,:,0,:]
    areas = np.abs(0.5*np.cross(v1, v2))
    
    # Sorted areas
    ash = areas.shape
    so = np.argsort(areas, axis=1)
    a = areas[np.arange(ash[0]), so[:,3]]
    ba = areas[np.arange(ash[0]), so[:,1]]/a
    ca = areas[np.arange(ash[0]), so[:,2]]/a
    
    return a, ba, ca, quad_indices
        
def _get_quad_pairs(V1, q1, V2, q2, ix):
    """
    Compute matched pair indices.
    
    Parameters
    ----------
    V1 : [N1,2] array
        Coordinate set #1.
    
    q1 : [M, 4] array
        Quad indices for set #1.
    
    V2 : [N1,2] array
        Coordinate set #2.
    
    q2 : [M, 4] array
        Quad indices for set #2.   
    
    ix : [Q, 2] array
        Indices of quads in `q1` and `q2` matched from the KDTree.
        
    Returns
    -------
    pair_ix : [P,2]
        Indices of unique pairs in `V1` and `V2`.
    
    """
    
    from astropy.coordinates import Angle
    import astropy.units as u
    
    # Matched quads
    q1i = q1[ix[:,0],:]
    q2i = q2[ix[:,1],:]
      
    ## Need to compute angles relative to the quad centroid to match 
    ## the quad order
    sh1 = q1i.shape
    
    # Matched coordinates
    cq1 = np.zeros((sh1[0], 4, 2))
    for i in range(sh1[0]):
        cq1[i,:,:] = V1[q1i[i,:],:]
    
    # Centroid 
    cx1 = cq1.mean(axis=1)
    dx1 = cq1*0.
    for i in range(4):
        dx1[:,i,:] = cq1[:,i,:] - cx1
    
    # Angle of quad points relative to centroid
    theta1 = np.arctan2(dx1[:,:,0], dx1[:,:,1])
    #so1 = np.argsort(theta1, axis=1)
    
    # Sorted on the circle
    aa1 = np.sort(theta1, axis=1)
    aa1p = Angle(np.hstack([aa1[:,-1:], aa1])*u.radian)
    so1 = np.argsort(aa1p.diff(axis=1).wrap_at(np.pi*u.radian))
    
    # Sort indices in array
    q1is = q1i*1
    for i in range(sh1[0]):
        so1a = np.argsort(theta1[i,:])
        aa1 = theta1[i,so1a]
        aa1p = Angle(np.hstack([aa1[-1:], aa1])*u.radian)
        so1b = np.argsort(aa1p.diff().wrap_at(np.pi*u.radian))
        
        #q1is[i,:] = q1is[i,so1[i,:]]
        q1is[i,:] = q1is[i,so1a[so1b]]
    
    # Now do the same for coordinate set #2
    sh2 = q2i.shape
    cq2 = np.zeros((sh2[0], 4, 2))
    for i in range(sh2[0]):
        cq2[i,:,:] = V2[q2i[i,:],:]

    cx2 = cq2.mean(axis=1)
    dx2 = cq2*0.
    for i in range(4):
        dx2[:,i,:] = cq2[:,i,:] - cx2
    
    theta2 = np.arctan2(dx2[:,:,0], dx2[:,:,1])
    #so2 = np.argsort(theta2, axis=1)
    aa2 = np.sort(theta2, axis=1)
    aa2p = Angle(np.hstack([aa2[:,-1:], aa2])*u.radian)
    so2 = np.argsort(aa2p.diff(axis=1).wrap_at(np.pi*u.radian))
        
    # q2is = q2i*1
    # for i in range(sh2[0]):
    #     q2is[i,:] = q2is[i,so2[i,:]]
    q2is = q2i*1
    for i in range(sh2[0]):
        so2a = np.argsort(theta2[i,:])
        aa2 = theta2[i,so2a]
        aa2p = Angle(np.hstack([aa2[-1:], aa2])*u.radian)
        so2b = np.argsort(aa2p.diff().wrap_at(np.pi*u.radian))
        
        #q2is[i,:] = q2is[i,so2[i,:]]
        q2is[i,:] = q2is[i,so2a[so1b]]
                
    all_pairs = np.vstack([q1is.flatten(), q2is.flatten()])
    pair_ix = np.unique(all_pairs, axis=1).T
    
    return pair_ix

def match_diagnostic_plot(V1, V2, pair_ix, tf=None, new_figure=False):
    """
    Show the results of the pair matching from `match_catalog_quads`.
    """
    import matplotlib.pyplot as plt
    
    if new_figure:
        fig = plt.figure(figsize=[4,4])
        ax = fig.add_subplot(111)
    else:
        ax = plt.gca()
        
    # Original catalog points
    ax.scatter(V1[:,0], V1[:,1], marker='o', alpha=0.1, color='k', 
               label='V1, N={0}'.format(V1.shape[0]))
               
    ax.scatter(V2[:,0], V2[:,1], marker='o', alpha=0.1, color='r', 
               label='V2, N={0}'.format(V2.shape[0]))
    
    if tf is not None:
        # First catalog matches
        tf_mat = V1[pair_ix[:,0],:]    
        ax.plot(tf_mat[:,0], tf_mat[:,1], marker='o', alpha=0.1, 
                color='k', linewidth=2)
        
        # Transformed first catalog
        tf_mat = tf(V1[pair_ix[:,0],:])
        ax.plot(tf_mat[:,0], tf_mat[:,1], marker='o', alpha=0.8, color='k', 
                linewidth=2, label='Transform:\n'+'  shift=[{0:.2f}, {1:.2f}]\n  rotation={2:.4f}'.format(tf.translation[0], 
                tf.translation[1], tf.rotation))
        
    else:
        # First catalog matches
        tf_mat = V1[pair_ix[:,0],:]    
        ax.plot(tf_mat[:,0], tf_mat[:,1], marker='o', alpha=0.8, color='k',
                linewidth=2)
        
    # Second catalog matches
    ax.plot(V2[pair_ix[:,1],0], V2[pair_ix[:,1],1], marker='.', alpha=0.8, 
            color='r', linewidth=0.8, 
            label='{0} pairs'.format(pair_ix.shape[0]))
    
    ax.legend(fontsize=8)
    
    if new_figure:
        fig.tight_layout(pad=0.2)
        return fig
        
def get_transform(V1, V2, pair_ix, transform=None, use_ransac=True):
    """
    Estimate parameters of an `~skimage.transform` tranformation given 
    a list of coordinate matches.
    
    Parameters
    ----------
    V1, V2 : [N,2] arrays
        Coordinate lists.  The transform is applied to V1 to match V2.
    
    pair_ix : [M,2] array
        Indices of matched pairs.
        
    transform : `~skimage.transform` transformation.
        Transformation to fit to the matched pairs.  If `None`, defaults to
        `~skimage.transform.SimilarityTransform`.
    
    Returns
    -------
    tf : `transform`
        Fitted transformation.
    
    dx : [M,2] array
        X & Y differences between the transformed V1 list and V2.
    
    rms : (float, float)
        Standard deviation of the residuals in X & Y.
        
    """
    import skimage.transform
    from skimage.measure import ransac
    
    if transform is None:
        transform = skimage.transform.SimilarityTransform
    
    if use_ransac:
        tf, inliers = ransac((V1[pair_ix[:,0],:], V2[pair_ix[:,1],:]),
                               transform, min_samples=3,
                               residual_threshold=3, max_trials=100)
        
        dx = tf(V1[pair_ix[:,0],:]) - V2[pair_ix[:,1],:]
        rms = np.std(dx[inliers,:], axis=0)
    else:
        tf = transform()
        tf.estimate(V1[pair_ix[:,0],:], V2[pair_ix[:,1],:])
        
        dx = tf(V1[pair_ix[:,0],:]) - V2[pair_ix[:,1],:]
        rms = np.std(dx, axis=0)
    
    return tf, dx, rms