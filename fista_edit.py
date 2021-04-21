import tensorly as tl
def fista(UtM, UtU, x=None, n_iter_max=100, non_negative=True, sparsity_coef=0,
          lr=10e-3, tol=10e-2):
    """
    Fast Iterative Shrinkage Thresholding Algorithm (FISTA)

    Computes an approximate (nonnegative) solution for Ux=M linear system.

    Parameters
    ----------
    UtM : ndarray
        Pre-computed product of the transposed of U and M
    UtU : ndarray
        Pre-computed product of the transposed of U and U
    x : init
       Default: None
    n_iter_max : int
        Maximum number of iteration
        Default: 100
    non_negative : bool, default is False
                   if True, result will be non-negative
    lr : float
        learning rate
    sparsity_coef : float or None
    tol : float
        stopping criterion

    Returns
    -------
    x : approximate solution such that Ux = M

    Reference
    ----------
    [1] : Beck, A., & Teboulle, M. (2009). A fast iterative
          shrinkage-thresholding algorithm for linear inverse problems.
          SIAM journal on imaging sciences, 2(1), 183-202.
    """
    if sparsity_coef is None:
        sparsity_coef = 0
    
    if x is None:
        x = tl.zeros(tl.shape(UtM))

    # Parameters
    momentum_old = tl.tensor(1.0)
    norm_0 = 0.0
    x_update = tl.copy(x)

    for iteration in range(n_iter_max):
        x_gradient = - UtM + tl.dot(UtU, x_update) + sparsity_coef

        if non_negative is True:
            x_gradient = tl.where(lr * x_gradient < x_update, x_gradient, x_update/lr)

        x_new = x_update - lr * x_gradient
        momentum = (1 + tl.sqrt(1 + 4 * momentum_old ** 2)) / 2
        x_update = x_new + ((momentum_old - 1) / momentum) * (x_new - x)
        momentum_old = momentum
        x = tl.copy(x_new)
        norm = tl.norm(lr * x_gradient)
        if iteration == 1:
            norm_0 = norm
        if norm < tol * norm_0:
            break
    return x