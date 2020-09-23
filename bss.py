import numpy as np
from scipy.stats import ortho_group
from sklearn.decomposition import FastICA

def mad(x):
    """Median absolute deviator"""

    if len(np.shape(x)) == 1:
        return np.median(abs(x - np.median(x)))/0.6735

    return np.median(abs(x - np.median(x, axis=1)[:, np.newaxis]), axis=1)/0.6735


def makeMixture(n=2, t=1024, m=2, p=0.02, s_type=1, noise_level=120, nneg=False, indep=True):
    """Make a mixture

    Parameters
    ----------
    n : int
        Number of sources
    t : int
        Number of samples
    m : int
        Number of observations
    s_type : int or np.ndarray
        Source types (1: Gaussian, 2: uniform, 3: approximately sparse)
    noise_level : float
        Noise level in dB
    nneg : bool
        Non-negative sources and mixing matrix (forces uniform source)
    indep: bool
        Independant sources

    Returns
    -------
    np.ndarray
        Observations
    np.ndarray
        Mixing matrix
    np.ndarray
        Sources
    """

    if nneg:
        s_type = 2

    if not nneg:
        A = np.random.randn(m,n)
    else:
        A = np.random.rand(m,n)

    if np.isscalar(s_type):
        s_type = s_type * np.ones(n)
    for i in range(n-len(s_type)):
        s_type.append(s_type[-1])
    s_type = s_type[:n]

    S = np.zeros((n,t))

    for i in range(n):

        if s_type[i] == 1:
            S[i, :] = np.random.randn(t)

        elif s_type[i] == 2:
            S[i, :] = np.random.rand(t) - 0.5*(1-nneg)

        elif s_type[i] == 3:
            S[i, :] = np.power(np.random.randn(t), 3)

        else:
            print('type takes an unexpected value ...')

    if not indep:
        S = ortho_group.rvs(n)@S

    S /= np.std(S, axis=1)[:, np.newaxis]

    X = np.dot(A,S)

    if noise_level < 120:
        #--- Add noise
        N = np.random.randn(m,t)
        N = 10**(-noise_level/20)*np.linalg.norm(X)/np.linalg.norm(N)*N
        X = X + N

    return X, A, S


def pca(X, n):
    """Perfom PCA
    """

    Y = X - np.diag(np.mean(X, axis=1)) @ np.ones(np.shape(X))

    _, Uy = np.linalg.eig(Y @ Y.T)

    Sy = Uy[:, :n].T @ X

    return Uy[:, :n], Sy


def fastica(X, n):
    """Perfom FastICA"""

    X = np.copy(X.T)

    fpica = FastICA()

    S = fpica.fit(X).transform(X).T  # Get the estimated sources
    A = fpica.mixing_  # Get estimated mixing matrix

    return A , S


def multiplicative_update(X, n, nbIt=100):
    """Perform the multiplicative update algorithm"""

    A = np.random.random((X.shape[0], n))
    A /= np.maximum(np.linalg.norm(A, axis=0), 1e-24)
    S = A.T @ X


    for i in range(nbIt):

        S = S*(A.T@X)/(A.T@A@S)
        A = A*(X@S.T)/(A@S@S.T)

    return A, S


def hals(Y, n=None, nbIt=100, A2S_iterationRatioVector=None, A=None, S=None, verb=False):
    """HALS blind source separation algorithm

    Parameters
    ----------
    Y : np.ndarray
        (m,p) float array, input data, each row corresponds to an observation
    n : int
        number of sources to be estimated
    nbIt : int
        number of iterations
    A2S_iterationRatioVector : np.ndarray
        (2,) int array, number of updates of A and S per iteration
    A : np.ndarray
        (m,n) float array, initialization mixing matrix (default: random)
    S : np.ndarray
        (n,p) float array, initialization sources (default: random)
    verb : int
        verbosity level, from 0 (mute) to 5 (most talkative)

    Returns
    -------
    (np.ndarray,np.ndarray)
        estimated mixing matrix ((m,n) float array),
        estimated sources ((n,p) float array)
    """

    if n is None:
        n = np.shape(A)[1]

    if A is None:
        A = np.random.random((Y.shape[0], n))
        A /= np.maximum(np.linalg.norm(A, axis=0), 1e-24)
    else:
        A = A.copy()
    if S is None:
        S = A.T @ Y
    else:
        S = S.copy()

    if A2S_iterationRatioVector is None:
        A2S_iterationRatioVector = np.array([1, 1])

    for i in range(nbIt):

        # Update A
        for k in range(0, A2S_iterationRatioVector[0]):
            hals_core(Y, A, S, n)
        if verb >= 5:
            print('A = \n', A)

        # Update S
        for k in range(0, A2S_iterationRatioVector[1]):
            hals_core(Y.T, S.T, A.T, n)

    return A, S


def hals_core(X, W, H, n):
    """HALS core in-place update.

    Parameters
    ----------
    X : np.ndarray
        (m,p) float array, input data, each row corresponds to an observation
    W : np.ndarray
        updated matrix (in-place)
    H : np.ndarray
        fixed matrix
    n : int
        number of sources

    Returns
    -------
    None
    """

    for i in range(n):

        # Generate the index array
        sumIdxArray = np.delete(np.arange(0, n), i)

        # HALS update
        h_sum = 0
        for j in sumIdxArray:
            h = np.dot(H[j:j+1, :], H[i:i+1, :].T)
            h_sum = h_sum + np.dot(W[:, j:j+1], h)

        w = (np.dot(X, H[i:i+1, :].T) - h_sum) / np.maximum(np.linalg.norm(H[i:i+1, :], 2)**2, 1e-24)

        for k in range(0, len(w)):
            if w[k] > 0:
                W[k, i] = w[k]
            else:
                W[k, i] = 0


def gmca(X, n, nmax=500, k=3, eps=1e-5, L0=True):
    """Perfom a basic version of GMCA

    Parameters
    ----------
    X : np.ndarray
        Observations
    n : int
        Number of sources
    k : float
        Thresholding parameter
    eps : float
        Convergence criteria
    L0 : bool
        Use hard-thresholding rather than soft

    Returns
    -------
    np.ndarray
        Mixing matrix
    np.ndarray
        Sources
    """

    # PCA initialization
    A, S_old = pca(X, n)
    A /= np.maximum(np.linalg.norm(A, axis=0), 1e-9)
    delta_S = np.inf

    for it in range(nmax):

        # I - Update of S with A fixed
        # 1 - Least-square update
        Ra = A.T@A
        Ua, Sa, Va = np.linalg.svd(Ra)
        Sa[Sa < np.max(Sa) * 1e-9] = np.max(Sa) * 1e-9
        iRa = Va.T@np.diag(1/Sa)@Ua.T
        piA = iRa@A.T
        S = piA @ X

        if delta_S <= eps:
            break

        # 2 - Thresholding
        stds = mad(S)
        thrds = k*stds
        if L0:
            S[np.abs(S)<thrds[:, np.newaxis]] = 0
        else:
            S = np.sign(S)*np.maximum(np.abs(S)-thrds[:, np.newaxis], 0)

        # II - Update of A with S fixed
        # 1 - Least-square update
        Rs = S@S.T
        Us, Ss, Vs = np.linalg.svd(Rs)
        Ss[Ss < np.max(Ss) * 1e-9] = np.max(Ss) * 1e-9
        iRs = np.dot(Us, np.dot(np.diag(1. / Ss), Vs))
        iRs = Vs.T@np.diag(1/Ss)@Us.T
        piS = S.T@iRs
        A = X@piS

        # 2 - Normalization of the columns of A
        A /= np.maximum(np.linalg.norm(A, axis=0), 1e-9)

        delta_S = np.sqrt(np.sum((S-S_old)**2) / np.sum(S**2))
        S_old = np.copy(S)

        if it==nmax-1:
            print("Convergence warning")

    return A, S


def gmca_getDetails(X, n, nmax=500, k=3, eps=1e-5, L0=True):

    """Perfom a basic version of GMCA (and get some extra details)

    Parameters
    ----------
    X : np.ndarray
        Observations
    n : int
        Number of sources
    k : float
        Thresholding parameter
    eps : float
        Convergence criteria
    L0 : bool
        Use hard-thresholding rather than soft

    Returns
    -------
    np.ndarray
        Mixing matrix
    np.ndarray
        Sources
    np.ndarray
        Mixing matrices along the iterations
    np.ndarray
        Sources along the iterations (before & after thresholding)
    np.ndarray
        Thresholds along the iterations
    """

    # PCA initialization
    A, S_old = pca(X, n)
    A /= np.maximum(np.linalg.norm(A, axis=0), 1e-9)
    delta_S = np.inf

    thrds_its = np.zeros((nmax, n))
    A_its = np.zeros((nmax, np.shape(A)[0], np.shape(A)[1]))
    S_its = np.zeros((nmax, 2, np.shape(S_old)[0], np.shape(S_old)[1]))

    for it in range(nmax):

        # I - Update of S with A fixed
        # 1 - Least-square update
        Ra = A.T@A
        Ua, Sa, Va = np.linalg.svd(Ra)
        Sa[Sa < np.max(Sa) * 1e-9] = np.max(Sa) * 1e-9
        iRa = Va.T@np.diag(1/Sa)@Ua.T
        piA = iRa@A.T
        S = piA @ X

        S_its[it, 0, :, :] = np.copy(S)

        if delta_S <= eps:
            break

        # 2 - Thresholding
        stds = mad(S)
        thrds = k*stds
        if L0:
            S[np.abs(S)<thrds[:, np.newaxis]] = 0
        else:
            S = np.sign(S)*np.maximum(np.abs(S)-thrds[:, np.newaxis], 0)

        S_its[it, 1, :, :] = np.copy(S)
        thrds_its[it, :] = np.copy(thrds)

        # II - Update of A with S fixed
        # 1 - Least-square update
        Rs = S@S.T
        Us, Ss, Vs = np.linalg.svd(Rs)
        Ss[Ss < np.max(Ss) * 1e-9] = np.max(Ss) * 1e-9
        iRs = np.dot(Us, np.dot(np.diag(1. / Ss), Vs))
        iRs = Vs.T@np.diag(1/Ss)@Us.T
        piS = S.T@iRs
        A = X@piS

        # 2 - Normalization of the columns of A
        A /= np.maximum(np.linalg.norm(A, axis=0), 1e-9)

        A_its[it, :, :] = np.copy(A)

        delta_S = np.sqrt(np.sum((S-S_old)**2) / np.sum(S**2))
        S_old = np.copy(S)

        if it==nmax-1:
            print("Convergence warning")

    return A, S, A_its[:it, :, :], S_its[:it, :, :, :], thrds_its[:it, :]


def corr_perm(A0, S0, A, S, inplace=False, optInd=False):
    """Correct the permutation of the solution.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    S0: np.ndarray
        (n,p) float array, ground truth sources
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    inplace: bool
        in-place update of A and S
    optInd: bool
        return permutation

    Returns
    -------
    None or np.ndarray or (np.ndarray,np.ndarray) or (np.ndarray,np.ndarray,np.ndarray)
        A (if not inplace),
        S (if not inplace),
        ind (if optInd)
    """

    nrmsA0 = np.linalg.norm(A0, axis=0)

    A0 = A0.copy()
    S0 = S0.copy()
    if not inplace:
        A = A.copy()
        S = S.copy()

    n = np.shape(A0)[1]

    for i in range(0, n):
        S[i, :] *= (1e-24 + np.linalg.norm(A[:, i]))
        A[:, i] /= (1e-24 + np.linalg.norm(A[:, i]))
        S0[i, :] *= (1e-24 + np.linalg.norm(A0[:, i]))
        A0[:, i] /= (1e-24 + np.linalg.norm(A0[:, i]))

    try:
        diff = abs(np.dot(np.linalg.inv(np.dot(A0.T, A0)), np.dot(A0.T, A)))
    except np.linalg.LinAlgError:
        diff = abs(np.dot(np.linalg.pinv(A0), A))
        print('Warning! Pseudo-inverse used.')

    ind = np.arange(0, n)

    for i in range(0, n):
        ind[i] = np.where(diff[i, :] == max(diff[i, :]))[0][0]

    A[:] = A[:, ind.astype(int)]
    S[:] = S[ind.astype(int), :]

    for i in range(0, n):
        p = np.sum(S[i, :] * S0[i, :])
        if p < 0:
            S[i, :] = -S[i, :]
            A[:, i] = -A[:, i]

    S /= nrmsA0[:, np.newaxis]
    A *= nrmsA0

    if inplace and not optInd:
        return None
    elif inplace and optInd:
        return ind
    elif not optInd:
        return A, S
    else:
        return A, S, ind


def nmse(S0, S):
    """Compute the normalized mean square error (NMSE) in dB.

    Parameters
    ----------
    S0: np.ndarray
        (n,p) float array, ground truth sources
    S: np.ndarray
        (n,p) float array, estimated sources

    Returns
    -------
    float
        NMSE (dB)
    """
    return -10 * np.log10(np.sum((S0-S)**2)/np.sum(S0**2))


def ca(A0, A):
    """Compute the criterion on A (CA) in dB.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix

    Returns
    -------
    float
        CA (dB)
    """
    return -10 * np.log10(np.mean(np.abs(np.dot(np.linalg.pinv(A), A0) - np.eye(np.shape(A0)[1]))))
