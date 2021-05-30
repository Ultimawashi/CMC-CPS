import numpy as np
from tools import gauss, gauss2
from math import sqrt, pi
from markov_chain import simu_mc, simu_mc_nonstat, calc_probaprio_mc

def forward_neigh(A, p, gauss, g2, g3):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs forward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les forward (de 1 à n)
    """
    proba2 = A @ g2[0]
    proba3 = A @ g3[0]
    forward = np.zeros((len(gauss), 2))
    forward[0] = p * (gauss[0]*proba2*proba3)
    forward[0] = forward[0] / (forward[0].sum())
    for l in range(1, len(gauss)):
        proba2 = A@g2[l]
        proba3 = A@g3[l]
        forward[l] = (gauss[l]*proba2*proba3) * (forward[l - 1] @ A)
        forward[l] = forward[l] / forward[l].sum()
    return forward


def backward_neigh(A, gauss, g2, g3):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs backward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les backward (de 1 à n).
    Attention, si on calcule les backward en partant de la fin de la chaine, je conseille quand même d'ordonner le vecteur backward du début à la fin
    """
    backward = np.zeros((len(gauss), 2))
    backward[len(gauss) - 1] = np.ones(2)
    backward[len(gauss) - 1] = backward[len(gauss) - 1] / (backward[len(gauss) - 1].sum())
    for k in reversed(range(0, len(gauss)-1)):
        proba2 = A @ g2[k+1]
        proba3 = A @ g3[k+1]
        backward[k] = A @ (backward[k + 1] * (gauss[k + 1]*proba2*proba3))
        backward[k] = backward[k] / (backward[k].sum())
    return backward


def mpm_mc_neigh(signal_noisy, neighboursh, neighboursv, w, p, A, m1, sig1, m2, sig2):
    """
     Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    g2 = gauss2(neighboursh, m1, sig1, m2, sig2)
    g3 = gauss2(neighboursv, m1, sig1, m2, sig2)
    alpha = forward_neigh(A, p, gausses, g2, g3)
    beta = backward_neigh(A, gausses, g2, g3)
    proba_apost = alpha * beta
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    return w[np.argmax(proba_apost, axis=1)]


def calc_param_EM_mc_neigh(signal_noisy, neighboursh, neighboursv, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, A, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    g2 = gauss2(neighboursh, m1, sig1, m2, sig2)
    g3 = gauss2(neighboursv, m1, sig1, m2, sig2)
    proba2 = np.einsum('ij,kj->ki',A,g2)
    proba3 = np.einsum('ij,kj->ki',A,g3)
    alpha = forward_neigh(A, p, gausses, g2, g3)
    beta = backward_neigh(A, gausses, g2, g3)
    proba_apost = alpha * beta
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    p = proba_apost.sum(axis=0)/proba_apost.shape[0]
    proba_c_apost = (
            alpha[:-1, :, np.newaxis]
            * ((gausses[1:, np.newaxis, :]*proba2[1:, np.newaxis, :]*proba3[1:, np.newaxis, :])
               * beta[1:, np.newaxis, :]
               * A[np.newaxis, :, :])
    )
    proba_c_apost = proba_c_apost / (proba_c_apost.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
    A = np.transpose(np.transpose((proba_c_apost.sum(axis=0))) / (proba_apost[:-1:].sum(axis=0)))
    m1 = (proba_apost[:,0] * signal_noisy).sum()/proba_apost[:,0].sum()
    sig1 = np.sqrt((proba_apost[:,0]*((signal_noisy-m1)**2)).sum()/proba_apost[:,0].sum())
    m2 = (proba_apost[:, 1] * signal_noisy).sum() / proba_apost[:, 1].sum()
    sig2 = np.sqrt((proba_apost[:, 1] * ((signal_noisy - m2) ** 2)).sum() / proba_apost[:, 1].sum())
    return p, A, m1, sig1, m2, sig2


def estim_param_EM_mc_neigh(iter, signal_noisy, neighboursh, neighboursv, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de l'écart type de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de l'écart type de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, A, m1, sig1, m2, sig2
    """
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_mc_neigh(signal_noisy, neighboursh, neighboursv,
                                                                            p_est, A_est, m1_est, sig1_est, m2_est,
                                                                            sig2_est)
        print({'iter':i,'p': p_est, 'A': A_est, 'm1': m1_est, 'sig1': sig1_est, 'm2': m2_est, 'sig2': sig2_est})

    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est


def calc_param_SEM_mc_neigh(signal_noisy, neighboursh, neighboursv, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de EM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, A, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    g2 = gauss2(neighboursh, m1, sig1, m2, sig2)
    g3 = gauss2(neighboursv, m1, sig1, m2, sig2)
    proba2 = np.einsum('ij,kj->ki',A,g2)
    proba3 = np.einsum('ij,kj->ki',A,g3)
    alpha = forward_neigh(A, p, gausses, g2, g3)
    beta = backward_neigh(A, gausses, g2, g3)
    proba_init = alpha[0] * beta[0]
    proba_init = proba_init / proba_init.sum()
    tapost = (
        ((gausses[1:, np.newaxis, :]*proba2[1:, np.newaxis, :]*proba3[1:, np.newaxis, :])
         * beta[1:, np.newaxis, :]
         * A[np.newaxis, :, :])
    )
    tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
    signal = simu_mc_nonstat(signal_noisy.shape[0], proba_init, tapost)
    p,A = calc_probaprio_mc(signal, np.array([0,1]))
    m1 = ((signal==0) * signal_noisy).sum()/(signal==0).sum()
    sig1 = np.sqrt(((signal==0)*((signal_noisy-m1)**2)).sum()/(signal==0).sum())
    m2 = ((signal==1) * signal_noisy).sum()/(signal==1).sum()
    sig2 = np.sqrt(((signal == 1) * ((signal_noisy - m2) ** 2)).sum() / (signal == 1).sum())
    return p, A, m1, sig1, m2, sig2


def estim_param_SEM_mc_neigh(iter, signal_noisy, neighboursh, neighboursv, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme EM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de l'écart type de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de l'écart type de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, A, m1, sig1, m2, sig2
    """
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_SEM_mc_neigh(signal_noisy, neighboursh, neighboursv,
                                                                            p_est, A_est, m1_est, sig1_est, m2_est,
                                                                            sig2_est)
        print({'iter':i,'p': p_est, 'A': A_est, 'm1': m1_est, 'sig1': sig1_est, 'm2': m2_est, 'sig2': sig2_est})

    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est


