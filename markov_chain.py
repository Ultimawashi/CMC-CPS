import numpy as np
from tools import gauss

def gauss(Y, m1, sig1, m2, sig2):
    """
    :param Y: Le signal bruité
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: tableau de valeurs des densité gaussiennes pour chaque élément du signal bruité
    """
    gauss1 = (1 / (sig1 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((Y - m1) / sig1) ** 2))
    gauss2 = (1 / (sig2 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((Y - m2) / sig2) ** 2))
    return np.stack((gauss1, gauss2), axis=-1)




def forward(A, p, gauss):
    """
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probabilité d'apparition a priori pour chaque classe
    :param gauss: tableau de valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les forward (de 1 à n)
    """
    forward = np.zeros((len(gauss), 2))
    forward[0] = p * gauss[0]
    forward[0] = forward[0] / (forward[0].sum())
    for l in range(1, len(gauss)):
        forward[l] = gauss[l] * (forward[l - 1] @ A)
        forward[l] = forward[l] / forward[l].sum()
    return forward


def backward(A, gauss):
    """
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probabilité d'apparition a priori pour chaque classe
    :param gauss: tableau de valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: un vecteur de taille la longueur de la chaîne, contenant tous les backward (de 1 à n).
    Attention, si on calcule les backward en partant de la fin de la chaine, je conseille quand même d'ordonner le vecteur backward du début à la fin
    """
    backward = np.zeros((len(gauss), 2))
    backward[len(gauss) - 1] = np.ones(2)
    for k in reversed(range(0, len(gauss)-1)):
        backward[k] = A @ (backward[k + 1] * gauss[k + 1])
        backward[k] = backward[k] / (backward[k].sum())
    return backward









def mpm_mc(Y, w, p, A, m1, sig1, m2, sig2):
    """
    :param Y: Signal bruité
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: signal segmenté à deux classes
    """
    gausses = gauss(Y, m1, sig1, m2, sig2)
    alpha = forward(A,p,gausses)
    beta = backward(A,gausses)
    proba_apost = alpha * beta
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    return w[np.argmax(proba_apost, axis=1)]


def calc_probaprio_mc(signal, w):
    """
    Cete fonction permet de calculer les probabilité a priori des classes w1 et w2 et les transitions a priori d'une classe à l'autre,
    en observant notre signal non bruité
    :param signal: Signal discret non bruité à deux classes (numpy array 1D d'int)
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :return: un vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    """
    p1 = np.sum((signal==w[0]))/signal.shape[0]
    p2 = np.sum((signal == w[1]))/signal.shape[0]
    p= np.array([p1,p2])
    C = sum([np.array([[(signal[i]==k and signal[i+1]==l) for l in w] for k in w]) for i in range(signal.shape[0]-1)])/signal.shape[0]
    A = (C.T/p).T
    return p, A


def simu_mc(n, w, p, A):
    """
    Cette fonction permet de simuler un signal discret à 2 classe de taille n à partir des probabilité d'apparition des deux classes et de la Matrice de transition
    :param n: taille du signal
    :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    simu = np.zeros((n,), dtype=int)
    aux = np.random.multinomial(1, p)
    simu[0] = w[np.argmax(aux)]
    for i in range(1, n):
        aux = np.random.multinomial(1, A[np.where(w == simu[i - 1])[0][0], :])
        simu[i] = w[np.argmax(aux)]
    return simu



def estim_param_sup(X, Y, w):
    """
   :param X: Signal d'origine
   :param Y: Signal bruité
   :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
   :return: Nouveaux paramètres estimé
       """
    p1 = np.sum((X == w[0])) / X.shape[0]
    p2 = np.sum((X == w[1])) / X.shape[0]
    p = np.array([p1, p2])
    C = sum([np.array([[(X[i] == k and X[i + 1] == l) for l in w] for k in w]) for i in
             range(X.shape[0] - 1)]) / X.shape[0]
    A = (C.T / p).T
    m1 = ((X == w[0]) * Y).sum() / (X == w[0]).sum()
    sig1 = np.sqrt(((X == w[0]) * ((Y - m1) ** 2)).sum() / (X == w[0]).sum())
    m2 = ((X == w[1]) * Y).sum() / (X == w[1]).sum()
    sig2 = np.sqrt(((X == w[1]) * ((Y - m2) ** 2)).sum() / (X == w[1]).sum())
    return p, A, m1, sig1, m2, sig2



def simu_mc_nonstat(n,  p, A):
    """
        Cette fonction permet de simuler un signal discret à 2 classe de taille n à partir des probabilité d'apparition des deux classes et de la Matrice de transition
        :param n: taille du signal
        :param w: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
        :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
        :param A: Matrice (2*2) de transition de la chaîne
        :return: Un signal discret à 2 classe (numpy array 1D d'int)
        """
    simu = np.zeros((n,), dtype=int)
    aux = np.random.multinomial(1, p)
    simu[0] = np.argmax(aux)
    for i in range(1, n):
        aux = np.random.multinomial(1, A[i-1, simu[i - 1], :])
        simu[i] = np.argmax(aux)
    return simu


def calc_param_EM_mc(Y, p, A, m1, sig1, m2, sig2):
    """
    :param Y: Signal bruité
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: Nouveaux paramètres estimé pour une itération de EM
    """
    gausses = gauss(Y, m1, sig1, m2, sig2)
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)
    proba_apost = alpha * beta
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    p = proba_apost.sum(axis=0)/proba_apost.shape[0]
    proba_c_apost = (
            alpha[:-1, :, np.newaxis]
            * (gausses[1:, np.newaxis, :]
               * beta[1:, np.newaxis, :]
               * A[np.newaxis, :, :])
    )
    proba_c_apost = proba_c_apost / (proba_c_apost.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
    A = np.transpose(np.transpose((proba_c_apost.sum(axis=0))) / (proba_apost[:-1:].sum(axis=0)))
    m1 = (proba_apost[:,0] * Y).sum()/proba_apost[:,0].sum()
    sig1 = np.sqrt((proba_apost[:,0]*((Y - m1) ** 2)).sum()/proba_apost[:,0].sum())
    m2 = (proba_apost[:, 1] * Y).sum() / proba_apost[:, 1].sum()
    sig2 = np.sqrt((proba_apost[:, 1] * ((Y - m2) ** 2)).sum() / proba_apost[:, 1].sum())
    return p, A, m1, sig1, m2, sig2


def estim_param_EM_mc(iter, Y, p, A, m1, sig1, m2, sig2):
    """
    :param iter: Nombre d'itération choisie
    :param Y: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de la variance de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de la variance de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme EM donc p, A, m1, sig1, m2, sig2
    """
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_mc(Y, p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)
        print({'iter':i,'p':p_est, 'A':A_est, 'm1':m1_est, 'sig1':sig1_est, 'm2':m2_est, 'sig2':sig2_est})
    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est


def calc_param_SEM_mc(signal_noisy, p, A, m1, sig1, m2, sig2):
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
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)
    proba_init = alpha[0] * beta[0]
    proba_init = proba_init / proba_init.sum()
    tapost = (
        ((gausses[1:, np.newaxis, :])
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


def estim_param_SEM_mc(iter, signal_noisy, p, A, m1, sig1, m2, sig2):
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
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_SEM_mc(signal_noisy,
                                                                            p_est, A_est, m1_est, sig1_est, m2_est,
                                                                            sig2_est)
        print({'iter':i,'p': p_est, 'A': A_est, 'm1': m1_est, 'sig1': sig1_est, 'm2': m2_est, 'sig2': sig2_est})

    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est

