import numpy as np
from math import log2, sqrt, pi


def add_border(img, size=1):
    new_img = np.zeros(tuple(x + 2*size for x in img.shape))
    new_img[size:-size,size:-size] = img
    return new_img


def crop_border(img, size=1):
    return img[size:-size,size:-size]


def bruit_gauss(signal, w, m1, sig1, m2, sig2):
    """
    Cette fonction permet de bruiter un signal discret à deux classes avec deux gaussiennes
    :param signal: Le signal a bruiter (un numpy array d'int)
    :param w: vecteur dont la première composante est la valeur de la classe 1 et la deuxième est la valeur de la classe 2
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: le signal bruité (numpy array de float)
    """
    signal_noisy = (signal == w[0]) * np.random.normal(m1, sig1, signal.shape) + \
                   (signal == w[1]) * np.random.normal(m2, sig2, signal.shape)
    return signal_noisy


def gauss(signal_noisy, m1, sig1, m2, sig2):
    """
    Cette fonction transforme le signal bruitée en appliquant à celui-ci deux densitées gausiennes
    :param signal_noisy: Le signal bruité (numpy array 1D)
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément de signal noisy
    """
    gauss1 = (1 / (sig1 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((signal_noisy - m1) / sig1) ** 2))
    gauss2 = (1 / (sig2 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((signal_noisy - m2) / sig2) ** 2))
    return np.stack((gauss1, gauss2), axis=-1)


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def heaviside_np(x):
    thresold = np.max(x)/2
    return (x < thresold) * 0 + (x >= thresold)


def thresolding_np(x, nb_class):
    thresold = np.max(x)/(2*(nb_class-1))
    first = [(x < thresold) * 0]
    last = [(x >= (nb_class*thresold))*(nb_class-1)]
    res = first + [np.logical_and(x >= ((i)*thresold ) , x<((i+2)*thresold ))*i for i in range(1,(nb_class-1))] + last
    return sum(res)


def standardize_np(x):
    return (x-np.mean(x))/np.std(x)


def gauss2(neighbours, m1, sig1, m2, sig2):
    # fonction qui crée une liste contenant la valeur de la densité de chaque gaussienne
    # Lorsque la valeur est 5 on remplace la densité par 1, pour ignorer les voisins qui n'existent pas
    gaussthingy = np.ones((len(neighbours), 2))
    for i in range(len(gaussthingy)):
        if neighbours[i][0] != 5:
            gaussthingy[i][0] = (1 / (sig1 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((neighbours[i][0] - m1) / sig1) ** 2))
            gaussthingy[i][1] = (1 / (sig2 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((neighbours[i][0] - m2) / sig2) ** 2))
    return gaussthingy


def calc_erreur(signal1, signal2):
    """
    Cette fonction permet de mesurer la difference entre deux signaux discret (de même taille) à deux classes
    :param signal1: le premier signal, un numpy array
    :param signal2: le deuxième signal, un numpy array
    :return: La différence entre les deux signaux (un float)
    """
    erreur= np.sum(signal1 != signal2) / np.prod(signal2.shape)
    if erreur <0.5:
        return erreur
    else:
        return 1-erreur


def get_peano_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée (dont la dimension est une puissance de 2)
    selon la courbe de Hilbert-Peano
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours de Hilbert-Peano
    """
    assert log2(dSize).is_integer(), 'veuillez donne une dimension étant une puissance de 2'
    xTmp = 0
    yTmp = 0
    dirTmp = 0
    dirLookup = np.array(
        [[3, 0, 0, 1], [0, 1, 1, 2], [1, 2, 2, 3], [2, 3, 3, 0], [1, 0, 0, 3], [2, 1, 1, 0], [3, 2, 2, 1],
         [0, 3, 3, 2]]).T
    dirLookup = dirLookup + np.array(
        [[4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [0, 4, 4, 0], [0, 4, 4, 0], [0, 4, 4, 0],
         [0, 4, 4, 0]]).T
    orderLookup = np.array(
        [[0, 2, 3, 1], [1, 0, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0], [1, 3, 2, 0], [3, 2, 0, 1], [2, 0, 1, 3],
         [0, 1, 3, 2]]).T
    offsetLookup = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    for i in range(int(log2(dSize))):
        xTmp = np.array([(xTmp - 1) * 2 + offsetLookup[0, orderLookup[0, dirTmp]] + 1,
                         (xTmp - 1) * 2 + offsetLookup[0, orderLookup[1, dirTmp]] + 1,
                         (xTmp - 1) * 2 + offsetLookup[0, orderLookup[2, dirTmp]] + 1,
                         (xTmp - 1) * 2 + offsetLookup[0, orderLookup[3, dirTmp]] + 1])

        yTmp = np.array([(yTmp - 1) * 2 + offsetLookup[1, orderLookup[0, dirTmp]] + 1,
                         (yTmp - 1) * 2 + offsetLookup[1, orderLookup[1, dirTmp]] + 1,
                         (yTmp - 1) * 2 + offsetLookup[1, orderLookup[2, dirTmp]] + 1,
                         (yTmp - 1) * 2 + offsetLookup[1, orderLookup[3, dirTmp]] + 1])

        dirTmp = np.array([dirLookup[0, dirTmp], dirLookup[1, dirTmp], dirLookup[2, dirTmp], dirLookup[3, dirTmp]])

        xTmp = xTmp.T.flatten()
        yTmp = yTmp.T.flatten()
        dirTmp = dirTmp.flatten()

    x = - xTmp
    y = - yTmp
    return x, y


def peano_transform_img(img):
    """
    Cette fonction prend une image carrée (dont la dimension est une puissance de 2) en entrée,
    et retourne l'image applatie (1 dimension) selon le parcours de Hilbert-Peano
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0] == img.shape[1], 'veuillez donner une image carrée en entrée'
    assert log2(img.shape[0]).is_integer(), 'veuillez donne rune image dont la dimension est une puissance de 2'
    idx = get_peano_index(img.shape[0])
    return img[idx[0], idx[1]]


def get_peano_neighbours_index(x, y):
    # fonction qui récupère les indices des voisins hors chaînes de markov
    indexes = []
    tmp = []
    comptmp = []
    comptmp.append((x[1], y[1]))
    n = int(np.sqrt(len(x))) - 1

    if (1, 0) in comptmp:
        tmp.append([0, 1])
    else:
        tmp.append([1, 0])
    indexes.append(tmp)

    for i in range(1, len(x) - 1):
        tmp = []
        comptmp = []
        comptmp.append([x[i - 1], y[i - 1]])
        comptmp.append([x[i + 1], y[i + 1]])
        if [x[i] + 1, y[i]] not in comptmp and x[i] != n:
            tmp.append([x[i] + 1, y[i]])
        if [x[i] - 1, y[i]] not in comptmp and x[i] != 0:
            tmp.append([x[i] - 1, y[i]])
        if [x[i], y[i] + 1] not in comptmp and y[i] != n:
            tmp.append([x[i], y[i] + 1])
        if [x[i], y[i] - 1] not in comptmp and y[i] != 0:
            tmp.append([x[i], y[i] - 1])
        indexes.append(tmp)

    tmp = []
    comptmp = []
    comptmp.append([x[n - 1], y[n - 1]])
    if [0, n - 1] in comptmp:
        tmp.append([0, n - 1])
    else:
        tmp.append([1, n])
    indexes.append(tmp)
    return indexes


def peano_to_neighbours(img):
    # fonction qui prend en argument une image et qui renvoie deux listes constituées des voisins hors chaîne de markov
    # Lorsqu'un pixel ne possède pas 2 voisins hors chaîne de Markov, on remplace les voisins par la valeur 5
    (x, y) = get_peano_index(img.shape[0])

    indexes = get_peano_neighbours_index(x, y)
    neighbours = []
    for i in range(len(indexes)):
        tmp = np.full(2, 5, dtype='float64')
        t = 0
        for j in indexes[i]:
            tmp[t] = img[j[0], j[1]]
            t += 1
        neighbours.append(tmp)

    neighbours = np.array(neighbours)
    neighboursh, neighboursv = np.hsplit(neighbours, 2)

    return neighboursh, neighboursv


def transform_peano_in_img(signal, dSize):
    """
    Cette fonction prend un signal 1D en entrée et une taille, et le transforme en image carrée 2D selon le parcours de Hilbert-Peano
    :param img: un signal 1D
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une image (donc un numpy array 2 dimensions)
    """
    assert dSize == int(sqrt(signal.shape[0])), 'veuillez donner un signal ayant pour dimension dSize^2'
    idx = get_peano_index(dSize)
    img = np.zeros((dSize, dSize))
    img[idx[0], idx[1]] = signal
    return img


def get_line_index(dSize):
    """
    Cette fonction permet d'obtenir l'ordre de parcours des pixels d'une image carrée selon un parcours ligne par ligne
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction ne fonctionne qu'avec des images carrées)
    :return: une liste de taille 2*dSize*dSize qui correspond aux coordonnées de chaque pixel ordonnée selon le parcours ligne par ligne
    """
    return [a.flatten() for a in np.indices((dSize, dSize))]


def line_transform_img(img):
    """
    Cette fonction prend une image carrée en entrée, et retourne l'image applatie (1 dimension) selon le parcours ligne par ligne
    :param img: une image (donc un numpy array 2 dimensions)
    :return: un numpy array 1 dimension
    """
    assert img.shape[0] == img.shape[1], 'veuillez donner une image carrée en entrée'
    idx = get_line_index(img.shape[0])
    return img[idx[0], idx[1]]


def transform_line_in_img(signal, dSize):
    """
    Cette fonction prend un signal 1D en entrée et une taille, et le transforme en image carrée 2D selon le parcours ligne par ligne
    :param img: un signal 1D
    :param dSize: largeur ou longueur de l'image en pixel (peu importe, la fonction de fonctionne qu'avec des images carrées)
    :return: une image (donc un numpy array 2 dimensions)
    """
    assert dSize == int(sqrt(signal.shape[0])), 'veuillez donner un signal ayant pour dimension dSize^2'
    idx = get_line_index(dSize)
    img = np.zeros((dSize, dSize))
    img[idx[0], idx[1]] = signal
    return img
