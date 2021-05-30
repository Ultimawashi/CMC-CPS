import numpy as np
import networkx as nx
from tools import gauss, add_border, crop_border


def calc_proba_champs(alpha, reg_seuil=10 ** -10):
    assert alpha.sum() <= reg_seuil, 'veuillez donner une matrice alpha qui somme a 0'
    proba = np.zeros((5, 2))
    for i in range(2):
        for j in range(5):
            proba[j, i] = np.exp((4 - j) * alpha[0, i] + j * alpha[1, i]) / (
                        np.exp((4 - j) * alpha[0, 0] + j * alpha[1, 0]) + np.exp(
                    (4 - j) * alpha[0, 1] + j * alpha[1, 1]))
    return proba


def calc_proba_champs_apost(proba, gauss, reg=10 ** -10):
    return proba[:, :] * gauss[np.newaxis, :] / ((proba[:, :] * gauss[np.newaxis, :]).sum(axis=-1) + reg)[
        ..., np.newaxis]


def config_voisinage(I, i, j):
    I_sub = I[(i - 1):(i + 2), (j - 1):(j + 2)]
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    idx_vos = (mask * (I_sub == 1)).sum()
    idx_class = (I[i, j]==1)*1
    return idx_vos, idx_class


def init_Gibbs(shape):
    shape = tuple(x + 2 for x in shape)
    p_tmp = 0.5
    x_init = np.random.binomial(1, p_tmp, shape)
    return x_init


def iter_Gibbs_proba(proba, x_init):
    x_new = x_init
    for i in range(1, x_init.shape[0] - 1):
        for j in range(1, x_init.shape[1] - 1):
            idx_vos, _ = config_voisinage(x_new, i, j)
            aux = np.random.multinomial(1, proba[idx_vos, :])
            x_new[i, j] = np.argmax(aux)
    return x_new


def genere_Gibbs_proba(proba, x_init, nb_iter):
    x = np.zeros(x_init.shape + (nb_iter,))
    x[:, :, 0] = x_init
    for i in range(1, nb_iter):
        x[:, :, i] = iter_Gibbs_proba(proba, x[:, :, i - 1])
    return crop_border(x[:, :, -1])


def iter_Gibbs_proba_apost_gauss(proba, gausses, x_init):
    x_new = x_init
    for i in range(1, x_init.shape[0] - 1):
        for j in range(1, x_init.shape[1] - 1):
            idx_vos, _ = config_voisinage(x_new, i, j)
            p_apost = calc_proba_champs_apost(proba, gausses[i - 1, j - 1])
            aux = np.random.multinomial(1, p_apost[idx_vos, :])
            x_new[i, j] = np.argmax(aux)
    return x_new


def genere_Gibbs_proba_apost(signal_2d_noisy, proba, m1, sig1, m2, sig2, nb_iter):
    gausses = gauss(signal_2d_noisy, m1, sig1, m2, sig2)
    x_init = init_Gibbs(signal_2d_noisy.shape)
    x = np.zeros(x_init.shape + (nb_iter,))
    x[:, :, 0] = x_init
    for i in range(1, nb_iter):
        x[:, :, i] = iter_Gibbs_proba_apost_gauss(proba, gausses, x[:, :, i - 1])
    return crop_border(x[:, :, -1])


def mpm_mf(signal_2d_noisy, proba, m1, sig1, m2, sig2, nb_iter, nb_simu):
    x_simu = np.zeros(signal_2d_noisy.shape + (nb_simu,))
    for h in range(nb_simu):
        x_simu[:, :, h] = genere_Gibbs_proba_apost(signal_2d_noisy, proba, m1, sig1, m2, sig2, nb_iter)
    est_proba_apost = x_simu.sum(axis=-1) / x_simu.shape[-1]
    return np.where(est_proba_apost >= (1 - est_proba_apost), 1, 0)


def estim_param_gradient_sup(signal_2d, signal_2d_noisy, iter_grad, iter_gibbs, alpha_init, lr=0.001):
    m1 = ((signal_2d == 0) * signal_2d_noisy).sum() / (signal_2d == 0).sum()
    sig1 = np.sqrt(((signal_2d == 0) * ((signal_2d_noisy - m1) ** 2)).sum() / (signal_2d == 0).sum())
    m2 = ((signal_2d == 1) * signal_2d_noisy).sum() / (signal_2d == 1).sum()
    sig2 = np.sqrt(((signal_2d == 1) * ((signal_2d_noisy - m2) ** 2)).sum() / (signal_2d == 1).sum())
    alpha = alpha_init
    for i in range(iter_grad):
        proba = calc_proba_champs(alpha)
        x_i = genere_Gibbs_proba(proba, add_border(signal_2d), iter_gibbs)
        alpha = alpha + (lr / (i + 1)) * (calc_energy_d_sup(x_i, alpha) - calc_energy_d_sup(signal_2d, alpha))
        print({'iter': i, 'alpha': alpha, 'proba': calc_proba_champs(alpha), 'm1': m1, 'sig1': sig1, 'm2': m2,
               'sig2': sig2})
    return alpha, m1, sig1, m2, sig2


def estim_param_pseudol_sup(signal_2d, signal_2d_noisy):
    m1 = ((signal_2d == 0) * signal_2d_noisy).sum() / (signal_2d == 0).sum()
    sig1 = np.sqrt(((signal_2d == 0) * ((signal_2d_noisy - m1) ** 2)).sum() / (signal_2d == 0).sum())
    m2 = ((signal_2d == 1) * signal_2d_noisy).sum() / (signal_2d == 1).sum()
    sig2 = np.sqrt(((signal_2d == 1) * ((signal_2d_noisy - m2) ** 2)).sum() / (signal_2d == 1).sum())
    proba = estim_proba_apri(signal_2d)
    return proba, m1, sig1, m2, sig2


def calc_param_EM_gibbsien_gauss(signal_2d_noisy, proba, m1, sig1, m2, sig2, iter_gibbs, nb_simu):
    x_simu = np.zeros(signal_2d_noisy.shape + (nb_simu,))
    for h in range(nb_simu):
        x_simu[:, :, h] = genere_Gibbs_proba_apost(signal_2d_noisy, proba, m1, sig1, m2, sig2, iter_gibbs)
    est_proba_apost_1 = x_simu.sum(axis=-1) / x_simu.shape[-1]
    est_proba_apost_0 = 1-est_proba_apost_1
    est_proba_v = (1 / nb_simu) * sum([estim_proba_apri(x_simu[:, :, h]) for h in range(nb_simu)])
    m1 = (est_proba_apost_0 * signal_2d_noisy).sum() / est_proba_apost_0.sum()
    sig1 = np.sqrt(
        (est_proba_apost_0 * ((signal_2d_noisy - m1) ** 2)).sum() / est_proba_apost_0.sum())
    m2 = (est_proba_apost_1 * signal_2d_noisy).sum() / est_proba_apost_1.sum()
    sig2 = np.sqrt((est_proba_apost_1 * ((signal_2d_noisy - m2) ** 2)).sum() / est_proba_apost_1.sum())
    return est_proba_v, m1, sig1, m2, sig2


def estim_param_EM_gibbsien_gauss(signal_2d_noisy, proba, m1, sig1, m2, sig2, nb_iter, iter_gibbs, nb_simu):
    proba_est = proba
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(nb_iter):
        proba_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_EM_gibbsien_gauss(signal_2d_noisy, proba_est, m1,
                                                                                     sig1, m2, sig2, iter_gibbs,
                                                                                     nb_simu)
        print({'iter': i, 'proba': proba_est, 'm1': m1_est, 'sig1': sig1_est, 'm2': m2_est, 'sig2': sig2_est})

    return proba_est, m1_est, sig1_est, m2_est, sig2_est


def calc_energy_d_sup(x, alpha):
    aux = np.moveaxis(np.indices(alpha.shape), 0, -1)
    res = (np.all(get_cliques(x)[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1).sum(axis=0))
    agrad = res[0, 0] - res[1, 0]
    bgrad = res[0, 1] - res[1, 0]
    cgrad = res[1, 1] - res[1, 0]
    return np.array([[agrad, bgrad], [-agrad - bgrad - cgrad, cgrad]])


def get_cliques(x):
    G = nx.grid_2d_graph(x.shape[0], x.shape[1])
    cl = nx.enumerate_all_cliques(G)
    cl_keep = [item for item in list(cl) if len(item) > 1]
    cl_keep_1 = np.array([list(item[0]) for item in cl_keep])
    cl_keep_2 = np.array([list(item[1]) for item in cl_keep])
    return np.stack((x[cl_keep_1[:, 0], cl_keep_1[:, 1]], x[cl_keep_2[:, 0], cl_keep_2[:, 1]]), axis=-1)


def estim_proba_apri(x):
    p_apri = np.zeros((5, 2))
    for i in range(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            idx_vos, idx_class = config_voisinage(x, i, j)
            p_apri[idx_vos, idx_class] = p_apri[idx_vos, idx_class] + 1
    p_apri = p_apri / p_apri.sum(axis=-1)[..., np.newaxis]
    return p_apri
