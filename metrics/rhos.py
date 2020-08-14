from weighted_lstar.PDFA import PDFA

def rho_infty_norm(M, N, w=-1, qM=-1, qN=-1):

    # if dealing with an RNN
    if not isinstance(N, PDFA):
        return rho_infty_norm_rnn(M, N, w)
    
    # if need rho given two states
    if w == -1:
        return rho_infty_norm_states(M, N, qM, qN)

    # if need rho given a word
    else:
        return rho_infty_norm_pdfas(M, N, w)


def rho_infty_norm_rnn(M, N, w):
    w = tuple(map(int, list(w)))
    Mw = M.transition_weights[M.state_after_word(w)]
    Nw = N.state_probs_dist(N.state_from_sequence(w))
    biggest = 0
    for a in M.internal_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    # M_eos = Mw['<EOS>']
    # N_eos = Nw[len(M.input_alphabet)]
    # biggest = max(biggest, abs(M_eos - N_eos))
    return biggest


def rho_infty_norm_pdfas(M, N, w):
    w_ints = tuple(map(int, list(w)))
    qM = M.state_after_word(w_ints)
    qN = N.state_after_word(w_ints)
    Mw = M.transition_weights[qM]
    Nw = N.transition_weights[qN]
    biggest = 0
    for a in M.internal_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    return biggest


def rho_infty_norm_states(M, N, qM, qN):
    Mw = M.transition_weights[qM]
    Nw = N.transition_weights[qN]
    biggest = 0
    for a in M.internal_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    return biggest