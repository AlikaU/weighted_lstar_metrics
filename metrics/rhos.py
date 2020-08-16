from weighted_lstar.PDFA import PDFA


# Kendall Tau metric between rankings given by probability distribution (most likely to least likely)
def rho_kt(M, N, w=-1, qM=-1, qN=-1):
    Mw, Nw = None, None

    # if dealing with an RNN
    if not isinstance(N, PDFA):
        w = tuple(map(int, list(w)))
        Mw = M.transition_weights[M.state_after_word(w)]
        Nw = N.state_probs_dist(N.state_from_sequence(w))
    
    # if given two states
    if qM != -1 and qN != -1:
        Mw = M.transition_weights[qM]
        Nw = N.transition_weights[qN]

    # if given a word
    else:
        w_ints = tuple(map(int, list(w)))
        qM = M.state_after_word(w_ints)
        qN = N.state_after_word(w_ints)
        Mw = M.transition_weights[qM]
        Nw = N.transition_weights[qN]

    # get rankings of symbols by M and N
    r_M = [k for k, v in sorted(Mw.items(), key=lambda item: item[1], reverse=True)]
    r_N = [k for k, v in sorted(Nw.items(), key=lambda item: item[1], reverse=True)]
    # print(f'rankings_M: {r_M}')
    # print(f'rankings_N: {r_N}')
    return kendall_tau(r_M, r_N, alphabet = M.internal_alphabet)

   


def kendall_tau(r_M, r_N, alphabet):

    # count how many pairs of symbols are in a different order in M and M
    count = 0
    
    for i in range(len(alphabet) - 1):
        for j in range(i+1, len(alphabet)):
            # are ith and jth symbols in different order in rankings given by M and N?
            if (r_M.index(alphabet[i]) - r_M.index(alphabet[j])) * (r_N.index(alphabet[i]) - r_N.index(alphabet[j])) < 0:
                count += 1
    
    # normalize by the total number of possible pairs of symbols
    k = len(alphabet)
    total_num_pairs = k*(k-1)/2 # len(alphabet) choose 2
    return count / total_num_pairs


# largest difference in next symbol probabilities
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