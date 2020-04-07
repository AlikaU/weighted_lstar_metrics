import numpy as np

# TODO method stubs

# in: M, N, accuracy thershold maybe
def compute_d(M, N, alpha):
    #list(pdfa.check_reachable_states())
    num_states_M = len(M.check_reachable_states())
    num_states_N = len(N.check_reachable_states())
    distances = np.zeros((num_states_M, num_states_N))
    distance = d(M._initial_state, N._initial_state, distances, alpha)

def d(qM, qN, distances, alpha):
    distance_has_changed = True
    dist = 'hi' #find which indices in distances matrix correspond to qM, qN and get it

    while distance_has_changed:
        distance_has_changed = False
        new_dist = alpha*rho(qM, qN)
        # find which indices in distances matrix correspond to qM, qN
        # update matrix to new_dist
        if dist != new_dist:
            distance_has_changed = True
    
    return dist

# TODO
def rho(M, N, w):
    # w = tuple(map(int, list(w)))
    # Mw = M.transition_weights[M.state_after_word(w)]
    # Nw = N.state_probs_dist(N.state_from_sequence(w))
    # biggest = 0
    # for a in M.input_alphabet:
    #     biggest = max(biggest, abs(Mw[a] - Nw[a]))
    # M_eos = Mw['<EOS>']
    # N_eos = Nw[len(M.input_alphabet)]
    # biggest = max(biggest, abs(M_eos - N_eos))
    # return biggest
