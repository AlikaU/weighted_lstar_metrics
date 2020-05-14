import argparse, ast, math, matplotlib.pyplot as plt, numpy as np
from time import time

from weighted_lstar.our_grammars import assert_and_give_pdfa
from weighted_lstar.PDFA import PDFA

from metrics.get_M_N import get_M_N, get_M_N_hack
from metrics.toy_pdfa import toy_pdfa1
from metrics.vasilevski_chow_test_set import get_vasilevskii_test_set



def compare_truedist_vs_bound(M, N, alpha, filename, resultfolder=None):
    dist, count = compute_d(M, N, alpha, filename, resultfolder)
    n = len(N.check_reachable_states())
    test_words = get_vasilevskii_test_set(M, n)
    upper_bound = bound_d(M, N, '', alpha, test_words, True)
    msg = f'M: {M.informal_name}, N: {N.informal_name}\nestimated distance (upper bound): {upper_bound}\nactual distance: {dist}, found after {count} iterations'
    return upper_bound, dist, msg


def bound_d(M, N, w, alpha, test_words, is_upper_bound):
    rho = rho_pdfas if isinstance(N, PDFA) else rho_rnn

    longest_wordlen = max(k for k in test_words.keys())
    if len(w) >= longest_wordlen: # base case
        x = 1 if is_upper_bound else 0
        return alpha * rho(M, N, w) + (1 - alpha) * x
    
    else: # recursive case
        a = alpha * rho(M, N, w)
        biggest = 0
        for next_w in test_words[len(w)+1]:
            if not w == next_w[:len(w)]: continue
            b = bound_d(M, N, next_w, alpha, test_words, is_upper_bound)
            biggest = max(biggest, b)
        return a + (1 - alpha) * biggest


# in: M, N, accuracy thershold maybe
def compute_d(M, N, alpha, filename, resultfolder=None):
    if resultfolder:
        M.draw_nicely(keep=True,filename=resultfolder+filename+'/M')
        N.draw_nicely(keep=True,filename=f'{resultfolder}{filename}/{N.informal_name}')
    M_states = list(M.check_reachable_states())
    N_states = list(N.check_reachable_states())
    distances = np.zeros((len(M_states), len(N_states)))
    count = 0
    changed = True

    while changed:
        changed = False
        count +=1 
        #print(f'iter {count}, current dist: {distances[0][0]}')
        for M_state_row in range(len(distances)):
            for N_state_col in range(len(distances[M_state_row])):
                old_dist = distances[M_state_row][N_state_col]
                max_next_dist = find_max_next_dist(M_state_row, N_state_col, distances, M, N)
                qM = M_states[M_state_row]
                qN = N_states[N_state_col]
                distances[M_state_row][N_state_col] = alpha*rho_pdfas_states(M, N, qM, qN) + (1 - alpha)*max_next_dist
                
                if distances[M_state_row][N_state_col] != old_dist:
                    changed = True
    
    return distances[0][0], count # d(qM0, qN0) distance between initial states of M, N


def rho_rnn(M, N, w):
    w = tuple(map(int, list(w)))
    Mw = M.transition_weights[M.state_after_word(w)]
    Nw = N.state_probs_dist(N.state_from_sequence(w))
    biggest = 0
    for a in M.input_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    M_eos = Mw['<EOS>']
    N_eos = Nw[len(M.input_alphabet)]
    biggest = max(biggest, abs(M_eos - N_eos))
    return biggest


def rho_pdfas(M, N, w):
    w_ints = tuple(map(int, list(w)))
    qM = M.state_after_word(w_ints)
    qN = N.state_after_word(w_ints)
    Mw = M.transition_weights[qM]
    Nw = N.transition_weights[qN]
    biggest = 0
    for a in M.input_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    return biggest


def rho_pdfas_states(M, N, qM, qN):
    Mw = M.transition_weights[qM]
    Nw = N.transition_weights[qN]
    biggest = 0
    for a in M.input_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    return biggest


def find_max_next_dist(qM, qN, distances, M, N):
    M_states = list(M.check_reachable_states())
    N_states = list(N.check_reachable_states())
    # maybe the worst case is if we see the stop symbol and stay in this state
    # TODO: should we do this case?
    #biggest = distances[M_states.index(qM)][M_states.index(qN)]
    biggest = 0
    for a in M.input_alphabet:
        row_idx = M_states.index(M.next_state(qM, a))
        col_idx = N_states.index(N.next_state(qN, a))
        biggest = max(biggest, distances[row_idx][col_idx])
    return biggest


def bound_d_vs_n():
    uhl_1 = 'results/uhl_1_1.59375'
    uhl_2 = 'results/uhl_2_1.546875'
    uhl_3 = 'results/uhl_3_1.578125'
    grammar = uhl_1
    grammarname = 'uhl_1'

    start_time = time()
    print(f'\ncomputing distance between PDFA and RNN!')
    M, N = get_M_N_hack(grammar, False)
    #alphas = [0.05, 0.1, 0.2]
    alphas = [0.2]

    results = { 'x': [], 'upper': {}, 'lower': {}}
    #nmax = 6
    nmax = 1

    m = len(list(M.check_reachable_states()))
    for n in range(m, m + nmax):
        results['x'].append(n)
        test_words = get_vasilevskii_test_set(M, n)

        for alpha in alphas:
            calc_start_time = time()
            upper = bound_d(M, N, '', alpha, test_words, True)
            calc_elapsed_time = time() - calc_start_time
            print(f'upper bound on d for n = {n}, alpha = {alpha}: {upper}')
            print(f'calculating bounds for n = {n}, alpha = {alpha} took {calc_elapsed_time:.3f}s')

            resultkeyupper = f'upper, alpha = {alpha}'
            if resultkeyupper in results['upper']: #and resultkeylower in results['lower']:
                results['upper'][resultkeyupper].append(upper)
            else: 
                results['upper'][resultkeyupper] = [upper]
    
    plot_results(results, grammarname, f'{grammar}/distance_plots.png')
    
    elapsed_time = time() - start_time
    print(f'\ntotal time elapsed: {elapsed_time}')


def plot_results(results, grammar, filename):
    for key in results['upper']:
        plt.plot(results['x'], results['upper'][key], '-', label=key)
    # for key in results['lower']:
    #     plt.plot(results['x'], results['lower'][key], '-', label=key)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fancybox=True, shadow=True)
    plt.legend(loc='upper left')
    plt.title(f'Results for {grammar}')
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #plt.ylim(0, 1)
    plt.ylabel('d')
    plt.xlabel('n')
    plt.savefig(filename)
    plt.close()
