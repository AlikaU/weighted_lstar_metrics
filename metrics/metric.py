import argparse, ast, math, matplotlib.pyplot as plt, numpy as np
from time import time

from weighted_lstar.our_grammars import assert_and_give_pdfa
from metrics.get_M_N import get_M_N, get_M_N_hack
from metrics.toy_pdfa import toy_pdfa1
from metrics.vasilevski_chow_test_set import get_vasilevskii_test_set


# M: known PDFA
# N: blackbox
def compare_truedist_vs_bound(M, N, alpha, rho, n, bound_type='bfs', max_depth=0, max_revisits=-1):
    dist, count = compute_d(M, N, alpha, rho)
    if M.num_reachable_states == 2:
        print('hi')
    #test_words = get_vasilevskii_test_set(M, n)
    #upper_bound = bound_d(M, N, '', alpha, test_words, True)
    upper_bound, _, _ = get_brute_force_d_bound(M, N, alpha, rho, bound_type, max_depth, max_revisits)
    msg = f'M: {M.informal_name}, N: {N.informal_name}\nestimated distance (upper bound): {upper_bound}\nactual distance: {dist}, found after {count} iterations'
    return upper_bound, dist, msg


# M: known PDFA
# N: blackbox
def bound_d(M, N, w, alpha, rho, test_words, is_upper_bound):

    longest_wordlen = max(k for k in test_words.keys())
    if len(w) >= longest_wordlen: # base case
        x = 1 if is_upper_bound else 0
        return alpha * rho(M, N, w=w) + (1 - alpha) * x
    
    else: # recursive case
        a = alpha * rho(M, N, w=w)
        biggest = 0
        for next_w in test_words[len(w)+1]:
            if not w == next_w[:len(w)]: continue
            b = bound_d(M, N, next_w, alpha, test_words, is_upper_bound)
            biggest = max(biggest, b)
        return a + (1 - alpha) * biggest


# in: M, N, accuracy thershold maybe
def compute_d(M, N, alpha, rho):
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
                distances[M_state_row][N_state_col] = alpha*rho(M, N, qM=qM, qN=qN) + (1 - alpha)*max_next_dist
                
                if distances[M_state_row][N_state_col] != old_dist:
                    changed = True
    
    return distances[0][0], count # d(qM0, qN0) distance between initial states of M, N


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

def str_to_ints(w):
    # res = []
    # for char in w:
    #     res.append(M.char2int[char])
    # return res
    return tuple(map(int, list(w)))


# M: known PDFA, N: blackbox
# search_type: 'bfs' or 'all_paths', all paths is pretty much useless
# max_depth: for bfs, we'll search all words of length up to max_depth
# max_revisits: for all paths search: how many times can a word revisit same state
def get_brute_force_d_bound(M, N, alpha, rho, search_type, max_depth, max_revisits, verbose=True):
    count = 0
    queue = [
            {
                'state':M._initial_state, 
                'word':'', 
                'cost':alpha*rho(M, N, w='')
            }
        ]
    costliest_path = None

    while len(queue) > 0:
        count +=1 # paths considered. TODO possible speedup: stop repeatedly considering end nodes when it's clear
        cur = queue.pop(0)

        for symbol in M.internal_alphabet:
            end_path = None
            end_cost = None

            if symbol == 'EOS':
                end_path = cur['word']+symbol
                pr_eos_m = M.transition_weights[M.state_after_word(str_to_ints(next_word))]['EOS']
                pr_eos_n = N.transition_weights[N.state_after_word(str_to_ints(next_word))]['EOS']
                eos_cost = abs(pr_eos_m - pr_eos_n)
                end_cost = cur['cost'] + (1-alpha)**len(end_path) * alpha * eos_cost

            else:
                next_word = cur['word']+str(symbol)
                next_state = M.state_after_word(str_to_ints(next_word))

                if (search_type != 'bfs' and num_revisits(M, next_word, next_state) > max_revisits or 
                    search_type == 'bfs' and len(next_word) > max_depth):             
                    upper_bound_of_unexplored = (1-alpha)**(len(cur['word'])+1)
                    end_cost = cur['cost'] + upper_bound_of_unexplored
                    end_path = cur['word'] + 'x'
                    
                else:
                    next_cost = cur['cost'] + (1-alpha)**len(next_word) * alpha * rho(M, N, w=next_word)
                    queue.append({
                        'state':next_state, 
                        'word':next_word, 
                        'cost':next_cost
                    })
                    continue

            #print(f'end_path: {end_path}, end_cost: {end_cost}')
            if not costliest_path or costliest_path['cost'] < end_cost:
                #print(f'costliest_path is now: {end_path}, {end_cost}')
                costliest_path = {'path': end_path, 'cost': end_cost}
    
    cp = costliest_path['path']
    cc = costliest_path['cost']
    if verbose: print(f'found costliest path {cp} with cost {cc} after considering {count} paths')
    return costliest_path['cost'], costliest_path['path'], count

def num_revisits(M, w, s):
    count = 0
    for i in range(len(w)):
        t = M.state_after_word(str_to_ints(w[:i]))
        if t == s: count += 1
    return count

