from get_M_N import get_M_N, get_M_N_hack
from our_grammars import assert_and_give_pdfa
from time import time
import argparse, ast, math
import matplotlib.pyplot as plt
from toy_pdfa import toy_pdfa1

UNMARKED = '_'
UNUSED = '*'

uhl_1 = 'results/uhl_1_1.59375'
uhl_2 = 'results/uhl_2_1.546875'
uhl_3 = 'results/uhl_3_1.578125'


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


def main():
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
            #lower = bound_d(M, N, '', alpha, test_words, False)
            calc_elapsed_time = time() - calc_start_time
            print(f'upper bound on d for n = {n}, alpha = {alpha}: {upper}')
            #print(f'lower bound on d for n = {n}, alpha = {alpha}: {lower}')
            print(f'calculating bounds for n = {n}, alpha = {alpha} took {calc_elapsed_time:.3f}s')

            resultkeyupper = f'upper, alpha = {alpha}'
            #resultkeylower = f'lower, alpha = {alpha}'
            if resultkeyupper in results['upper']: #and resultkeylower in results['lower']:
                results['upper'][resultkeyupper].append(upper)
                #results['lower'][resultkeylower].append(lower)
            else: 
                results['upper'][resultkeyupper] = [upper]
                #results['lower'][resultkeylower] = [lower]
    
    plot_results(results, grammarname, f'{grammar}/distance_plots.png')
    
    elapsed_time = time() - start_time
    print(f'\ntotal time elapsed: {elapsed_time}')

# options:
# criterion: alphabetical, greedy
def get_vasilevskii_test_set(M, n, criterion='alphabetical'):

    test_words = {}
    k = len(list(M.check_reachable_states()))
    alphabet = M.input_alphabet
    #print(f"The pdfa had k={k} states, n is {n}!")

    if k > n:
        print(f"The pdfa had k={k} states, which is more than n={n}!")
        n = k
        print(f"Increased n to {n}")

    characterizing_set = {''} # no need unless using a threshold for 'almost same states'
    #characterizing_set = construct_characterizing_set(M) 
    #print(f'characterizing_set: {characterizing_set}')
    spanning_tree_words = construct_spanning_tree_words(M)
    #print(f'spanning_tree_words: {spanning_tree_words}')

    for i in range(n - k + 1):
        all_words_of_len = get_all_words_of_len(i + 1, alphabet)
        #print(f'all_words_of_len: {all_words_of_len}')
        construct_test_words(test_words, spanning_tree_words, all_words_of_len, characterizing_set)
    
    for words_of_len in test_words:
        test_words[words_of_len] = list(test_words[words_of_len])
        test_words[words_of_len].sort()
    #print(test_words)
    return test_words


def bound_d(M, N, w, alpha, test_words, is_upper_bound):

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


def rho(M, N, w):
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


        

def construct_spanning_tree_words(pdfa):
    result = {''}
    queue = [(pdfa._initial_state, '')]
    visited = {pdfa._initial_state}

    while len(queue) > 0:
        (q, word) = queue.pop()
        for s in pdfa.input_alphabet:
            newstate = pdfa.next_state(q, pdfa.char2int[s])
            if not newstate in visited:
                queue.insert(0, (newstate, word + str(s)))
                visited.add(newstate)
                result.add(word + str(s))
    result = list(result)
    result.sort()
    result.sort(key=len)
    return result

# TODO maybe add optional threshold so can accept distributions that are 'almost same'
# in that case, the piece of code below can be used, after being tested to make sure it works
def construct_characterizing_set(pdfa):
    return {''}

# def construct_characterizing_set(pdfa):
#     states = list(pdfa.check_reachable_states())
#     marked = True
#     dist_strs = [[UNMARKED] * len(states) for i in range(len(states))]
#     dist_strs_set = {''}

#     # mark states p, q if their output distributions are not the same
#     # maybe add optional threshold so can accept distributions that are 'almost same'
#     for p in states:
#         for q in states:
#             if pdfa.transition_weights[p] != pdfa.transition_weights[q]:
#                 dist_strs[states.index(p)][states.index(q)] = '' # distinguishable by empty string
    
#     for i in range(len(dist_strs)):
#         for j in range(len(dist_strs[0])):
#             if i < j:
#                 dist_strs[i][j] = UNUSED
    
#     # repeat until can no longer mark any pair of states
#     while marked:
#         marked = False
#         for p in states:
#             for q in states:
#                 idx_p = states.index(p)
#                 idx_q = states.index(q)

#                 # for each unmarked pair p, q
#                 if dist_strs[idx_p][idx_q] == UNMARKED and idx_p != idx_q:

#                     # check if already distinguished by existing distinguishing strings
#                     dist_string = try_using_existing_dstrings(pdfa, p, q, dist_strs_set, dist_strs, states)
#                     if dist_string:
#                         marked = True
#                         continue

#                     for s in pdfa.alphabet:
#                         idx_p_s = states.index(pdfa.delta[p][s])
#                         idx_q_s = states.index(pdfa.delta[q][s])
#                         dist_str = dist_strs[idx_p_s][idx_q_s] if dist_strs[idx_p_s][idx_q_s] != UNUSED else dist_strs[idx_q_s][idx_p_s]
#                         if dist_str != UNMARKED:
#                             dist_strs[idx_p][idx_q] = s + dist_str
#                             dist_strs_set.add(s + dist_str)
#                             marked = True
#                             break

#     res = list(dist_strs_set)
#     #res = list(itertools.chain.from_iterable(dist_strs))
#     #res = list(set([val for val in res if (val != UNMARKED and val != UNUSED)]))
#     res.sort()
#     res.sort(key=len)
#     return res


def try_using_existing_dstrings(pdfa, p, q, dist_strs_set, dist_strs, states):
    ret_val = ''
    idx_p = states.index(p)
    idx_q = states.index(q)
    for d_str in dist_strs_set:
        p_st = pdfa_transition(pdfa, p, d_str)
        q_st = pdfa_transition(pdfa, q, d_str)
        if pdfa.transition_weights[p_st] != pdfa.transition_weights[q_st]:
            dist_strs[idx_p][idx_q] = d_str
            ret_val = d_str
    return ret_val


def pdfa_transition(pdfa, state, word):
    for s in word:
        state = pdfa.next_state(state, s)
    return state


def get_all_words_of_len(length, alphabet):
    queue = ['']
    while len(queue[-1]) < length:
        word = queue.pop()
        for s in alphabet:
            queue.insert(0, word + str(s))
    queue.sort()
    return queue


def construct_test_words(test_words, spanning_tree_words, all_words_of_length, characterizing_set):
    for b in all_words_of_length:
        for a in spanning_tree_words:            
            for c in characterizing_set:
                new_word = a + b + c
                if len(new_word) in test_words:
                    test_words[len(new_word)].add(new_word)
                else: 
                    test_words[len(new_word)] = {new_word}


#main()
    