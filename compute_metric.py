from get_M_N import get_M_N
from time import time
import argparse, ast, math

UNMARKED = '_'
UNUSED = '*'


def main():
    start_time = time()
    M, N = get_M_N(False)
    
    elapsed_time = time() - start_time
    print(f'\ntime elapsed: {elapsed_time}')

# options:
# criterion: alphabetical, greedy
def get_vasilevskii_test_set(M, n, criterion='alphabetical'):

    k = len(list(M.check_reachable_states()))
    alphabet = M.input_alphabet

    if k > n:
        print(f"The pdfa had k={k} states, which is more than n={n}!")
        n = k
        print(f"Increased n to {n}")

    # characterizing_set = construct_characterizing_set(M)
    # print(f'characterizing_set: {characterizing_set}')
    spanning_tree_words = construct_spanning_tree_words(M)
    print(f'spanning_tree_words: {spanning_tree_words}')

    # for i in range(n - k + 1):
    #     all_words_of_len = get_all_words_of_len(i + 1, alphabet)
    #     #print(f'all_words_of_len: {all_words_of_len}')
    #     test_words = construct_test_words(spanning_tree_words, all_words_of_len, characterizing_set)

def construct_spanning_tree_words(pdfa):
    result = {''}
    queue = [(pdfa._initial_state, '')]
    visited = {pdfa._initial_state}

    while len(queue) > 0:
        (q, word) = queue.pop()
        for s in pdfa.alphabet:
            newstate = pdfa.delta[q][s]
            if not newstate in visited:
                queue.insert(0, (newstate, word + s))
                visited.add(newstate)
                result.add(word + s)
    result = list(result)
    result.sort()
    result.sort(key=len)
    return result


def construct_characterizing_set(pdfa):
    states = pdfa.Q
    marked = True
    dist_strs = [[UNMARKED] * len(states) for i in range(len(states))]
    dist_strs_set = {''}

    # mark states p, q if one is accepting and another rejecting
    for p in states:
        for q in states:
            if (p in pdfa.F and q not in pdfa.F) or (p not in pdfa.F and q in pdfa.F):
                dist_strs[states.index(p)][states.index(q)] = ''
    
    for i in range(len(dist_strs)):
        for j in range(len(dist_strs[0])):
            if i < j:
                dist_strs[i][j] = UNUSED
    
    # repeat until can no longer mark any pair of states
    while marked:
        marked = False
        for p in states:
            for q in states:
                idx_p = states.index(p)
                idx_q = states.index(q)

                # for each unmarked pair p, q
                if dist_strs[idx_p][idx_q] == UNMARKED and idx_p != idx_q:

                    # check if already distinguished by existing distinguishing strings
                    dist_string = try_using_existing_dstrings(pdfa, p, q, dist_strs_set, dist_strs)
                    if dist_string:
                        marked = True
                        continue

                    for s in pdfa.alphabet:
                        idx_p_s = states.index(pdfa.delta[p][s])
                        idx_q_s = states.index(pdfa.delta[q][s])
                        dist_str = dist_strs[idx_p_s][idx_q_s] if dist_strs[idx_p_s][idx_q_s] != UNUSED else dist_strs[idx_q_s][idx_p_s]
                        if dist_str != UNMARKED:
                            dist_strs[idx_p][idx_q] = s + dist_str
                            dist_strs_set.add(s + dist_str)
                            marked = True
                            break

    res = list(dist_strs_set)
    #res = list(itertools.chain.from_iterable(dist_strs))
    #res = list(set([val for val in res if (val != UNMARKED and val != UNUSED)]))
    res.sort()
    res.sort(key=len)
    return res


def try_using_existing_dstrings(pdfa, p, q, dist_strs_set, dist_strs):
    ret_val = ''
    states = pdfa.Q
    idx_p = states.index(p)
    idx_q = states.index(q)
    for d_str in dist_strs_set:
        p_str = pdfa_transition(pdfa, p, d_str)
        q_str = pdfa_transition(pdfa, q, d_str)
        if (p_str in pdfa.F and q_str not in pdfa.F) or (p_str not in pdfa.F and q_str in pdfa.F):
            dist_strs[idx_p][idx_q] = d_str
            ret_val = d_str
    return ret_val


def pdfa_transition(pdfa, state, word):
    for s in word:
        state = pdfa.delta[state][s]
    return state


def get_all_words_of_len(length, alphabet):
    queue = ['']
    while len(queue[-1]) < length:
        word = queue.pop()
        for s in alphabet:
            queue.insert(0, word + s)
    queue.sort()
    return queue


def construct_test_words(spanning_tree_words, all_words_of_length, characterizing_set):
    test_words = []
    for b in all_words_of_length:
        for a in spanning_tree_words:            
            for c in characterizing_set:
                test_words.append(a + b + c)
    return test_words

main()
    