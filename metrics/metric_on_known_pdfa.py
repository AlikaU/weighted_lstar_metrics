import math, time, os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from datetime import datetime
from metrics import toy_pdfa
from metrics.metric import bound_d, rho_pdfas_states
from metrics.vasilevski_chow_test_set import get_vasilevskii_test_set

from weighted_lstar.our_grammars import uhl1, uhl2, uhl3, assert_and_give_pdfa

change_types = {
    'chg_nstates': 'number of states',
    'chg_dist_all': 'next state distribution of all states by 0.01',
    'chg_dist_one': 'next state distribution of one state by 0.01'
}


# result is one chg_type with either d_as_difference_increases or delta_as_difference_increases
# both objects are structured like so:
# y axis: bounds, actual_vals
# x axis: assumed to be i in range (steps)
def plot_results(result, steps):
    colors = {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}
    mpl.style.use('seaborn')
    print(result)
    x = list(range(steps))
    # line_labels = ['NB', 'LR']
    for i, res in enumerate(result['results']):
        y = res['data']
        lbl = res['label']
        line_style = '--' if res['type']=='bound' else '-' 
        clr = f'C{math.floor(i/2.0)}'

        plt.plot(x, y, line_style, color=clr, label=lbl)
 
    
    #plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
    plt.minorticks_on()
    #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #     plt.ylim(0.5, 1)

    #     plt.ylabel(f'test accuracy')
    #     if (i == 1): plt.xlabel('percentage of train set used')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fancybox=True, shadow=True)
    plt.xlabel(result['xlabel'])
    plt.ylabel(result['ylabel'])
    plt.tight_layout()
    #plt.suptitle(result['graph_name'])
    plt.text(5, 10, result['graph_name'], ha='center', va='bottom', wrap=True)
    sn = result['short_name']
    results_filename = f'{resultpath}/{sn}.png'
    plt.savefig(results_filename)
    plt.close()


def plot_6(alpha, steps):
    Ms = [toy_pdfa_10statesA(), toy_pdfa_10statesB(), uhl1(), uhl2(), uhl3()]
    results = []
    for change_type in change_types:
        results.append(d_as_difference_increases(Ms, change_type, steps, alpha))

        M = Ms[0] # TODO decide which
        results.append(delta_as_difference_increases(M, change_type, steps, alpha))
    
    for result in results:
        plot_results(result, steps)


# d between M and different variations of it
def d_as_difference_increases(Ms, changetype, steps, alpha):
    results = []
    for M in Ms:
        bounds, actual_vals = [], []
        for i in range (steps):
            N = get_modified_aut(M, i, changetype)
            upper_bound, true_dist, msg = compare_truedist_vs_bound(M, N, alpha, f'{M.informal_name}_{changetype}_{i}', resultpath)
            logger.log(msg)
            bounds.append(upper_bound)
            actual_vals.append(true_dist)
        logger.log(f'Results as we modify the {changetype}:\nUpper bounds: {bounds} \nTrue distances: {actual_vals}')
        results.append({'label': f'M = {M.informal_name}, bound', 'type': 'bound', 'data': bounds})
        results.append({'label': f'M = {M.informal_name}, actual', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of distances between M and its modified versions. Each incremental modification of M reduces the {changetype}'
    sname = f'd_{changetype}'
    return {'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'd'}


# difference in predictions, for different words, for M and different variations of it
def delta_as_difference_increases(M, changetype, steps, alpha):
    ws = ['0', '00', '000'] # TODO decide which
    results = []
    for w in ws:
        bounds, actual_vals = [], []
        for i in range (steps):
            N = get_modified_aut(M, i, changetype)
            n = len(N.check_reachable_states())
            test_words = get_vasilevskii_test_set(M, n)
            upper_bound = bound_d(M, N, '', alpha, test_words, True)
            bounds.append(get_delta_w_bound(upper_bound, alpha))
            actual_vals.append(get_delta_w_actual(M, N, w, alpha))
        results.append({'label': f'w = {w}, bound', 'type': 'bound', 'data': bounds})
        results.append({'label': f'w = {w}, actual', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of discrepancy between the outputs of {M.informal_name} and its modified versions as we read different input words. Each incremental modification of M reduces the {change_types[changetype]}'
    sname = f'delta_{M.informal_name}_{changetype}'
    return {'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'discr(w)'}


def get_modified_aut(M, i, changetype):
    M_states = list(M.check_reachable_states())
    n = len(M_states)
    informal_name = f'{M.informal_name}_{i}_{changetype}'
    transitions = {}
    transition_weights = {}
    alphabet = list(M.internal_alphabet)
    alphabet.remove('EOS')
    alphabet = tuple(alphabet)

    for j in range(n):
        transition_weights[j] = {}
        transitions[j] = {}
        for symbol in alphabet:        
            transition_weights[j][symbol]=M.transition_weights[j][symbol]
            transitions[j][symbol] = M.transitions[j][symbol]

    if changetype == 'chg_nstates':
        nstates = n
        for idx in range(i):
            transitions = remove_one_state(transitions, nstates, alphabet)
            nstates -= 1
          
    else:
        for state in range(n):
            transitions[state] = M.transitions[state]

            if changetype == 'chg_dist_all':
                transition_weights[state][alphabet[0]] += i * 0.01
                transition_weights[state][alphabet[-1]] -= i * 0.01

            elif changetype == 'chg_dist_one':
                if state == n - 1:
                    transition_weights[state][alphabet[0]] += i * 0.01
                    transition_weights[state][alphabet[-1]] -= i * 0.01

    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def remove_one_state(transitions, nstates, alphabet):
    to_remove = nstates - 1
    for state in range (nstates):
        for symbol in alphabet:
            if transitions[state][symbol] == to_remove:
                transitions[state][symbol] = transitions[to_remove][symbol]
            # if it still points to removed state, then loop back to itself
            if transitions[state][symbol] == to_remove:
                transitions[state][symbol] = state
    del transitions[to_remove]
    return transitions


def get_delta_w_bound(upper_bound, alpha):
    return upper_bound/alpha


def str_to_ints(M, w):
    # res = []
    # for char in w:
    #     res.append(M.char2int[char])
    # return res
    return tuple(map(int, list(w)))


# discounted sum of discrepancies between M and N as we read w
def get_delta_w_actual(M, N, w, alpha):
    sum = 0
    w = str_to_ints(M, w)
    for i in range(len(w) + 1):
        w_subs = w[0:i]
        #w_ints = tuple(map(int, list(w[0:i])))
        qM = M.state_after_word(w_subs)
        qN = N.state_after_word(w_subs)
        sum += rho_pdfas_states(M, N, qM, qN) * (1 - alpha)**i
    return sum
