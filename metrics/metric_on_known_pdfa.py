import math, os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from metrics import toy_pdfa
from metrics.metric import bound_d, rho_pdfas_states, compare_truedist_vs_bound
from metrics.vasilevski_chow_test_set import get_vasilevskii_test_set, construct_spanning_tree_words

from weighted_lstar.our_grammars import assert_and_give_pdfa

change_types = {
    'chg_nstates': 'number of states',
    'chg_dist_all': 'next state distribution of all states by 0.01',
    'chg_dist_one': 'next state distribution of one state by 0.01'
}


# result is one chg_type with either d_as_difference_increases or delta_as_difference_increases
# both objects are structured like so:
# y axis: bounds, actual_vals
# x axis: assumed to be i in range (steps)
def plot_results(result, steps, resultpath):
    colors = {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}
    mpl.style.use('seaborn')
    print(result)
    #x = list(range(steps))
    for i, res in enumerate(result['results']):
        y = res['data']
        lbl = res['label']
        line_style = '--' if res['type']=='bound' else '-' 
        clr = f'C{math.floor(i/2.0)}'

        #plt.plot(x, y, line_style, color=clr, label=lbl)
        plt.plot(y, line_style, color=clr, label=lbl)
 
    
    #plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
    plt.minorticks_on()
    #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #     plt.ylim(0.5, 1)

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

# TODO maybe play with test_depth
# test_depth: when computing the bound, how far do we look beyond the reachable states of the smaller PDFA
# (we will treat the smaller PDFA as the known automaton and the larger one as the blackbox)
def plot_6(PDFAs, alpha, steps, resultpath, logger, test_depth=3):

    results = []
    for change_type in change_types:
        results.append(d_as_difference_increases(PDFAs, change_type, steps, alpha, resultpath, logger, test_depth))

        pdfa = PDFAs[0] # TODO decide which
        results.append(delta_as_difference_increases(pdfa, change_type, steps, alpha, resultpath, test_depth))
    
    for result in results:
        plot_results(result, steps, resultpath)



# d between original PDFAs and different variations of them

def d_as_difference_increases(PDFAs, changetype, steps, alpha, resultpath, logger, test_depth):
    results = []
    for original in PDFAs:
        bounds, actual_vals = [], []
        if resultpath:
            original.draw_nicely(keep=True,filename=f'{resultpath}/{original.informal_name}/original_pdfa')
        for i in range (min(steps, original.num_reachable_states)):
            modified = get_modified_aut(original, i, changetype, steps)
            if resultpath:
                modified.draw_nicely(keep=True,filename=f'{resultpath}/{original.informal_name}/modified_{changetype}/{i}')
            n = len(modified.check_reachable_states()) + test_depth
            upper_bound, true_dist, msg = compare_truedist_vs_bound(modified, original, alpha, n)
            logger.log(msg)
            bounds.append(upper_bound)
            actual_vals.append(true_dist)
        logger.log(f'Results as we modify the {changetype}:\nUpper bounds: {bounds} \nTrue distances: {actual_vals}')
        results.append({'label': f'original = {original.informal_name}, bound', 'type': 'bound', 'data': bounds})
        results.append({'label': f'original = {original.informal_name}, actual', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of distances between the original PDFA and its modified versions. Each incremental modification reduces the {changetype}'
    sname = f'd_{changetype}'
    return {'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'd'}


# difference in predictions, for different words, for M and different variations of it
def delta_as_difference_increases(original, changetype, steps, alpha, resultpath, test_depth):
    #ws = ['0', '00', '000'] # TODO decide which
    ws = construct_spanning_tree_words(original)
    print(f'testing discrepancy between original and modified on the following words: {ws}')
    results = []
    for w in ws:
        bounds, actual_vals = [], []
        for i in range (min(steps, original.num_reachable_states)):
            modified = get_modified_aut(original, i, changetype, steps)
            n = len(modified.check_reachable_states()) + test_depth
            test_words = get_vasilevskii_test_set(modified, n)
            upper_bound = bound_d(modified, original, '', alpha, test_words, True)
            bounds.append(get_delta_w_bound(upper_bound, alpha))
            actual_vals.append(get_delta_w_actual(original, modified, w, alpha))
        results.append({'label': f'w = {w}, bound', 'type': 'bound', 'data': bounds})
        results.append({'label': f'w = {w}, actual', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of discrepancy between the outputs of {original.informal_name} and its modified versions as we read different input words. Each incremental modification of M reduces the {change_types[changetype]}'
    sname = f'delta_{original.informal_name}_{changetype}'
    return {'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'discr(w)'}


def get_modified_aut(M, i, changetype, totalsteps):
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
            start_val = transition_weights[state][0]
            for idx in range(i):
                if changetype == 'chg_dist_all':
                    chg_distr_state(alphabet, transition_weights, state, totalsteps, start_val)

                elif changetype == 'chg_dist_one':
                    if state == n - 1:
                        chg_distr_state(alphabet, transition_weights, state, totalsteps, start_val)
                        # transition_weights[state][alphabet[0]] += i * 0.01
                        # transition_weights[state][alphabet[-1]] -= i * 0.01

    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

# TODO test on all automata
# how to cycle through states that we are modifying and ensure that with each call to this, we get an increasingly different automaton?
def chg_distr_state(alphabet, transition_weights, state, totalsteps, start_val):
    step = (1 - start_val)/totalsteps
    assert transition_weights[state][0] + step <= 1
    transition_weights[state][0] += step
    if transition_weights[state][1] - step >= 0:
        transition_weights[state][1] -= step
    elif transition_weights[state][2] - step >= 0:
        transition_weights[state][2] -= step
    else: # probs for 1 and 2 must sum up to 0.09
        assert round(transition_weights[state][1] + transition_weights[state][2], 5) == step
        # make them almost 0 and make sure they add up to 1. this is to avoid completely making probabilities 0 which 'erases' the state
        # by making it unreachable
        transition_weights[state][2] = 0.0001
        transition_weights[state][1] = 0.0001 
        transition_weights[state][0] = 0.9998


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
