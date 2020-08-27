import math, os, random, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from metrics import toy_pdfa
from metrics.metric import compare_truedist_vs_bound, str_to_ints, get_brute_force_d_bound
from metrics.vasilevski_chow_test_set import construct_spanning_tree_words
from metrics.rhos import rho_kt

from weighted_lstar.our_grammars import assert_and_give_pdfa

change_types = {
    'chg_nstates': 'number of states',
    'chg_dist_all': 'next state distribution of all states by 0.01',
    'chg_dist_one': 'next state distribution of one state by 0.01'
}


def plot_using_test_set(PDFAs, alpha, rho, steps, resultpath, logger, test_depth=3, bound_type='bfs', max_depth_add=0, disc=True, max_wordlen=10):
    to_plot = []
    for change_type in change_types:

        results = []

        for original in PDFAs:
            result_for_pdfa = []
            if resultpath:
                original.draw_nicely(keep=True,filename=f'{resultpath}/{original.informal_name}/original_pdfa')

            # get samples
            test_set = get_test_set(original, 2000, maxlen=100) # TODO what should maxlen be?

            # for each M, N pair:
            steps = min(steps + 1, original.num_reachable_states) if change_type == 'chg_nstates' else steps + 1
            perf_bounds_d_true, perf_bounds_d_bound, perf_actuals = [], [], []
            for i in range (steps):
                modified = get_modified_aut(original, i, change_type, steps)
                if resultpath:
                    modified.draw_nicely(keep=True,filename=f'{resultpath}/{original.informal_name}/modified_{change_type}/{i}')
            
                max_d = modified.depth + max_depth_add
                dist_bound, dist_true, _ = compare_truedist_vs_bound(modified, original, alpha, rho, 0, max_depth=max_d)

                # save bound and actual for current M, N pair
                perf_bounds_d_true.append(get_performance_bound(dist_true, alpha, discounted=disc, maxlen=max_wordlen)) # using true d (will be same for all w)
                perf_bounds_d_bound.append(get_performance_bound(dist_bound, alpha, discounted=disc, maxlen=max_wordlen)) # using bound on d (will be same for all w)
                if disc:
                    perf_actuals.append(get_perf_actual_discounted(modified, original, alpha, rho, 2000))
                else:
                    perf_actuals.append(get_perf_actual_undiscounted(modified, original, rho, 2000, max_wordlen))
                
            result_for_pdfa.append({'label': f'bound (true d), {original.informal_name}', 'type': 'bound_true_d', 'data': perf_bounds_d_true, 'xlabel': original.informal_name, })
            result_for_pdfa.append({'label': f'bound (estimate of d), {original.informal_name}', 'type': 'bound', 'data': perf_bounds_d_bound, 'xlabel': original.informal_name, })
            result_for_pdfa.append({'label': f'actual, {original.informal_name}', 'type': 'actual', 'data': perf_actuals, 'xlabel': original.informal_name, })
            results.append(result_for_pdfa)

        gname = f'Graph of performande of modified versions of {original.informal_name} wrt. the original. Performance is measured using the cumulative discounted discrepancy, averaged across the dataset. Each incremental modification of M reduces the {change_types[change_type]}'
        disc_str = 'discounted' if disc else 'undiscounted'
        rho_name = 'Kendall Tau' if rho == rho_kt else 'infinity norm between distributions'
        sname = f'performance_{change_type}_{disc_str}'
        to_plot.append({'results': results, 'graph_name': gname, 'short_name': sname, 'ylabel': rho_name})
    
    for result in to_plot:
        plot_results_subplts(result, steps, resultpath)


# TODO maybe play with test_depth
# test_depth: when computing the bound, how far do we look beyond the reachable states of the smaller PDFA
# (we will treat the smaller PDFA as the known automaton and the larger one as the blackbox)
def plot_6(PDFAs, alpha, rho, steps, resultpath, logger, test_depth=3, bound_type='bfs', max_depth_add=0, max_revisits=-1):

    results = []
    for change_type in change_types:
        results.append(d_as_difference_increases(PDFAs, change_type, steps, alpha, rho, resultpath, logger, test_depth, bound_type, max_depth_add, max_revisits))

        pdfa = PDFAs[0] # TODO decide which
        results.append(delta_as_difference_increases(pdfa, change_type, steps, alpha, rho, resultpath, test_depth, bound_type, max_depth_add, max_revisits))
    
    for result in results:
        plot_results(result, steps, resultpath)



# d between original PDFAs and different variations of them
def d_as_difference_increases(PDFAs, changetype, steps, alpha, rho, resultpath, logger, test_depth, bound_type, max_depth_add, max_revisits):
    results = []
    for original in PDFAs:
        step_s = min(steps + 1, original.num_reachable_states) if changetype == 'chg_nstates' else steps + 1
        bounds, actual_vals = [], []
        if resultpath:
            original.draw_nicely(keep=True,filename=f'{resultpath}/{original.informal_name}/original_pdfa')
        for i in range (step_s):
            modified = get_modified_aut(original, i, changetype, step_s)
            if resultpath:
                modified.draw_nicely(keep=True,filename=f'{resultpath}/{original.informal_name}/modified_{changetype}/{i}')
            n = len(modified.check_reachable_states()) + test_depth
            max_d = modified.depth + max_depth_add
            if modified.depth > 3 or len(modified.input_alphabet) > 3:
                max_revisits=0
            upper_bound, true_dist, msg = compare_truedist_vs_bound(modified, original, alpha, rho, n, bound_type, max_d, max_revisits)
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
def delta_as_difference_increases(original, changetype, steps, alpha, rho, resultpath, test_depth, bound_type, max_depth_add, max_revisits):
    #ws = ['0', '00', '000'] # TODO decide which
    ws = construct_spanning_tree_words(original)
    steps = min(steps + 1, original.num_reachable_states) if changetype == 'chg_nstates' else steps + 1
    print(f'testing discrepancy between original and modified on the following words: {ws}')
    results = []
    for w in ws:
        bounds, actual_vals = [], []
        for i in range (steps):
            modified = get_modified_aut(original, i, changetype, steps)
            n = len(modified.check_reachable_states()) + test_depth
            #test_words = get_vasilevskii_test_set(modified, n)
            #upper_bound = bound_d(modified, original, '', alpha, test_words, True)
            max_d = modified.depth + max_depth_add
            upper_bound, _, _ = get_brute_force_d_bound(modified, original, alpha, rho, bound_type, max_d, max_revisits, verbose=False)
            bounds.append(get_performance_bound(upper_bound, alpha))
            actual_vals.append(get_perf_actual_discounted_1word(original, modified, w, alpha, rho))
        results.append({'label': f'w = {w}, bound', 'type': 'bound', 'data': bounds})
        results.append({'label': f'w = {w}, actual', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of discrepancy between the outputs of {original.informal_name} and its modified versions as we read different input words. Each incremental modification of M reduces the {change_types[changetype]}'
    sname = f'delta_{original.informal_name}_{changetype}'
    return {'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'discr(w)'}


def get_modified_aut(M, modif_amount, changetype, totalsteps):
    M_states = list(M.check_reachable_states())
    n_states = len(M_states)
    informal_name = f'{M.informal_name}_{modif_amount}_{changetype}'
    transitions = {}
    transition_weights = {}
    alphabet = list(M.internal_alphabet)
    alphabet.remove('EOS')
    alphabet = tuple(alphabet)

    for j in range(n_states):
        transition_weights[j] = {}
        transitions[j] = {}
        for symbol in alphabet:        
            transition_weights[j][symbol]=M.transition_weights[j][symbol]
            transitions[j][symbol] = M.transitions[j][symbol]

    if changetype == 'chg_nstates':
        n_states_new = n_states
        for idx in range(modif_amount):
            transitions = remove_one_state(transitions, n_states_new, alphabet)
            n_states_new -= 1
          
    else:
        for state in range(n_states):
            transitions[state] = M.transitions[state]
            start_val = transition_weights[state][0]
            for idx in range(modif_amount):
                if changetype == 'chg_dist_all':
                    chg_distr_state(alphabet, transition_weights, state, totalsteps, start_val)

                elif changetype == 'chg_dist_one':
                    if state == n_states - 1:
                        chg_distr_state(alphabet, transition_weights, state, totalsteps, start_val)

    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

# make one incremental step, towards making the distribution have Pr(0) ~= 1
# knowing that we started from start_val and that this function will be called totalsteps times
def chg_distr_state(alphabet, transition_weights, state, totalsteps, start_val):

    max_sum = sum(transition_weights[state].values()) # probabilities may sum to less than 1 (cause Pr($)=0.1 and it isn't here)
    step = (max_sum - start_val)/(totalsteps - 1) # by how much we increase Pr(0) at each step

    # 1. increase Pr(0)
    # if Pr(0) exceeds 1, there's something really wrong
    assert round(transition_weights[state][0] + step, 7) <= max_sum or math.isclose(transition_weights[state][0] + step, max_sum, rel_tol = 1e-05)
    transition_weights[state][0] += step

    # 2. now we need to remove the same amount from Pr(symbols other than 0)
    mass_to_remove = step 
    sum_of_other_probs = sum(value for key, value in transition_weights[state].items() if key != 0)
    assert sum_of_other_probs > mass_to_remove or math.isclose(sum_of_other_probs, mass_to_remove, abs_tol = 1e-04)
    for i in range(1, len(transition_weights[state])):
        if transition_weights[state][i] - mass_to_remove >= 0 or math.isclose(transition_weights[state][i] - mass_to_remove, 0, abs_tol = 1e-04):
            transition_weights[state][i] -= mass_to_remove
            break
        elif transition_weights[state][i] > 0.001:
            mass_to_remove -= transition_weights[state][i]
            transition_weights[state][i] = 0

    # 3. if a weight is 0, make make it almost 0 instead. making probabilities 0
    # 'erases' some states, making them unreachable, which we don't want to deal with right now
    for i in range(len(transition_weights[state])):
        if transition_weights[state][i] < 0.001:
            fix = 0.001 - transition_weights[state][i]
            transition_weights[state][i] = 0.001
            transition_weights[state][0] -= fix

        # there might be a cleaner way, just needed to make it add up to <=1, and this works for now
        transition_weights[state][i] = round_down(transition_weights[state][i], 7)

    assert sum(transition_weights[state].values()) <= 1

def round_down(x, precision):
    decimal_pt_shift = 10 ** precision
    return math.floor(x * decimal_pt_shift) / decimal_pt_shift


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


# discounted: means we're bounding a discounted sum of discrepancies (rhos) as we read the words
# not discounted: means we're bounding the discrepancy (rho) at the end of reading a word
def get_performance_bound(dist, alpha, discounted=True, maxlen=-1):
    if discounted:
        return dist/alpha
    else:
        return dist / (alpha*(1-alpha)**maxlen)


# discounted sum of discrepancies between M and N as we read w
def get_perf_actual_discounted_1word(M, N, w, alpha, rho):
    sum = 0
    w = str_to_ints(w)
    for i in range(len(w) + 1):
        w_subs = w[0:i]
        #w_ints = tuple(map(int, list(w[0:i])))
        qM = M.state_after_word(w_subs)
        qN = N.state_after_word(w_subs)
        sum += rho(M, N, qM=qM, qN=qN) * (1 - alpha)**i
    return sum

# average across test set
def get_perf_actual_discounted(M, N, alpha, rho, size):
    test_set = get_test_set(N, size)
    score = 0
    for w in test_set:
        score += get_perf_actual_discounted_1word(M, N, w, alpha, rho)
    return score / len(test_set)


def get_perf_actual_undiscounted(M, N, rho, size, maxlen):
    test_set = get_test_set(N, size, maxlen=-1)
    score = 0
    for w in test_set:
        score += rho(M, N, w=w)
    return score / len(test_set)


# N is the 'blackbox' or the 'bigger automaton' that M is supposed to be an approximation of
def get_test_set(N, size, maxlen=-1):
    samples = []
    for i in range(size):
        word = ''
        state = N.state_after_word(word)
        next_symbol_dist = N.state_probs_dist(state)

        # draw from next symbol distribution
        symbols = N.internal_alphabet
        symbol = np.random.choice(symbols, p=next_symbol_dist)

        while symbol != 'EOS':
            if maxlen > 0 and len(word) > maxlen-1:
                break
            word = word+symbol
            # keep drawing from next symbol distrivution
            symbol = np.random.choice(symbols, p=next_symbol_dist)
      
        samples.append(word)
    #print(f'samples:{samples}')
    return samples


# result is one chg_type with either d_as_difference_increases or delta_as_difference_increases
# both objects are structured like so:
# y axis: bounds, actual_vals
# x axis: assumed to be i in range (steps)
def plot_results(result, steps, resultpath):
    colors = {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}
    mpl.style.use('seaborn')
    print(result)
    #x = list(range(steps))
    three_types_of_lines = False
    for i, res in enumerate(result['results']):
        y = res['data']
        lbl = res['label']
        if res['type'] == 'bound':
            line_style = '--'
        elif res['type'] == 'actual':
            line_style = '-'
        else:
            line_style = ':'
            three_types_of_lines = True
        
        if three_types_of_lines:
            clr = f'C{math.floor(i/3.0)}'
        else:
            clr = f'C{math.floor(i/2.0)}'

        #plt.plot(x, y, line_style, color=clr, label=lbl)
        plt.plot(y, line_style, color=clr, label=lbl, marker='.')
 
    
    #plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
    plt.minorticks_on()
    #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #     plt.ylim(0.5, 1)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fancybox=True, shadow=True)
    plt.xlabel(result['xlabel'])
    plt.ylabel(result['ylabel'])
    plt.tight_layout()
    #plt.suptitle(result['graph_name'])
    #plt.text(5, 10, result['graph_name'], ha='center', va='bottom', wrap=True)
    sn = result['short_name']
    results_filename = f'{resultpath}/{sn}.png'
    plt.savefig(results_filename)
    plt.close()


# quick hack copypaste of the other plot method, with some tweaks
def plot_results_subplts(pdfa_results, steps, resultpath):
    colors = {'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}
    mpl.style.use('seaborn')
    print(pdfa_results)
    
    ylimit = getmax(pdfa_results) + 1
    xlimit = get_xlim(pdfa_results)
    
    print(f'ylim: {ylimit}')
    #plt.figure(figsize=(14,7))
    plt.ylabel(pdfa_results['ylabel'])

    for j, pdfa in enumerate(pdfa_results['results']):
        for i, res in enumerate(pdfa):
            y = res['data']
            lbl = res['label']
            if res['type'] == 'bound':
                line_style = '--'
            elif res['type'] == 'actual':
                line_style = '-'
            else:
                line_style = ':'
            
            clr = f'C{j}'
            plt.subplot(1, len(pdfa_results['results']), j+1)
            plt.ylim(-0.2, ylimit)
            plt.xlim(0, xlimit)
            plt.plot(y, line_style, color=clr, label=lbl, marker='|')
            plt.xlabel(res['xlabel'])
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fancybox=True, shadow=True)
    
    plt.minorticks_on()

   
    # plt.xlabel(pdfa_results['xlabel'])
    
    plt.tight_layout()
    #plt.subplots_adjust(hspace=1.0, wspace=1.0)
    sn = pdfa_results['short_name']
    results_filename = f'{resultpath}/{sn}.png'
    plt.savefig(results_filename)
    plt.close()


def getmax(pdfa_results):
    curr_max = 0
    for j, pdfa in enumerate(pdfa_results['results']):
        for i, res in enumerate(pdfa):
            temp = max(res['data'])
            curr_max = max(temp, curr_max)
    return curr_max

def get_xlim(pdfa_results):
    curr_max = 0
    for j, pdfa in enumerate(pdfa_results['results']):
        for i, res in enumerate(pdfa):
            temp = len(res['data'])
            curr_max = max(temp, curr_max)
    return curr_max


