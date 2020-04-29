import numpy as np
from toy_pdfa import toy_pdfa1, toy_pdfa2, toy_pdfa3, toy_pdfa4, toy_pdfa5, toy_pdfa6, toy_pdfa7, toy_pdfa8, toy_pdfa9, toy_pdfa10, toy_pdfa11, toy_pdfa12, uhl1_last_st, uhl1_first_st, uhl1_add_st, uhl1_remove_st, toy_pdfa_10statesA, toy_pdfa_10statesB
from our_grammars import uhl1, uhl2, uhl3
from bound_metric import get_vasilevskii_test_set
resultfolder = 'results/compute_metric/'

# TODO dict with shorter names
CHG_NSTATES = 'number of states'
CHG_DIST_ALL = 'next state distribution of all states by 0.01'
CHG_DIST_ONE = 'next state distribution of one state by 0.01'
change_types = [CHG_NSTATES, CHG_DIST_ALL, CHG_DIST_ONE]

def plot_results(result):
    pass # TODO


def plot_6(alpha):
    steps = 10
    Ms = [toy_pdfa_10statesA(), toy_pdfa_10statesB(), uhl1(), uhl2(), uhl3()]
    results = []
    for change_type in change_types:
        results.append(d_as_difference_increases(Ms, change_type, steps, alpha))

        M = Ms[0] # TODO decide which
        results.append(delta_as_difference_increases(M, change_type, steps, alpha))
    
    for result in results:
        plot_results(result)


# d between M and different variations of it
def d_as_difference_increases(Ms, changetype, steps, alpha):
    results = []
    for M in Ms:
        bounds, actual_vals = [], []
        for i in range (steps):
            N = get_modified_aut(M, i, changetype)
            upper_bound, true_dist = compare_truedist_vs_bound(M, N, alpha, f'{M.informal_name}_{changetype}_{i}')
            bounds.append(upper_bound)
            actual_vals.append(true_dist)
            x.append(i)
        Logger.log(f'Results as we modify the {changetype}:\nUpper bounds: {upper_bounds} \nTrue distances: {true_dists}')
        results.append({'label': f'M = {M.informal_name}', 'type': 'bound', 'data': bounds})
        results.append({'label': f'M = {M.informal_name}', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of distances between M and its modified versions. Each incremental modification of M reduces the {changetype}'
    sname = f'd_{changetype}'
    return {'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'd'}


# difference in predictions, for different words, for M and different variations of it
def delta_as_difference_increases(M, changetype, steps, alpha):
    ws = ['a', 'aa', 'aaa'] # TODO decide which
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
        results.append({'label': f'w = {w}', 'type': 'bound', 'data': bounds})
        results.append({'label': f'w = {w}', 'type': 'actual', 'data': actual_vals})
    gname = f'Graph of discrepancy between the outputs of {M.informal_name} and its modified versions as we read different input words. Each incremental modification of M reduces the {changetype}'
    sname = f'delta_{M.informal_name}_{changetype}'
    return 'results': results, 'graph_name': gname, 'short_name': sname, 'xlabel': 'change_amount', 'ylabel': 'discr(w)'}
    

# TODO test all 3 cases
def get_modified_aut(M, i, changetype):
    M_states = list(M.check_reachable_states())
    n = len(M_states)
    informal_name = f'{M.informal_name}_{i}_{changetype}'
    transitions = {}
    transition_weights = {}
    alphabet = M.internal_alphabet
            
    for state in range(n):
        if changetype == CHG_NSTATES:
 
            if state < n - i:
                transitions[state] = M.transitions[state]
                transition_weights[state]=M.transitions[state]

                # to remove the states, we're just going to reroute all their incoming transitions to their outgoing transitions
                for symbol in M.input_alphabet:
                    if M.transitions[state][symbol] >= n - i:
                        removed_state = M.transitions[state][symbol]
                        transitions[state][symbol] = M.transitions[removed_state][symbol]

                        # if still pointing to the removed state, then just loop back to itself
                        if transitions[state][symbol] == removed_state:
                            transitions[state][symbol] = state
                
        elif changetype == CHG_DIST_ALL:
            transitions[state] = M.transitions[state]
            transition_weights[state]=M.transitions[state]
            transition_weights[state][alphabet[0]] += i * 0.01
            transition_weights[state][alphabet[-1]] -= i * 0.01

        elif changetype == CHG_DIST_ONE:
            transitions[state] = M.transitions[state]
            transition_weights[state]=M.transitions[state]
            if state == n - 1:
                transition_weights[state][alphabet[0]] += i * 0.01
                transition_weights[state][alphabet[-1]] -= i * 0.01

    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def get_delta_w_bound(upper_bound, alpha)):
    return upper_bound/alpha


# TODO test
# discounted sum of discrepancies between M and N as we read w
def get_delta_w_actual(M, N, w, alpha):
    sum = 0
    for i in range(len(w)):
        w_ints = tuple(map(int, list(w[0:i])))
        qM = M.state_after_word(w_ints)
        qN = N.state_after_word(w_ints)
        sum += rho_pdfas(M, N, qM, qN) * (1 - alpha)**i
    return sum


def main():
    alpha = 0.2
    plot_6(alpha)
    
    # compute_d(toy_pdfa6(), toy_pdfa7(), 0.2, 'example3') # expected: 0.4444444444444445
    # compute_d(toy_pdfa6(), toy_pdfa8(), 0.2, 'example4') # expected: 0.5555555555555556
    # compute_d(toy_pdfa9(), toy_pdfa10(), 0.2, 'example5') # expected: 0.4
    # compute_d(toy_pdfa6(), toy_pdfa11(), 0.2, 'example8') # expected: 0.3555555555555556
    # compute_d(toy_pdfa6(), toy_pdfa12(), 0.2, 'example9') # expected: 0.2622950819672131
    # compute_d(uhl1(), toy_pdfa12(), 0.2, 'example10') # expected: ?
    # compute_d(uhl1(), uhl3(), 0.2, 'example11') # expected: ?
    # compute_d(toy_pdfa6(), uhl1(), 0.2, 'example12') # expected: ?

    # same, but on the last state, the transition probabilities are changed by 0.01
    #compare_truedist_vs_bound(toy_pdfa3(), toy_pdfa5(), 0.2, 'example1') # expected: 0.4
    #compare_truedist_vs_bound(uhl1(), uhl1_first_st(), 0.2, 'uhl1_first_st') # expected: something big
    #compare_truedist_vs_bound(uhl1(), uhl1_last_st(), 0.2, 'uhl1_last_st') # expected: something small
    #compare_truedist_vs_bound(uhl1(), uhl1_add_st(), 0.2, 'uhl1_add_st') # expected: something small
    #compare_truedist_vs_bound(uhl1(), uhl1_remove_st(), 0.2, 'uhl1_remove_st') # expected: something small but bigger

def compare_truedist_vs_bound(M, N, alpha, filename):
    dist, count = compute_d(M, N, alpha, filename)
    n = len(N.check_reachable_states())
    test_words = get_vasilevskii_test_set(M, n)
    Logger.log(f'test words: {test_words}')
    upper_bound = bound_d(M, N, '', alpha, test_words, True)
    msg = f'M: {M.informal_name}, N: {N.informal_name}\nestimated distance (upper bound): {upper_bound}\nactual distance: {dist}, found after {count} iterations'
    Logger.log(msg)
    return upper_bound, dist


def bound_d(M, N, w, alpha, test_words, is_upper_bound):

    longest_wordlen = max(k for k in test_words.keys())
    w_ints = tuple(map(int, list(w)))
    qM = M.state_after_word(w_ints)
    qN = N.state_after_word(w_ints)
    if len(w) >= longest_wordlen: # base case
        x = 1 if is_upper_bound else 0
        return alpha * rho_pdfas(M, N, qM, qN) + (1 - alpha) * x
    
    else: # recursive case
        a = alpha * rho_pdfas(M, N, qM, qN)
        biggest = 0
        for next_w in test_words[len(w)+1]:
            if not w == next_w[:len(w)]: continue
            b = bound_d(M, N, next_w, alpha, test_words, is_upper_bound)
            biggest = max(biggest, b)
        return a + (1 - alpha) * biggest



# in: M, N, accuracy thershold maybe
def compute_d(M, N, alpha, filename):
    M.draw_nicely(keep=True,filename=resultfolder+filename+'/M')
    N.draw_nicely(keep=True,filename=resultfolder+filename+'/N')
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
                distances[M_state_row][N_state_col] = alpha*rho_pdfas(M, N, qM, qN) + (1 - alpha)*max_next_dist
                
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


def rho_pdfas(M, N, qM, qN):
    # w = tuple(map(int, list(w)))
    Mw = M.transition_weights[qM]
    Nw = N.transition_weights[qN]
    biggest = 0
    for a in M.input_alphabet:
        biggest = max(biggest, abs(Mw[a] - Nw[a]))
    #M_eos = Mw['<EOS>']
    #N_eos = Nw['<EOS>']
    #biggest = max(biggest, abs(M_eos - N_eos))
    return biggest

def toy_pdfa(): # just a small pdfa to test that the spanning tree is done correctly
    informal_name = "toy_pdfa"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 2}
    transitions[1] = {a: 3, b: 1}
    transitions[2] = {a: 3, b: 4}
    transitions[3] = {a: 2, b: 1}
    transitions[4] = {a: 0, b: 4}
    for i in range(5):
        transition_weights[i]={a:0.75,b:0.15}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


class Logger():
    def __init__(self, path):
        self.logfile = open(path,"w+")
    def log(self, msg):
        print(msg)
        self.logfile.write(msg)

start_time = time.time()
resultpath = 'results/' + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
try:
    os.mkdir(resultpath)
    logfilepath = f'{resultpath}/log.txt'
    logger = Logger(logfilepath)

except OSError:
    print(OSError.errno)
    print('Could not create results directory!')

main()

elapsed_time = time.time() - start_time
logger.log(f'\ntime elapsed: {elapsed_time}')
