import numpy as np
from toy_pdfa import toy_pdfa1, toy_pdfa2, toy_pdfa3, toy_pdfa4, toy_pdfa5, toy_pdfa6, toy_pdfa7
resultfolder = 'results/compute_metric/'

def main():
    compute_d(toy_pdfa3(), toy_pdfa5(), 0.2, 'example1')
    compute_d(toy_pdfa6(), toy_pdfa7(), 0.2, 'example3')

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
                max_next_dist = find_max_next_dist(M_state_row, N_state_col, distances, M_states, N_states, M, N)
                qM = M_states[M_state_row]
                qN = N_states[N_state_col]
                distances[M_state_row][N_state_col] = alpha*rho(M, N, qM, qN) + (1 - alpha)*max_next_dist
                
                if distances[M_state_row][N_state_col] != old_dist:
                    changed = True
    
    f = open(resultfolder+filename+'/log.txt', 'w')
    msg = f'M: {M.informal_name}, N: {N.informal_name}\ndistance is {distances[0][0]}, found after {count} iterations'
    f.write(msg)
    print(msg)
    f.close()
    return distances[0][0] # d(qM0, qN0) distance between initial states of M, N

def find_max_next_dist(qM, qN, distances, M_states, N_states, M, N):
    # maybe the worst case is if we see the stop symbol and stay in this state
    # TODO: should we do this case?
    #biggest = distances[M_states.index(qM)][M_states.index(qN)]
    biggest = 0
    for a in M.input_alphabet:
        row_idx = M_states.index(M.next_state(qM, a))
        col_idx = N_states.index(N.next_state(qN, a))
        biggest = max(biggest, distances[row_idx][col_idx])
    return biggest


def rho(M, N, qM, qN):
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

main()
