from metrics.metric import rho_rnn, rho_pdfas, str_to_ints
from weighted_lstar.PDFA import PDFA

# M: known PDFA, N: blackbox
# search_type: 'bfs' or 'all_paths', all paths is pretty much useless
# max_depth: for bfs, we'll search all words of length up to max_depth
# max_revisits: for all paths search: how many times can a word revisit same state
def get_d_estimate(M, N, alpha, search_type='bfs', max_depth=5, max_revisits=0):
    count = 0
    rho = rho_pdfas if isinstance(N, PDFA) else rho_rnn
    queue = [
            {
                'state':M._initial_state, 
                'word':'', 
                'cost':alpha*rho(M, N, '')
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
                    next_cost = cur['cost'] + (1-alpha)**len(next_word) * alpha * rho(M, N, next_word)
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
    print(f'found costliest path {cp} with cost {cc} after considering {count} paths')
    return costliest_path['cost'], costliest_path['path'], count

def num_revisits(M, w, s):
    count = 0
    for i in range(len(w)):
        t = M.state_after_word(str_to_ints(w[:i]))
        if t == s: count += 1
    return count