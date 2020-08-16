import time
from datetime import datetime

from metrics.metric_on_known_pdfa import plot_6
from metrics.rhos import rho_infty_norm, rho_kt
from metrics.toy_pdfa import toy_pdfa_10statesA, toy_pdfa_10statesB
from weighted_lstar.our_grammars import uhl1, uhl2, uhl3
log_results = False

def main():
    log_results = True
    alpha = 0.2
    steps = 10
    PDFAs = [toy_pdfa_10statesA(), toy_pdfa_10statesB(), uhl1(), uhl2(), uhl3()]
    #PDFAs = [uhl1(), uhl2(), uhl3()]
    #PDFAs = [toy_pdfa_10statesA(), toy_pdfa_10statesB()]
    #PDFAs = [uhl2()]

    max_depth_addition = 1 # for all-words search, we search up to words of length = depth(M) + max_depth_addition
    max_rev = 1             # for all-paths search, we search paths that revisit the same state at most max_rev times
    #rho = rho_infty_norm
    rho = rho_kt
    #plot_6(PDFAs, alpha, steps, resultpath, logger, test_depth=3, bound_type='all_path', max_depth_add=max_depth_addition, max_revisits=max_rev)
    plot_6(PDFAs, alpha, rho, steps, resultpath, logger, test_depth=3, bound_type='bfs', max_depth_add=max_depth_addition, max_revisits=max_rev)
    
    # compute_d(uhl1(), toy_pdfa12(), 0.2, 'example10') # expected: ?
    # compute_d(uhl1(), uhl3(), 0.2, 'example11') # expected: ?
    # compute_d(toy_pdfa6(), uhl1(), 0.2, 'example12') # expected: ?

    # same, but on the last state, the transition probabilities are changed by 0.01
    #compare_truedist_vs_bound(uhl1(), uhl1_first_st(), 0.2, 'uhl1_first_st') # expected: something big
    #compare_truedist_vs_bound(uhl1(), uhl1_last_st(), 0.2, 'uhl1_last_st') # expected: something small
    #compare_truedist_vs_bound(uhl1(), uhl1_add_st(), 0.2, 'uhl1_add_st') # expected: something small
    #compare_truedist_vs_bound(uhl1(), uhl1_remove_st(), 0.2, 'uhl1_remove_st') # expected: something small but bigger


class Logger():
    def __init__(self, path):
        if log_results:
            self.logfile = open(path,"w+")
    def log(self, msg):
        print(msg)
        if log_results:
            self.logfile.write(msg)

start_time = time.time()
resultpath = 'results/metric/' + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")

if log_results:
    try:
        os.mkdir(resultpath)
        logfilepath = f'{resultpath}/log.txt'
        logger = Logger(logfilepath)

    except OSError:
        print(OSError.errno)
        print('Could not create results directory!')
else:
    logger = Logger('')

main()

elapsed_time = time.time() - start_time
logger.log(f'\ntime elapsed: {elapsed_time}')
