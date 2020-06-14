# import sys
# sys.path.append('../')
# print(sys.path)
import math
from metrics.metric import compute_d, rho_pdfas, rho_pdfas_states, compare_truedist_vs_bound 
from metrics.metric_on_known_pdfa import get_modified_aut, get_delta_w_actual
from metrics.toy_pdfa import M_for_bf_test, N_for_bf_test, N_for_bf_test2, toy_pdfa1, toy_pdfa2, toy_pdfa3, toy_pdfa4, toy_pdfa5, toy_pdfa6, toy_pdfa7, toy_pdfa8, toy_pdfa9, toy_pdfa10, toy_pdfa_10statesA, toy_pdfa11, toy_pdfa12, toy_pdfa13, toy_pdfa14
from metrics.brute_force_bound import get_d_estimate
resultfolder = 'results/unit_tests/'
# for each:
# 1. happy path
# 2. list the possible kinds of inputs, as well as boundaries/edge cases
# 3. list the different code paths
# 4. run and assert it's right

# compute metric
def main():
    test_rho_pdfas_states()
    test_compute_d()
    test_bound_d()
    test_compare_truedist_vs_bound()
    test_chg_nstates()
    test_chg_dist_all()
    test_chg_dist_one()
    test_get_delta_w_actual()

    test_construct_test_words()
    test_get_all_words_of_len()

    test_brute_force_bound_all_paths()
    test_brute_force_bound_bfs()
    # TODO call the rest


def test_get_delta_w_actual():
    ws = ['0', '00', '000', '0000']
    expected = [0.4, 0.72, 0.976, 1.1808]
    for i, w in enumerate(ws):
        res = get_delta_w_actual(toy_pdfa3(), toy_pdfa5(), w, 0.2)
        #print(f'res: {res}')
        assert expected[i] == round(res, 4)

def test_chg_nstates():
    M = toy_pdfa_10statesA()
    expected_ds = [0, 0.1193, 0.192, 0.192, 0.192, 0.192, 0.2304, 0.2304, 0.224, 0.288]
    test_get_modified_aut(M, expected_ds, 'chg_nstates', 4)

def test_chg_dist_all():
    M = toy_pdfa_10statesA()
    expected_ds = [0, 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.719, 0.809, 0.898]
    test_get_modified_aut(M, expected_ds, 'chg_dist_all')

def test_chg_dist_one():
    M = toy_pdfa_10statesA()
    expected_ds = [0, 0.02, 0.039, 0.059, 0.078, 0.097, 0.117, 0.136, 0.156, 0.175, 0.195]
    test_get_modified_aut(M, expected_ds, 'chg_dist_one')


def test_get_modified_aut(M, expected_ds, change_type, round_amount=3):
    for change_amount in range (len(expected_ds)):
        steps = 10 if change_type == 'chg_nstates' else 11
        N = get_modified_aut(M, change_amount, change_type, steps)
        d = round(compute_d(M, N, 0.2)[0], round_amount)
        #print(f'change type: {change_type}, modification amount: {change_amount}, d: {d}')
        assert d == expected_ds[change_amount]

def test_brute_force_bound_all_paths():
    bound, path, count = get_d_estimate(M_for_bf_test(), N_for_bf_test(), 0.2, search_type='all_paths', max_revisits=0)
    actual = compute_d(M_for_bf_test(), N_for_bf_test(), 0.2)[0]
    print(f'brute force all-paths bound: {bound}, actual: {actual}')
    assert math.isclose(bound, 0.64, rel_tol=1e-05)
    assert path == '0x'

    bound, path, count = get_d_estimate(M_for_bf_test(), N_for_bf_test2(), 0.2, search_type='all_paths', max_revisits=0)
    actual = compute_d(M_for_bf_test(), N_for_bf_test2(), 0.2)[0]
    print(f'brute force all-paths bound: {bound}, actual: {actual}')
    assert math.isclose(bound, 0.712, rel_tol=1e-05)
    assert path == '0x'
    
def test_brute_force_bound_bfs():
    bound, path, count = get_d_estimate(M_for_bf_test(), N_for_bf_test(), 0.2, search_type='bfs', max_depth=3)
    actual = compute_d(M_for_bf_test(), N_for_bf_test(), 0.2)[0]
    print(f'brute force bfs bound: {bound}, actual: {actual}')

    bound, path, count = get_d_estimate(M_for_bf_test(), N_for_bf_test2(), 0.2, search_type='bfs', max_depth=3)
    actual = compute_d(M_for_bf_test(), N_for_bf_test2(), 0.2)[0]
    print(f'brute force bfs bound: {bound}, actual: {actual}')

def test_rho_pdfas_states():
    assert rho_pdfas_states(toy_pdfa3(), toy_pdfa5(), 0, 0) == 0
    assert rho_pdfas_states(toy_pdfa3(), toy_pdfa5(), 0, 1) == 0.5
    assert rho_pdfas_states(toy_pdfa6(), toy_pdfa7(), 0, 1) == 1
    assert round(rho_pdfas_states(toy_pdfa13(), toy_pdfa14(), 0, 0),1) == 0.3

def test_compute_d():
    assert compute_d(toy_pdfa6(), toy_pdfa6(), 0.2)[0] == 0 
    assert compute_d(toy_pdfa6(), toy_pdfa7(), 0.2)[0] == 0.4444444444444445
    assert compute_d(toy_pdfa6(), toy_pdfa8(), 0.2)[0] == 0.5555555555555556
    assert round(compute_d(toy_pdfa9(), toy_pdfa10(), 0.2)[0],5) == 0.4
    assert compute_d(toy_pdfa6(), toy_pdfa11(), 0.2)[0] == 0.3555555555555556
    assert compute_d(toy_pdfa6(), toy_pdfa12(), 0.2)[0] == 0.26229508196721313


def test_compare_truedist_vs_bound():
    n = len(toy_pdfa3().check_reachable_states()) + 3
    bound, truedist, msg = compare_truedist_vs_bound(toy_pdfa3(), toy_pdfa5(), 0.2, n)
    assert round(bound, 5) == 0.56384
    assert round(truedist, 5) == 0.4

# bound metric

def test_construct_test_words():
    pass

def test_get_all_words_of_len():
    pass

def test_pdfa_transition():
    pass

def test_try_using_existing_dstrings():
    pass

def test_construct_spanning_tree_words():
    pass

def test_rho():
    pass

def test_bound_d(): # also test the one in compute_metric.py or consolidate them into one method
    pass

def test_get_vasilevskii_test_set():
    pass

main()
