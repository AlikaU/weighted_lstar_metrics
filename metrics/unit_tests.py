# import sys
# sys.path.append('../')
# print(sys.path)
from metrics.metric import compute_d, rho_pdfas, rho_pdfas_states, compare_truedist_vs_bound 
from metrics.metric_on_known_pdfa import get_modified_aut, get_delta_w_actual
from metrics.toy_pdfa import toy_pdfa1, toy_pdfa2, toy_pdfa3, toy_pdfa4, toy_pdfa5, toy_pdfa6, toy_pdfa7, toy_pdfa8, toy_pdfa9, toy_pdfa10, toy_pdfa_10statesA, toy_pdfa11, toy_pdfa12, toy_pdfa13, toy_pdfa14
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
    expected_ds = [0, 0.2202, 0.2534, 0.2534, 0.2534, 0.2534, 0.2534, 0.3424, 0.368, 0.368]
    test_get_modified_aut(M, expected_ds, 'chg_nstates')

def test_chg_dist_all():
    M = toy_pdfa_10statesA()
    expected_ds = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    test_get_modified_aut(M, expected_ds, 'chg_dist_all')

def test_chg_dist_one():
    M = toy_pdfa_10statesA()
    expected_ds = [0, 0.00217, 0.00434, 0.0065, 0.00867, 0.01084, 0.01301, 0.01518, 0.01734, 0.01951, 0.02168]
    test_get_modified_aut(M, expected_ds, 'chg_dist_one', 5)


def test_get_modified_aut(M, expected_ds, change_type, round_amount=4):
    for change_amount in range (len(expected_ds)):
        N = get_modified_aut(M, change_amount, change_type)
        d = round(compute_d(M, N, 0.2)[0], round_amount)
        print(f'change type: {change_type}, modification amount: {change_amount}, d: {d}')
        assert d == expected_ds[change_amount]

    

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
    bound, truedist, msg = compare_truedist_vs_bound(toy_pdfa3(), toy_pdfa5(), 0.2)
    assert round(bound, 5) == 0.656
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
