from compute_metric import rho_pdfas, compute_d, compare_truedist_vs_bound
from toy_pdfa import toy_pdfa1, toy_pdfa2, toy_pdfa3, toy_pdfa4, toy_pdfa5, toy_pdfa6, toy_pdfa7, toy_pdfa8, toy_pdfa9, toy_pdfa10, toy_pdfa11, toy_pdfa12, toy_pdfa13, toy_pdfa14
# for each:
# 1. happy path
# 2. list the possible kinds of inputs, as well as boundaries/edge cases
# 3. list the different code paths
# 4. run and assert it's right

# compute metric
def main():
    test_rho_pdfas()
    test_compute_d()
    test_bound_d()
    test_compare_truedist_vs_bound()


    test_construct_test_words()
    test_get_all_words_of_len()
    # TODO call the rest
    

def test_rho_pdfas():
    assert rho_pdfas(toy_pdfa3(), toy_pdfa5(), 0, 0) == 0
    assert rho_pdfas(toy_pdfa3(), toy_pdfa5(), 0, 1) == 0.5
    assert rho_pdfas(toy_pdfa6(), toy_pdfa7(), 0, 1) == 1
    assert round(rho_pdfas(toy_pdfa13(), toy_pdfa14(), 0, 0),1) == 0.3

def test_compute_d():
    assert compute_d(toy_pdfa6(), toy_pdfa6(), 0.2, 'example0')[0] == 0 
    assert compute_d(toy_pdfa6(), toy_pdfa7(), 0.2, 'example3')[0] == 0.4444444444444445
    assert compute_d(toy_pdfa6(), toy_pdfa8(), 0.2, 'example4')[0] == 0.5555555555555556
    assert round(compute_d(toy_pdfa9(), toy_pdfa10(), 0.2, 'example5')[0],5) == 0.4
    assert compute_d(toy_pdfa6(), toy_pdfa11(), 0.2, 'example8')[0] == 0.3555555555555556
    assert compute_d(toy_pdfa6(), toy_pdfa12(), 0.2, 'example9')[0] == 0.26229508196721313


def test_compare_truedist_vs_bound():
    bound, truedist, msg = compare_truedist_vs_bound(toy_pdfa3(), toy_pdfa5(), 0.2, 'example1')
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
