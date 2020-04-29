from compute_metric import rho_pdfas
from toy_pdfa import toy_pdfa1, toy_pdfa2, toy_pdfa3, toy_pdfa4, toy_pdfa5, toy_pdfa6, toy_pdfa7, toy_pdfa13, toy_pdfa14
# for each:
# 1. happy path
# 2. list the possible kinds of inputs, as well as boundaries/edge cases
# 3. list the different code paths
# 4. run and assert it's right

# compute metric
def main():
    test_rho_pdfas()
    test_find_max_next_dist()
    test_compute_d()
    test_bound_d()

def test_rho_pdfas():
    assert rho_pdfas(toy_pdfa3(), toy_pdfa5(), 0, 0) == 0
    assert rho_pdfas(toy_pdfa3(), toy_pdfa5(), 0, 1) == 0.5
    assert rho_pdfas(toy_pdfa6(), toy_pdfa7(), 0, 1) == 1
    assert round(rho_pdfas(toy_pdfa13(), toy_pdfa14(), 0, 0),1) == 0.3

def test_find_max_next_dist():
    distances = np.zeros((len(M_states), len(N_states)))
    find_max_next_dist(0, 0, distances, toy_pdfa3(), toy_pdfa5())
    pass

def test_compute_d():
    pass

def test_bound_d():
    pass

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

def test_bound_d():
    pass

def test_get_vasilevskii_test_set():
    pass

main()
