from PDFA import PDFA
from our_grammars import assert_and_give_pdfa

# small PDFA for tests
def toy_pdfa1(): # just a small pdfa to test that the spanning tree is done correctly
    informal_name = "toy_pdfa1"
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

def toy_pdfa2():
    informal_name = "toy_pdfa2"
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
        transition_weights[i]={a:0.8,b:0.2}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def toy_pdfa3():
    informal_name = "toy_pdfa3"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 0, b: 0}
    transition_weights[0]={a:0.75,b:0.25}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def toy_pdfa4(): 
    informal_name = "toy_pdfa4"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 0, b: 0}
    transition_weights[0]={a:0.25,b:0.75}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def toy_pdfa5(): 
    informal_name = "toy_pdfa5"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 1}
    transitions[1] = {a: 1, b: 1}
    transition_weights[0]={a:0.75,b:0.25}
    transition_weights[1]={a:0.25,b:0.75}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def toy_pdfa6(): 
    informal_name = "toy_pdfa6"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 0, b: 0}
    transition_weights[0]={a:1,b:0}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def toy_pdfa7(): 
    informal_name = "toy_pdfa7"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 1}
    transitions[1] = {a: 0, b: 0}
    transition_weights[0]={a:1,b:0}
    transition_weights[1]={a:0,b:1}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def toy_pdfa8(): 
    informal_name = "toy_pdfa8"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 1}
    transitions[1] = {a: 0, b: 0}
    transition_weights[0]={a:0,b:1}
    transition_weights[1]={a:1,b:0}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def toy_pdfa9(): 
    informal_name = "toy_pdfa9"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 0, b: 0}
    transition_weights[0]={a:0.5,b:0.5}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)


def toy_pdfa10():
    informal_name = "toy_pdfa10"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 2}
    transitions[1] = {a: 1, b: 1}
    transitions[2] = {a: 2, b: 2}
    transition_weights[0]={a:0.5,b:0.5}
    transition_weights[1]={a:1,b:0}
    transition_weights[2]={a:0,b:1}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def toy_pdfa11(): 
    informal_name = "toy_pdfa11"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 1}
    transitions[1] = {a: 2, b: 2}
    transitions[2] = {a: 1, b: 1}
    transition_weights[0]={a:1,b:0}
    transition_weights[1]={a:1,b:0}
    transition_weights[2]={a:0,b:1}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def toy_pdfa12(): 
    informal_name = "toy_pdfa12"
    transitions = {}
    transition_weights = {}
    
    alphabet = [0,1]
    a, b = alphabet
    transitions[0] = {a: 1, b: 1}
    transitions[1] = {a: 2, b: 2}
    transitions[2] = {a: 0, b: 0}
    transition_weights[0]={a:1,b:0}
    transition_weights[1]={a:1,b:0}
    transition_weights[2]={a:0,b:1}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)

def uhl1a():
    max_as=3
    informal_name = "uhl_1a"
    transitions = {}
    transition_weights = {}

    alphabet = [0,1]
    a,b = alphabet
    for i in range(sum(range(2,max_as+2))):
        transitions[i]={a:i+1,b:i+1}
        transition_weights[i]={a:0.75,b:0.15} 
    transitions[i]={a:0,b:0} # last one needs to loop back
    j=0
    for i in range(1,max_as+1):
        j+=i # skip the as
        transition_weights[j]={a:0.15,b:0.75} # places where b is higher
        j+=1 # skip the b
    transition_weights[sum(range(2,max_as+2))-1]={a:0.14,b:0.76}
    return assert_and_give_pdfa(informal_name,transitions,transition_weights,alphabet,0)