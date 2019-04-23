from nltk.corpus import treebank
from nltk.grammar import Nonterminal, Production
from nltk import CFG
from nltk.tree import Tree
import nltk
from collections import defaultdict, Counter
import numpy as np
import pickle

def backoff_to_unk(rules_to_prob, top=10000):
    terminal_nonterms = set(rule.lhs() for rule in rules_to_prob if len(rule.rhs()) == 1 and type(rule.rhs()[0]) == str)
    words = [wrd.lower() for wrd in treebank.words()]
    vocab = [wrd for wrd,freq in Counter(treebank.words()).most_common(top)]
    for nonterm in terminal_nonterms:
        nonterm_to_term = [rule for rule in rules_to_prob if rule.lhs() == nonterm and len(rule.rhs()) == 1 and type(rule.rhs()[0]) == str]
        unkp = 0.0
        removed_rules = []
        for rule in nonterm_to_term:
            if rule.rhs()[0] not in vocab:
                unkp += rules_to_prob[rule]
                removed_rules.append(rule)
        if len(removed_rules) > 0:
            for rule in removed_rules:
                del rules_to_prob[rule]
            unk_prod = Production(nonterm, ['UNK'])
            rules_to_prob[unk_prod] = unkp
    return rules_to_prob, vocab

def store_model(fname, data):
    f = open('%s.pkl' % (fname), 'wb')
    pickle.dump(data, f)
    f.close()

def generate():
    print("Generating PCFG dataset")
    tbank_trees = []
    for sent in treebank.parsed_sents():
        sent.chomsky_normal_form()
        tbank_trees.append(sent)

    tbank_productions = set(production for tree in tbank_trees for production in tree.productions())
    tbank_grammar = CFG(Nonterminal('S'), list(tbank_productions))
    production_rules = tbank_grammar.productions()

    rules_to_prob = defaultdict(int)
    nonterm_occurrence = defaultdict(int)
    for sent in tbank_trees:
        for production in sent.productions():
            if len(production.rhs()) == 1 and not isinstance(production.rhs()[0], Nonterminal):
                production = Production(production.lhs(), [production.rhs()[0].lower()])
            nonterm_occurrence[production.lhs()] += 1
            rules_to_prob[production] += 1

    for rule in rules_to_prob:
         rules_to_prob[rule] /= nonterm_occurrence[rule.lhs()]

    rules_to_prob, vocab = backoff_to_unk(rules_to_prob)

    rules = list(rules_to_prob.keys())
    rules_reverse_dict = dict((j,i) for i, j in enumerate(rules))
    left_rules = defaultdict(set)
    right_rules = defaultdict(set)
    unary_rules = defaultdict(set)
    for rule in rules:
        if len(rule.rhs()) > 1:
            left_rules[rule.rhs()[0]].add(rule)
            right_rules[rule.rhs()[1]].add(rule)
        else:
            unary_rules[rule.rhs()[0]].add(rule)
    terminal_nonterms_rules = set(rule for rule in rules_to_prob if len(rule.rhs()) == 1 and isinstance(rule.rhs()[0], str))
    terminal_nonterms = defaultdict(int)
    for rule in terminal_nonterms_rules:
        terminal_nonterms[rule.lhs()] += 1

    parser_data = {
        'vocab': vocab,
        'left_rules': left_rules,
        'right_rules': right_rules,
        'unary_rules': unary_rules,
        'rules_to_prob': rules_to_prob,
        'terminal_nonterms': terminal_nonterms
    }
    print("Done")
    return parser_data

if __name__ == '__main__':
    parser_data = generate()
    store_model('model', parser_data)
