from nltk.grammar import Nonterminal, Production
from nltk.tree import Tree
import nltk
from collections import defaultdict
import numpy as np
import pickle
import argparse

import Train

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', help='Specify sentence to generate parse tree', default='We were eating by the river' ,type=str)

def read_model(fname):
    f = open(fname, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def handle_unary(productions_dict, unary_rules, rules_to_prob):
    for lhs in list(productions_dict.keys()):
        for rule in unary_rules[lhs]:
            P = np.log(rules_to_prob[rule])
            if rule.lhs() not in productions_dict or P + productions_dict[lhs]['score'] > productions_dict[rule.lhs()]['score']:
                productions_dict[rule.lhs()]['rule'] = rule
                productions_dict[rule.lhs()]['score'] = P + productions_dict[lhs]['score']
                productions_dict[rule.lhs()]['back'] = lhs
                productions_dict[rule.lhs()]['back_type'] = 'unary'
    return productions_dict

def cky_parser(tokens, left_rules, right_rules, unary_rules, rules_to_prob, vocab, terminal_nonterms, backoff='UNK'):
    M = [[{} for _ in range(len(tokens)+1)] for _ in range(len(tokens)+1)]
    for l in range(1, len(tokens) + 1):
        for i in range(len(tokens) - l + 1):
            ts = tokens[i:l+i]
            print("Processing: ", ts)
            cur_prod_dict = defaultdict(dict)
            if l == 1:
                if tokens[i] in unary_rules and len(unary_rules[tokens[i]]) > 0:
                    for rule in unary_rules[tokens[i].lower()]:
                        cur_prod_dict[rule.lhs()] = {
                            'rule': rule,
                            'score': np.log(rules_to_prob[rule]),
                            'back': tokens[i],
                            'back_type': 'terminal'
                        }
                elif backoff == 'UNK':
                    for rule in unary_rules['UNK']:
                        cur_prod_dict[rule.lhs()] = {
                            'rule': Production(rule.lhs(), [tokens[i]]),
                            'score': np.log(rules_to_prob[rule]),
                            'back': tokens[i],
                            'back_type': 'terminal'
                        }
                elif backoff == 'EQL':
                    for nonterm in terminal_nonterms:
                        rule = Production(nonterm, [tokens[i]])
                        cur_prod_dict[rule.lhs()] = {
                            'rule': rule,
                            'score': np.log(1.0/len(terminal_nonterms)),
                            'back': tokens[i],
                            'back_type': 'terminal'
                        }
            for s in range(i+1, i+l):
                left_set = list(M[i][s].keys())
                right_set = list(M[s][i+l].keys())
                for left in left_set:
                    prodsl = left_rules[left]
                    for right in right_set:
                        prodsr = right_rules[right].intersection(prodsl)
                        for rule in prodsr:
                            P = np.log(rules_to_prob[rule])
                            nscore = P + M[i][s][left]['score'] + M[s][i+l][right]['score']
                            if rule.lhs() not in cur_prod_dict or nscore > cur_prod_dict[rule.lhs()]['score']:
                                cur_prod_dict[rule.lhs()]['rule'] = rule
                                cur_prod_dict[rule.lhs()]['score'] = nscore
                                cur_prod_dict[rule.lhs()]['back'] = [left, right, s]
                                cur_prod_dict[rule.lhs()]['back_type'] = 'binary_split'
            M[i][i+l] = handle_unary(cur_prod_dict, unary_rules, rules_to_prob)
            if len(M[i][i+l]) == 0 and l == 1:
                print("Failed to generate any productions for '%s' substring" % (' '.join(ts)))
                return M
            #print("M[%d][%d] = " % (i, i+l), M[i][i+l])
    return M

def print_tree_from_array(M, nonterm, beg, end):
    rule_dict = M[beg][end][nonterm]
    tree_string = ''
    count = 0
    tree_string += ' ' + str(rule_dict['rule'].lhs()) + ' '
    while rule_dict['back_type'] == 'unary':
        tree_string += ' ( '
        rule_dict = M[beg][end][rule_dict['back']]
        tree_string += ' ' + str(rule_dict['rule'].lhs()) + ' '
        count += 1
    if rule_dict['back_type'] == 'terminal':
        tree_string += ' ' + rule_dict['back'] + ' '
    if rule_dict['back_type'] == 'binary_split':
        l, r, s = rule_dict['back']
        left_string = print_tree_from_array(M, l, beg, s)
        right_string = print_tree_from_array(M, r, s, end)
        tree_string += ' ( ' + left_string + ' ) ( ' + right_string + ' ) '
    for _ in range(count):
        tree_string += ' ) '
    return tree_string

def tree(M):
    t = Tree.fromstring('(' + print_tree_from_array(M, Nonterminal('S'), 0, len(M)-1) + ')')
    #t.pretty_print()
    return t

def parse_sentence(sentence, parser_data):
    vocab = parser_data['vocab']
    left_rules = parser_data['left_rules']
    right_rules = parser_data['right_rules']
    unary_rules = parser_data['unary_rules']
    rules_to_prob = parser_data['rules_to_prob']
    terminal_nonterms = parser_data['terminal_nonterms']
    tokens = [tok for tok in nltk.word_tokenize(sentence)]
    M = cky_parser(tokens, left_rules, right_rules, unary_rules, rules_to_prob, vocab, terminal_nonterms)
    return M

if __name__ == '__main__':
    args = parser.parse_args()
    parser_data = Train.generate()
    M = parse_sentence(args.sentence, parser_data)
    t = tree(M)
    t.pretty_print()
