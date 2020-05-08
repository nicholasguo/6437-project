import numpy as np
import pandas as pd
import random
import math

INF = 1000000000
MAX_TRANS = 150000000
# NUM_RESTARTS = 10
# DECODER = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, '.': 26, ' ': 27}
DECODER = {}
alphabet = open("data/alphabet.csv", "r").readline().strip().split(",")
for idx, char in enumerate(alphabet):
    DECODER[char] = idx
# letter_transition_matrix = genfromtxt('data/letter_transition_matrix.csv', delimiter=',')
letter_transition_matrix = pd.read_csv('data/letter_transition_matrix.csv', sep=',',header=None).values.tolist()
for i in range(len(alphabet)):
    for j in range(len(alphabet)):
        if letter_transition_matrix[i][j] == 0:
            letter_transition_matrix[i][j] = -INF
        else:
            letter_transition_matrix[i][j] = math.log(letter_transition_matrix[i][j])

# letter_probabilities = [math.log(x) for x in genfromtxt('data/letter_probabilities.csv', delimiter=',')]
letter_probabilities = [math.log(x) for x in pd.read_csv('data/letter_probabilities.csv', sep=',',header=None).values.tolist()[0]]

def apply_perm(ciphertext, permutation):
    plaintext = ''
    for char in ciphertext:
        plaintext += permutation[DECODER[char]]

    return plaintext

def find_space_period(ciphertext):
    trans = {}
    for i in range(1, len(ciphertext)):
        if ciphertext[i-1] not in  trans.keys():
            trans[ciphertext[i-1]] = {}
        if ciphertext[i] not in trans[ciphertext[i-1]].keys():
            trans[ciphertext[i-1]][ciphertext[i]] = 0
        trans[ciphertext[i-1]][ciphertext[i]] += 1
    for char in alphabet:
        if char in trans.keys() and len(trans[char].keys()) == 1:
            dot = char
            space = list(trans[char].keys())[0]
            chars = list(alphabet)
            chars.remove('.')
            chars.remove(' ')
            if DECODER[dot] < DECODER[space]:
                chars.insert(DECODER[dot], '.')
                chars.insert(DECODER[space], ' ')
            else:
                chars.insert(DECODER[space], ' ')
                chars.insert(DECODER[dot], '.')
            return "".join(chars)
    print("SOMETHINGS WRONG")

def swap_perm(permutation, x, y):
    if y == x:
            y = 0
    x, y = max(x, y), min(x, y)
    return permutation[:y] + permutation[x] + permutation[y+1:x] + permutation[y] + permutation[x+1:]

def log_likelihood(ciphertext, permutation):
    plaintext = apply_perm(ciphertext, permutation)
    prob = letter_probabilities[DECODER[plaintext[0]]]
    for i in range(1, len(ciphertext)):
        prob += letter_transition_matrix[DECODER[plaintext[i]]][DECODER[plaintext[i-1]]]
    return prob

def accept_prob(p1, p2, v1 = 0, v2 = 0):
    if p2 < -INF:
        return 1
    if p1 - p2 + v1 - v2 > 0:
        return 1
    return math.exp(p1 - p2 + v1 - v2)

def best_permutation(ciphertext, trans, DEBUG=False):
    best_permutation = "".join(alphabet)
    best_likelihood = -INF
    iters = 0
    while trans > 0 and iters < 50:
        iters += 1
        cipherbet = alphabet.copy()
        random.shuffle(cipherbet)
        permutation = "".join(cipherbet)
        # permutation = find_space_period(ciphertext)
        p2 = log_likelihood(ciphertext, permutation)
        count = 0
        lol = 0
        while trans > 0:
            lol += 1
            x = random.randrange(0, len(alphabet))
            y = random.randrange(1, len(alphabet))
            newperm = swap_perm(permutation, x, y)
            p1 = log_likelihood(ciphertext, newperm)

            a = accept_prob(p1, p2)

            # if random.random() < a: # happens with prob a
            if a > 0.51:
                permutation = newperm
                p2 = p1
                count = 0
            else:
                count += 1

            if p2 / len(ciphertext) < -2.7 and lol * len(ciphertext) > MAX_TRANS / 20:
                break
            if p2 / len(ciphertext) < -2.5 and lol * len(ciphertext) > MAX_TRANS / 10:
                break
            if lol * len(ciphertext) > MAX_TRANS / 5:
                break
            if count > 2000:
                break
            trans -= len(ciphertext)

        if DEBUG:
            print(lol)

        likelihood = log_likelihood(ciphertext, permutation)
        if DEBUG:
            print(permutation, likelihood / len(ciphertext))
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_permutation = permutation

    return best_permutation

def find_breakpoint_rough(ciphertext, permutation, leftside): # leftside = bool
    if leftside:
        plaintext = apply_perm(ciphertext, permutation)
        prob = [letter_probabilities[DECODER[plaintext[0]]]]
        for i in range(1, len(ciphertext)):
            prob.append(letter_transition_matrix[DECODER[plaintext[i]]][DECODER[plaintext[i-1]]])
            prob[-1] += prob[-2]
            
        for i in range(len(ciphertext) - 1, 9, -1):
            prob[i] -= prob[i-10]
            
        for i in range(10, len(ciphertext)):
            if prob[i] < -75:
                return i
    else:
        plaintext = apply_perm(ciphertext, permutation)
        prob = [letter_probabilities[DECODER[plaintext[0]]]]
        for i in range(1, len(ciphertext)):
            prob.append(letter_transition_matrix[DECODER[plaintext[i]]][DECODER[plaintext[i-1]]])
            prob[-1] += prob[-2]
            
        for i in range(len(ciphertext) - 1, 9, -1):
            prob[i] -= prob[i-10]

        for i in range(len(ciphertext) - 1, 9, -1):
            if prob[i] < -75:
                return i - 10

def find_breakpoint(ciphertext, permutation1, permutation2):
    plaintext1 = apply_perm(ciphertext, permutation1)
    prob1 = [letter_probabilities[DECODER[plaintext1[0]]]]
    for i in range(1, len(ciphertext)):
        prob1.append(letter_transition_matrix[DECODER[plaintext1[i]]][DECODER[plaintext1[i-1]]])
        prob1[-1] += prob1[-2]

    plaintext2 = apply_perm(ciphertext, permutation2)
    prob2 = [letter_probabilities[DECODER[plaintext2[0]]]]
    for i in range(1, len(ciphertext)):
        prob2.append(letter_transition_matrix[DECODER[plaintext2[i]]][DECODER[plaintext2[i-1]]])
        prob2[-1] += prob2[-2]
    prob = [prob1[i-1] + prob2[-1] - prob2[i] + letter_transition_matrix[DECODER[plaintext2[i]]][DECODER[plaintext1[i-1]]] for i in range(len(ciphertext))]
    return np.argmax(np.array(prob))

def decode(ciphertext, has_breakpoint, DEBUG=False):
    if has_breakpoint:
        c1 = ciphertext[:len(ciphertext) // 2]
        permutation1 = best_permutation(c1, MAX_TRANS / 4, DEBUG)
        p1 = log_likelihood(c1, permutation1) / len(c1)
        if DEBUG:
            print(permutation1)
            print(p1)
        c2 = ciphertext[len(ciphertext) // 2:]
        permutation2 = best_permutation(c2, MAX_TRANS / 4, DEBUG)
        p2 = log_likelihood(c2, permutation2) / len(c2)
        if DEBUG:
            print(permutation2)
            print(p2)
        if p1 > p2:
            breakpoint = find_breakpoint_rough(ciphertext, permutation1, True)
            permutation1 = best_permutation(c1, MAX_TRANS * 1.5, DEBUG)
            permutation2 = best_permutation(ciphertext[breakpoint:], MAX_TRANS / 2, DEBUG)
        else:
            breakpoint = find_breakpoint_rough(ciphertext, permutation2, False)
            permutation1 = best_permutation(ciphertext[:breakpoint], MAX_TRANS / 2, DEBUG)
            permutation2 = best_permutation(c2, MAX_TRANS * 1.5, DEBUG)
        if DEBUG:
            print("break ", breakpoint)

        breakpoint = find_breakpoint(ciphertext, permutation1, permutation2)
        if p1 > p2:
            permutation2 = best_permutation(ciphertext[breakpoint:], MAX_TRANS * 1.5, DEBUG)
        else:
            permutation1 = best_permutation(ciphertext[:breakpoint], MAX_TRANS * 1.5, DEBUG)

        if DEBUG:
            print("break ", breakpoint)
        return apply_perm(ciphertext[:breakpoint], permutation1) + apply_perm(ciphertext[breakpoint:], permutation2)
    else:
        permutation = best_permutation(ciphertext, MAX_TRANS * 2, DEBUG)
        return apply_perm(ciphertext, permutation)

def accuracy(decoded, plaintext):
    count = 0
    for x, y in zip(decoded, plaintext):
        if x == y:
            count += 1
    return count / len(plaintext)

def decode_plot(ciphertext, plaintext, has_breakpoint):
    # permutation = find_space_period(ciphertext)
    permutation = "".join(alphabet)
    count = {}
    likelihoods = []
    accepted = []
    accuracies = []
    for i in range(NUM_ITER):
        x = random.randrange(0, len(alphabet))
        y = random.randrange(1, len(alphabet))
        newperm = swap_perm(permutation, x, y)

        a = accept_prob(ciphertext, newperm, permutation)

        if random.random() < a: # happens with prob a
            permutation = newperm
            accepted.append(1)
        else:
            accepted.append(0)
        # if permutation not in count:
        #     count[permutation] = 0
        # count[permutation] += 1
        if i % (NUM_ITER / 10) == 0:
            print(permutation)
        likelihoods.append(log_likelihood(ciphertext, permutation))
        accuracies.append(accuracy(apply_perm(ciphertext, permutation), plaintext))

    return apply_perm(ciphertext, permutation), likelihoods, accepted, accuracies
