import sys
import csv
import os
from copy import deepcopy

import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import normalize

import numpy as np

import play_yahtzee

enum_table = ['3K', '4K', 'FH', 'SS', 'LS', 'C']
expectation = [1.88, 5.28, 8.57, 12.16, 15.69, 19.19, 21.66, 13.10, 22.59, 29.46, 32.71, 16.87, 22.01]
expectation_sorted = [0, 1, 2, 3, 7, 4, 11, 5, 6, 12, 8, 9, 10]
regret_factors = [17.398936170212767, 6.195075757575758, 3.8168028004667445, 2.689967105263158, 2.0847673677501595, 1.7045336112558624, 1.510156971375808, 2.4969465648854965, 1.4479858344400178, 1.1103190767141888, 1.0, 1.938944872554831, 1.4861426624261698]


enum_all_macros = ['1s', '2s', '3s', '4s', '5s', '6s', 'nK', 'FH', 'straight', 'C', 'R']
enum_all_cats = ['1', '2', '3', '4', '5', '6', '3K', '4K', 'FH', 'SS', 'LS', 'C', 'Y']
macro_to_category = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6, 7, 12], 7: [8], 8: [9, 10], 9: [11]}
category_to_size = {6: 3, 7: 4, 12: 5, 9: 4, 10: 5}

class NN:
    def __init__(self, input_shape, output_shape, size):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = Sequential()
        self.optimizer = SGD(lr=0.23, decay=1e-6, momentum=0.82, nesterov=True)
        self.model.add(Dense(size // 2, activation='relu', input_dim = input_shape))
        self.model.add(Dropout(0.10))
        self.model.add(Dense(size, activation='relu'))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(output_shape, activation='softmax'))

    def __deepcopy__(self, log = None):
        new_nn = NN(self.input_shape, self.output_shape)
        new_nn.model = keras.models.clone_model(self.model)
        new_nn.compile()
        new_nn.model.set_weights(self.model.get_weights())
        return new_nn

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer)

    def train(self, inp, output, eps, batch = 1024):
        normalize(inp)
        self.model.fit(inp, output, epochs=eps, batch_size=batch)

class NNPolicy:
    def __init__(self, nn):
        self.nn = nn

    def __deepcopy__(self, log = None):
        new_main = deepcopy(self.nn)
        return NNPolicy(new_main)

    def choose_dice(self, sheet, roll, rolls_remaining):

        #gamestate = sheetToList(sheet, roll)
        #working_one_hot = parseGameState(gamestate, rolls_remaining)
        working_one_hot = sheet.as_one_hot(roll, rolls_remaining)

        nn_o = self.nn.model.predict(np.matrix(working_one_hot)).tolist()[0]
        #print(nn_o)

        while (1):
            max_index = argmax(nn_o)
            if max_index_corresponds_to_good_action(max_index, sheet, roll):
                break
            else:
                nn_o[max_index] = -1

        subroll = roll
        if max_index < 6:                       # 1s, 2s, 3s, etc.
            subroll = roll.select_all([max_index + 1])            # increment
        elif max_index == 6:                    # nkind
            subroll = roll.select_for_n_kind(sheet, rolls_remaining)
        elif max_index == 7:                    # full house
            subroll = roll.select_for_full_house()
        elif max_index == 8:                    # straight
            subroll = roll.select_for_straight(sheet)
        elif max_index == 9:                                   # chance
            subroll = roll.select_for_chance(rolls_remaining)
        else:                                   # reroll
            subroll = roll.select_all([1], 0)
        return subroll

    def choose_category(self, sheet, roll):

        working_one_hot = sheet.as_one_hot(roll, 0)

        nn_o = self.nn.model.predict(np.matrix(working_one_hot)).tolist()[0]
        macro_cat = None
        max_index = None
        cats_to_score = None
        cat = None

        while (1):
            max_index = argmax(nn_o)
            cats_to_score = max_index_corresponds_to_good_action(max_index, sheet, roll)
            if cats_to_score:
                break
            else:
                nn_o[max_index] = -1

        valid_categories = sheet.valid_categories(roll)
        macro_cat = max_index
        cats_to_score.sort(reverse = True)

        if macro_cat < 6:   # maybe put in yahtzee, 4k, 3k...
            cat = macro_cat
            if 12 in valid_categories and roll.is_n_kind(5):
                cat = 12
        elif macro_cat == 6:      # unpack to Y, 4K, 3K
            for x in cats_to_score:
                if roll.is_n_kind(category_to_size[x]):     # give priority to Y, then 4K
                    cat = x
                    break
            if cat == None:                                 # must assign a zero, assign to Y, then 4K
                cat = cats_to_score[0]
        elif macro_cat == 7:
            cat = 8
        elif macro_cat == 8:
            for x in cats_to_score:
                if roll.is_straight(category_to_size[x]):
                    cat = x
                    break
            if cat == None:                                 # must assign a zero, assign to LS, then SS
                cat = cats_to_score[0]
        elif macro_cat == 9:
            cat = 11

        if sheet.is_joker(roll):                            # score in category that returns largest score
            all_possible_scores = []
            for x in cats_to_score:
                all_possible_scores.append(sheet.score(x, roll))
            cat_index = argmax(all_possible_scores)
            cat = cats_to_score[cat_index]

        # if about to score a zero:
            # if it's an upper category and the last upper category available:
                # if it's reasonable to attain the upper bonus:
                    # score somewhere else
            # if there's a category that could get you to the upper bonus:
                # score in that category
            # if early in game, score in 4K. Else yahtzee
            # if it's late in game

        # if about to score in chance and roll is 3, 4 kind:
            # if total > 22:
                # score in 3, 4 kind

        return cat


def argmax(lst):
    if len(lst) == 0:
        return None
    max_val = lst[0]
    best_index = 0
    for x in range(0, len(lst)):
        if lst[x] > max_val:
            best_index = x
            max_val = lst[x]
    return best_index


def max_index_corresponds_to_good_action(index, sheet, roll):
    valid_categories = sheet.valid_categories(roll)
    candidates = []
    for category in macro_to_category[index]:
        if category in valid_categories:
            candidates.append(category)
    return False if len(candidates) == 0 else candidates

def sheetToList(sheet, roll):
    raw_state = sheet.as_list()
    gamestate = []
    for x in range(0, 6):
        if (raw_state[x][1] != None):
            gamestate.append(raw_state[x][0])

    for x in range(6, 12):
        new_str = []
        if (raw_state[x][1] != None):
            gamestate.append(raw_state[x][0])

    if raw_state[12][1] == 0:
        gamestate.append('Y')
    elif raw_state[12][1] == 50:
        gamestate.append('Y+')
    gamestate.append('UP' + str(raw_state[13][1]))
    roll_str = ""
    for die in roll.as_list():
        roll_str += str(die)

    gamestate.append(roll_str)

    return gamestate

#def testIt():
#   sheet = YahtzeeScoresheet()
#   roll = YahtzeeRoll()
#   roll.reroll()
#   policy = RandomPolicy()
#   while not sheet.game_over():
#       # do the initial roll
#       roll = YahtzeeRoll()
#       roll.reroll()
#       # reroll twice
#       for i in [2, 1]:
#           # choose dice to keep
#           keep = policy.choose_dice(sheet, roll, i)
#           if not keep.subroll(roll):
#               raise ValueError("dice to keep %s not a subset of roll %s" % (keep.as_list(), roll.as_list()))
#           keep.reroll()
#           roll = keep
#       # choose category to use and mark it
#       cat = policy.choose_category(sheet, roll)
#       sheet.mark(cat, roll)
#       print(sheet.as_list())
#       print(roll.as_list())
#       inputs = sheetToList(sheet, roll)
#       print(inputs)
#       print(parseGameState(inputs, 0))

# function for combining labeling input with labeling output
#def combineData():
#    if not len(sys.argv) == 3:
#        print("Usage: python3 nn.py inputfile outputfile")
#        exit(1)
#
#    inputfile = open(sys.argv[1], 'r')
#    outputfile = open(sys.argv[2], 'r')
#
#    input_line = inputfile.readline()
#    output_line = outputfile.readline()
#
#    while(1):
#        if (not input_line or not output_line):
#            break
#        sys.stdout.write(input_line[:-1])
#        sys.stdout.write(',')
#        sys.stdout.write(output_line[:-1])
#        sys.stdout.write('\n')
#        sys.stdout.flush()
#        input_line = inputfile.readline()
#        output_line = outputfile.readline()
#
#
#    inputfile.close()
#    outputfile.close()
#    os.system("sed 's/ /,/g' combinedout > combinedoutfinal")


def main():
    model = learn()
    policy = NNPolicy(model)

# inputs: [1s, 2s, 3s, 4s, 5s, 6s, 3K, 4K, FH, SS, LS, C, Y, Y+, roll1s, roll2s, roll3s, roll4s, roll5s, roll6s, UPTOT, rolls_remaining / 2]
#          [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,     15,     16,     17,     18,     19,     20,          21]

def parse_game_state(inputs, rolls_remaining):
    roll = inputs[-1]
    roll = list(roll)
    working_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    #parse inputs
    for x in range(1, 7):
        if str(x) in inputs:
            working_one_hot[x - 1] = 1.0

    for x in range(0, len(enum_table)):
        if enum_table[x] in inputs:
            working_one_hot[x + 6] = 1.0

    if 'Y' in inputs:
        working_one_hot[12] = 1.0
    elif 'Y+' in inputs:
        working_one_hot[13] = 1.0

    uptot = 0
    for x in range(0, len(inputs)):
        if inputs[x][0] == 'U' and inputs[x][1] == 'P':
            uptot = int(inputs[x][2:]) / 63.0
            working_one_hot[-2] = uptot

    #max_of_dice = 0
    #for z in range(1, 7):
    #    ncounts = roll.count(str(z))
    #    if ncounts > max_of_dice:
    #        max_of_dice = ncounts

    for y in range(1, 7):
        ncounts = roll.count(str(y))
        working_one_hot[y + 13] += ncounts / 5.0 #if max_of_dice == 0 else ncounts / float(max_of_dice)

    working_one_hot[-1] = rolls_remaining / 2.0
    return working_one_hot


def train(training_data_filename, size, eps = 200):
    x_all = [] #inputs
    y_all = [] #outputs

    # read the data
    reader = None
    with open("training/" + training_data_filename) as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')

        for row in reader:
            inputs = []
            y_all.append([int(x) for x in row[-10:]])
            inputs += [str(x) for x in row[:-12]]
            rolls_remaining = int(row[-12])

            working_one_hot = parse_game_state(inputs, rolls_remaining)

            x_all.append(working_one_hot)


    x_train = np.matrix(x_all)
    y_train = np.matrix(y_all)

    # set the topology of the neural network
    net = NN(x_train.shape[1], y_train.shape[1], size)
    net.compile()
    net.train(x_train, y_train, eps)
    return net

if __name__ == "__main__":
    pass
    # main()
    