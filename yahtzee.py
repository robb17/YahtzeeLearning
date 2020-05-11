import random
import fileinput
import sys
import os
import csv
from copy import deepcopy

import rulebook
from roll import Roll

import multiprocessing as mp
from yahtzee_scoresheet import YahtzeeScoresheet


import label

import query_optimal_fast as optimal_strategy
import nn

''' yahtzee.py
        Program that determines how well a neural network can learn to
        play variant y of Yahtzee from a limited set of training examples
        generated for y.

        Run "python3/pypy3 yahtzee.py -help" for a list of available
        command-line arguments.
'''


N_PROCESSES = mp.cpu_count()
lookup_macro = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 7, 9: 8, 10: 8, 11: 9, 12: 6}

class OptimalPolicy:
    def __init__(self, dictionary):
        self.state_potentials = dictionary

    def choose_dice(self, sheet, roll, rolls_remaining):
        dice = optimal_strategy.queryMove(self.state_potentials, sheet.as_bitmask(), roll.as_list(), rolls_remaining)
        roll = Roll(dice)
        return roll
        
    def choose_category(self, sheet, roll):
        return optimal_strategy.queryMove(self.state_potentials, sheet.as_bitmask(), roll.as_list(), 0)

    def generate_training_data(self, n_trials, all_data):
        for x in range(0, n_trials):
            current_line = ""
            sheet, roll, n_rerolls = sample_gamespace(self.choose_dice, self.choose_category)
            opt_move = optimal_strategy.queryMove(self.state_potentials, sheet.as_bitmask(), roll.as_list(), n_rerolls)
            labeled_one_hot = None
            current_line = sheet.as_csv() + "," + roll.as_string() + "," + str(n_rerolls) + ","
            try:                # assume in this situation you choose a category to score in
                best_category = YahtzeeScoresheet.categories[opt_move]
                current_line += best_category + ","
                labeled_one_hot = label.build_one_hot(lookup_macro[opt_move], 10)
            except (ValueError, TypeError) as e:
                kept_dice = Roll(opt_move)
                current_line += "[" + kept_dice.as_string() + "],"
                labeled_one_hot = label.find_macro(kept_dice, roll, sheet, n_rerolls)
            current_line += labeled_one_hot + "\n"
            all_data.append(current_line)

def null_log(sheet, roll, rerolls, it, rand):
    pass


def stdout_log(sheet, roll, rerolls, it, rand):
    #print(sheet.as_state_string() + "," + "".join(str(x) for x in roll.as_list()) + "," + str(rerolls))
    pass
    #if (rand == it):
    #    outputsamples.write(sheet.as_state_string() + "," + "".join(str(x) for x in roll.as_list()) + "," + str(rerolls) + "\n")
    
def play_solitaire(choose_dice, choose_category, log=null_log):
    ''' Returns the score earned in one game played with the policy
        defined by the two given functions.  Each position is logged with
        the given function.

        choose_dice -- a function that takes a scoresheet, roll, and number of
                       rerolls, returns a subroll of the roll
        choose_category -- a function that takes a non-filled scoresheet and
                           a roll and returns the index of an unused category
                           on that scoresheet
        log -- a function that takes a scoresheet, roll, and number of rerolls
    '''
    # start with empty scoresheet
    
    sheet = YahtzeeScoresheet()
    while not sheet.game_over():
        # do the initial roll
        roll = Roll()
        roll.reroll()
        # reroll twice
        for i in [2, 1]:

            # choose dice to keep
            keep = choose_dice(sheet, roll, i)
            if not keep.subroll(roll):
                raise ValueError("dice to keep %s not a subset of roll %s" % (keep.as_list(), roll.as_list()))
            keep.reroll()
            roll = keep

        # choose category to use and mark it
        cat = choose_category(sheet, roll)
        sheet.mark(cat, roll)
    # print("=========GAME OVER==========")
        
    return sheet.grand_total()

def play_solitaire_test(choose_dice, choose_category, optimal_policy, log=null_log):
    ''' Returns the score earned in one game played with the policy
        defined by the two given functions.  Each position is logged with
        the given function.

        choose_dice -- a function that takes a scoresheet, roll, and number of
                       rerolls, returns a subroll of the roll
        choose_category -- a function that takes a non-filled scoresheet and
                           a roll and returns the index of an unused category
                           on that scoresheet
        log -- a function that takes a scoresheet, roll, and number of rerolls
    '''
    # start with empty scoresheet
    
    sheet = YahtzeeScoresheet()
    while not sheet.game_over():
        # do the initial roll
        roll = Roll()
        roll.reroll()
        # reroll twice
        for i in [2, 1]:

            # choose dice to keep
            keep = choose_dice(sheet, roll, i)
            if not keep.subroll(roll):
                raise ValueError("dice to keep %s not a subset of roll %s" % (keep.as_list(), roll.as_list()))
            keep.reroll()
            roll = keep

        # choose category to use and mark it
        cat = choose_category(sheet, roll)
        best_cat = optimal_policy.choose_category(sheet, roll)
        if not cat == best_cat:
            print(cat, best_cat, roll.as_list(), sheet.as_csv())
        sheet.mark(cat, roll)
    # print("=========GAME OVER==========")
        
    return sheet.grand_total()


def print_scoresheet(sheet):
    print("\n".join(str(id) + " " + name + " " + str(score) for (id, (name, score)) in zip(range(1, 18), sheet.as_list())))


class RandomPolicy:
    ''' A policy that picks a category at the beginning of each
        turn and tries to score in that category.

        This is not intended to be a good policy.  It averages about 191.
    '''
    def __init__(self):
        self.cat = None

        # what we hope to score in each category
        self.goals = [3, 6, 9, 12, 15, 18, 20, 10, 15, 20, 15, 25, 10]

    def choose_dice(self, sheet, roll, rerolls):
        # randomly choose an unsed category at the beginning of each turn
        if rerolls == 2:
            self.pick_random_category(sheet)

        # select dice according to which category we chose to try for
        # at the beginning of the turn
        if self.cat >= 0 and self.cat < YahtzeeScoresheet.THREE_KIND:
            return roll.select_all([self.cat + 1])
        elif self.cat in [YahtzeeScoresheet.THREE_KIND, YahtzeeScoresheet.FOUR_KIND, YahtzeeScoresheet.YAHTZEE]:
            return roll.select_for_n_kind(sheet, rerolls)
        elif self.cat == YahtzeeScoresheet.FULL_HOUSE:
            return roll.select_for_full_house()
        elif self.cat == YahtzeeScoresheet.CHANCE:
            return roll.select_for_chance(rerolls)
        else:
            return roll.select_for_straight(sheet)


    def choose_category(self, sheet, roll):
        ''' Returns the free category that minimizes regret.
        '''
        # for each category, compute the difference between what we
        # would score in that category and what we hoped to score
        regrets = [(cat, self.goals[cat] - sheet.score(cat, roll))  for cat in range(0, 13) if not sheet.is_marked(cat)]

        # greedily choose the category that minimizes that difference
        return min(regrets, key=lambda x:x[1])[0]
        

    def pick_random_category(self, sheet):
        ''' Randomly uniformly chooses an unsed category on the given
            scoresheet.

            sheet -- a YahtzeeScoresheet
        '''
        self.cat = None
        count = 0
        for c in range(13):
            if not sheet.is_marked(c):
                count += 1
                if random.random() < 1.0 / count:
                    self.cat = c
                    

def choose_dice_interactive(sheet, roll, rerolls):
    if rerolls == 2:
        print_scoresheet(sheet)
    print(roll)
    keep = None
    while keep is None:
        response = input("Select dice to keep (or 'all'):")
        if response == 'all':
            keep = roll
        else:
            try:
                keep = Roll.parse(response)
                if not keep.subroll(roll):
                    keep = None
                    print("select only dice in the current roll")
            except ValueError:
                keep = None
                print("select only dice in the current roll")
    return keep


def choose_category_interactive(sheet, roll):
    print(roll)
    cat = None
    while cat is None:
        try:
            cat = int(input("Choose category by index:")) - 1
            if sheet.is_marked(cat):
                print("select an unused category")
                cat = None
        except ValueError:
            cat = None
            print("select the index of an unused catagory")
    return cat
            

def evaluate_policy(n, choose_dice, choose_category):
    ''' Evaluates a policy by using it for the given number of games
        and returning its average score.

        n -- a positive integer
        choose_dice -- a function that takes a scoresheet, a roll, and a
                       number of rerolls and returns a subroll of the roll
        choose_category -- a function that takes a scoresheet and a roll
                           and returns the index of an unused category on
                           that scoresheet
    '''
    total = 0
    sys.stdout.write('Playing games')
    sys.stdout.flush()
    for i in range(n):
        if i % 100 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        total += play_solitaire(choose_dice, choose_category)
    return total / n

def evaluate_policy_test(n, choose_dice, choose_category, optimal_policy):
    ''' Evaluates a policy by using it for the given number of games
        and returning its average score.

        n -- a positive integer
        choose_dice -- a function that takes a scoresheet, a roll, and a
                       number of rerolls and returns a subroll of the roll
        choose_category -- a function that takes a scoresheet and a roll
                           and returns the index of an unused category on
                           that scoresheet
    '''
    total = 0
    sys.stdout.write('Playing games')
    sys.stdout.flush()
    for i in range(n):
        if i % 100 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        total += play_solitaire_test(choose_dice, choose_category, optimal_policy)
    return total / n

#def evaluate_policy_mp(n, choose_dice, choose_category, all_means):
#    ''' Evaluates a policy by using it for the given number of games
#        and returning its average score.
#
#        n -- a positive integer
#        choose_dice -- a function that takes a scoresheet, a roll, and a
#                       number of rerolls and returns a subroll of the roll
#        choose_category -- a function that takes a scoresheet and a roll
#                           and returns the index of an unused category on
#                           that scoresheet
#    '''
#    total = 0
#    for i in range(n):
#        print("starting")
#        total += play_solitaire(choose_dice, choose_category)
#        print(total / (i + 1))
#    all_means.append(total / n)

def generateNNTrainingData(policy, state_potentials, new_training_title, n_trials):
    labelout = open("training/" + new_training_title, "a")
    sys.stdout.write('Generating training data')
    sys.stdout.flush()
    for x in range(0, n_trials):
        sheet, roll, n_rerolls = sample_gamespace(policy.choose_dice, policy.choose_category)
        opt_move = optimal_strategy.queryMove(state_potentials, sheet.as_bitmask(), roll.as_list(), n_rerolls)
        labeled_one_hot = None
        labelout.write(sheet.as_csv() + "," + roll.as_string() + "," + str(n_rerolls) + ",")
        try:
            best_category = YahtzeeScoresheet.categories[opt_move]
            labelout.write(best_category + ",")
            labeled_one_hot = label.build_one_hot(opt_move, 13)
        except (ValueError, TypeError) as e:
            kept_dice = Roll(opt_move)
            labelout.write("[" + kept_dice.as_string() + "],")
            labeled_one_hot = label.find_macro(kept_dice, roll, sheet, n_rerolls)
        labelout.write(labeled_one_hot + "\n")
        if x % 1000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    labelout.close()

def sample_gamespace(choose_dice, choose_category):
    ''' Returns the score earned in one game played with the policy
        defined by the two given functions.  Each position is logged with
        the given function.

        choose_dice -- a function that takes a scoresheet, roll, and number of
                            rerolls, returns a subroll of the roll
        choose_category -- a function that takes a non-filled scoresheet and
                            a roll and returns the index of an unused category
                            on that scoresheet
        log -- a function that takes a scoresheet, roll, and number of rerolls
    '''
    # start with empty scoresheet

    sheet = YahtzeeScoresheet()
    rand = random.randint(0, 38)
    it = 0
    while not sheet.game_over():
        # do the initial roll
        roll = Roll()
        roll.reroll()

        # reroll twice
        for i in [2, 1]:

            if it == rand:
                return (sheet, roll, i)
            it += 1

            # choose dice to keep
            keep = choose_dice(sheet, roll, i)
            if not keep.subroll(roll):
                raise ValueError("dice to keep %s not a subset of roll %s" % (keep.as_list(), roll.as_list()))
            keep.reroll()
            roll = keep

        if it == rand:
            return (sheet, roll, 0)
        it += 1

        # choose category to use and mark it
        cat = choose_category(sheet, roll)
        sheet.mark(cat, roll)

    return (None, None, None)

def parseInput():
    ngames = 10000
    epochs = 500
    upperbonus = 35
    upperbonusthreshold = 63
    nGenerations = 3
    nTrains = 1
    nSteps = 32
    test_NN_size = False
    for x in range(1, len(sys.argv)):
        if sys.argv[x][0] == "-":
            if sys.argv[x][1:] == "ngames":
                ngames = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "epochs":
                epochs = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "setupperbonusthreshold":
                upperbonusthreshold = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "setupperbonus":
                upperbonus = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "nsteps":
                nSteps = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "nevaluations":
                nTrains = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "ngenerations":
                nGenerations = int(sys.argv[x + 1])
            elif sys.argv[x][1:] == "varyNNsize":
                test_NN_size = True
            elif sys.argv[x][1:] == "help":
                print("Usage: pypy3/python3 yahtzee.py\n [-ngames <# of games over which to gather training data>]\n [-epochs <# of training epochs>]\n [-setupperbonusthreshold <threshold>]\n [-setupperbonus <reward value>]\n [-nsteps <# of training example sets to visit and evaluate>]\n [-nevaluations <# of times to evluate NN>]\n [-ngenerations <# of times to generate and evaluate with a training set per step>]\n [-varyNNsize (forces comparison of differently sized NNs)]")
                exit(0)
    return (ngames, epochs, upperbonus, upperbonusthreshold, nSteps, nTrains, nGenerations, test_NN_size)

#def singleUse():
#    state_potentials = None
#    datain, dataout, ngames, epochs = parseInput()
#    potentials_dict = None
#    main_model, side_model = nn.train(datain, epochs)
#    policy = nn.NNPolicy(main_model, side_model)
#    if not dataout == None:
#        if potentials_dict == None:
#            print("Building dictionary of potentials from scratch")
#            state_potentials = optimal_strategy.buildDict()
#        else:
#            state_potentials = optimal_strategy.loadDict(potentials_dict)
#        generateTrainingData(policy, state_potentials, dataout, trial * 1000)
#    print(evaluate_policy(ngames, policy.choose_dice, policy.choose_category, stdout_log))

def varyTrainingData():
    ngames, epochs, upperbonus, upperbonusthreshold, nSteps, nTrains, nGenerations, test_NN_size = parseInput()
    y_axis = []
    state_potentials = None

    dict_name = "dict_UPT" + str(upperbonusthreshold) + "_UP" + str(upperbonus)
    optimal_strategy.initGame(upperbonusthreshold, upperbonus)
    YahtzeeScoresheet.UPPER_BONUS = upperbonus
    YahtzeeScoresheet.UPPER_BONUS_THRESHOLD = upperbonusthreshold

    dataout = "dataset_UPT" + str(upperbonusthreshold) + "_UP" + str(upperbonus)
    dataout += "_testNNsize" if test_NN_size else ""

    try:
        state_potentials = optimal_strategy.loadDict(dict_name)
        print("Value: " + str(optimal_strategy.queryValueOfModifiedGame(upperbonusthreshold, upperbonus)))
    except FileNotFoundError:
        print("Dictionary " + dict_name + " not found, generating now")
        state_potentials = optimal_strategy.buildDict()
        optimal_strategy.saveDict(dict_name, state_potentials)

    for trial in range(1, nSteps + 1):
        dataout = dataout.split('-')
        dataout = dataout[0]
        dataout += '-'
        dataout += str(trial * 2500) if not test_NN_size else str(trial)
        global_mean = 0
        for generation in range(0, nGenerations):
            filename = dataout + "_" + str(generation)
            try:
                f = open("training/" + filename, "r")
                f.close()
            except FileNotFoundError:
                n_training_examples = trial * 2500 if not test_NN_size else 55000
                print("Training set " + filename + " not found, generating now.")
                manager = mp.Manager()
                all_data = manager.list()
                jobs = []
                for procid in range(0, N_PROCESSES):
                    proc_dict = deepcopy(state_potentials)
                    proc_policy = OptimalPolicy(proc_dict)
                    proc = mp.Process(target = proc_policy.generate_training_data, args = (n_training_examples // N_PROCESSES, all_data))
                    jobs.append(proc)
                    proc.start()
                for proc in jobs:
                    proc.join()

                labelout = open("training/" + filename, "a")
                for line in all_data:
                    labelout.write(line)
                labelout.close()

            mean = 0
            size = 100 if not test_NN_size else trial * 5
            for training_session in range(0, nTrains):
                trial_model = nn.train(filename, size, epochs)
                trial_policy = nn.NNPolicy(trial_model)
                mean += evaluate_policy(ngames, trial_policy.choose_dice, trial_policy.choose_category)
            global_mean += mean / nTrains
        y_axis.append(global_mean / nGenerations)
        print(y_axis)

    dataoutresults = open("training/" + dataout.split("-")[0] + "_final", "a")
    dataoutresults.write(str(y_axis))
    dataoutresults.close()

def main():
    varyTrainingData()

if __name__ == "__main__":
    # testIt()
    main()

