from math import factorial
from itertools import combinations, chain, repeat, islice, count
from copy import deepcopy
import time
import sys
import multiprocessing as mp
from roll import Roll
import pickle

import rulebook

JOKER_BONUS = 100
NPIPS = 6
N_PROCESSES = mp.cpu_count()

''' query_optimal_fast
		Program that computes the optimal strategy for a given variant of Yahtzee

		Run "python3/pypy3 query_optimal_fast.py -help" for a list of available command-line arguments
'''

class memoize:
	def __init__(self, f):
		self.f = f
		self.memo = {}

	def __call__(self, *args):
		state_roll = makeItHashable(args[0], args[1])
		if state_roll not in self.memo:
			self.memo[state_roll] = self.f(args[0], args[1], args[2])
		return self.memo[state_roll]

	def clear(self):
		self.memo = {}

	def getLength(self):
		print(len(self.memo))

class memoizeRolls:
	def __init__(self, f):
		self.f = f
		self.memo = {}

	def __call__(self, *args):
		state_roll = makeItHashable(0, args[0])
		if state_roll not in self.memo:
			self.memo[state_roll] = self.f(args[0])
		return self.memo[state_roll]

def upper(n):
	def points(roll, state):
		if isJoker(roll, state):
			tot = (n * roll.count(n))
			if YSCORED & state:
				tot += JOKER_BONUS
			return tot
		else:
			return n * roll.count(n)
	return points

def nKind(n):
	def points(roll, state):
		tot = 0
		if roll.is_n_kind(n):
			tot = roll.total()
			if isJoker(roll, state) and (C_MASKS[roll.sample_dice() - 1] & state) and YSCORED & state:
				tot += JOKER_BONUS
		return tot
	return points

def fullHouse(score):
	def points(roll, state):
		tot = 0
		if isJoker(roll, state) and (C_MASKS[roll.sample_dice() - 1] & state):
			if YSCORED & state:
				tot += JOKER_BONUS
			tot += score
		if roll.is_full_house():
			tot += score
		return tot
	return points

def straight(n, score):
	def points(roll, state):
		tot = 0
		if isJoker(roll, state) and (C_MASKS[roll.sample_dice() - 1] & state):
			if YSCORED & state:
				tot += JOKER_BONUS
			tot += score
		if roll.is_straight(n):
			tot += score
		return tot
	return points

def yahtzee(score):
	def points(roll, state):
		if roll.is_n_kind(5):
			return score
		else:
			return 0
	return points

def isJoker(roll, state):
	return roll.is_n_kind(5) and (state & C_Y)

all_cats = ['1', '2', '3', '4', '5', '6', '3K', '4K', 'FH', 'SS', 'LS', 'C', 'Y']
#			   0	0	 0	  0	   0    0    0     0     0     0     0     0    0
#  42 bits, 13 for cats, 8 for upper total, 1 for flag to indicate that
#  yahtzee bonus is possible, 3 (x6) for each roll encoded in the state, and
#  2 for the number of rerolls
ALL_STATES_MASK =0b111111111111100000000000000000000000000000
YSCORED =        0b000000000000010000000000000000000000000000 	#  true if 50 scored in Y
N_ONES =         0b000000000000000010000000000000000000000000
N_TWOS =         0b000000000000000000010000000000000000000000
N_THREES =       0b000000000000000000000010000000000000000000
N_FOURS =        0b000000000000000000000000010000000000000000
N_FIVES =        0b000000000000000000000000000010000000000000
N_SIXES =        0b000000000000000000000000000000010000000000
N_REROLLS_MASK = 0b000000000000000000000000000000001100000000
N_REROLLS =      0b000000000000000000000000000000000100000000
UTOT =           0b000000000000000000000000000000000011111111
C_ONES =         0b100000000000000000000000000000000000000000
C_TWOS =         0b010000000000000000000000000000000000000000
C_THREES =       0b001000000000000000000000000000000000000000
C_FOURS =        0b000100000000000000000000000000000000000000
C_FIVES =        0b000010000000000000000000000000000000000000
C_SIXES =        0b000001000000000000000000000000000000000000
C_3K =           0b000000100000000000000000000000000000000000
C_4K =           0b000000010000000000000000000000000000000000
C_FH =           0b000000001000000000000000000000000000000000
C_SS =           0b000000000100000000000000000000000000000000
C_LS =           0b000000000010000000000000000000000000000000
C_C =            0b000000000001000000000000000000000000000000
C_Y =            0b000000000000100000000000000000000000000000

C_MASKS = [C_ONES, C_TWOS, C_THREES, C_FOURS, C_FIVES, C_SIXES, C_3K, C_4K, C_FH, C_SS, C_LS, C_C, C_Y]
CATS_TO_MASKS = {"1": C_ONES, "2": C_TWOS, "3": C_THREES, "4": C_FOURS, "5": C_FIVES, "6": C_SIXES, "3K": C_3K, "4K": C_4K, "FH": C_FH, "SS": C_SS, "LS": C_LS, "C": C_C, "Y": C_Y}


index_to_function = [
	upper(1),
	upper(2),
	upper(3),
	upper(4),
	upper(5),
	upper(6),
	nKind(3),
	nKind(4),
	fullHouse(25),
	straight(4, 30),
	straight(5, 40),
	nKind(1),
	yahtzee(50)
]

cat_to_index = {
	C_ONES: 0,
	C_TWOS: 1,
	C_THREES: 2,
	C_FOURS: 3,
	C_FIVES: 4,
	C_SIXES: 5,
	C_3K: 6,
	C_4K: 7,
	C_FH: 8,
	C_SS: 9,
	C_LS: 10,
	C_C: 11,
	C_Y: 12
}

n_unique_outcomes = {0: 1, 1: 6, 2: 21, 3: 56, 4: 126, 5: 252}
n_outcomes = {0: 1, 1: 6, 2: 36, 3: 216, 4: 1296, 5: 7776}

#  guaranteed that the category c is unused
#  returns the marginal score that one would attain by scoring the roll in
#  category c AND the marginal upper total increase.
def scoreRoll(state, dice, c):
	roll = Roll(dice)
	score = index_to_function[cat_to_index[c]](roll, state)
	marginal_upper_total = (score % 100) if c >= C_SIXES else 0		# if 1s, 2s, ... 6s, marginal upper total cannot include bonuses
	return score, marginal_upper_total

#		[number of dice to roll]! / ([number of duplicates]! * [number of duplicates]! * ...)
#  probability of generating all possible rolls was tested, p summed to 1
def probabilityOfBecomingRoll(dice, new_dice):
	prob_denominator = n_outcomes[len(new_dice)]

	fac_denominator = 1
	for x in range(1, NPIPS + 1):
		if x in new_dice:
			fac_denominator *= factorial(new_dice.count(x))
	fac_numerator = factorial(len(new_dice))
	prob_numerator = fac_numerator // fac_denominator
	return prob_numerator / prob_denominator

def makeItHashable(state, roll=None):
	new_state = state
	if roll != None:
		for d in roll:
			if d == 1:
				new_state += N_ONES
			elif d == 2:
				new_state += N_TWOS
			elif d == 3:
				new_state += N_THREES
			elif d == 4:
				new_state += N_FOURS
			elif d == 5:
				new_state += N_FIVES
			elif d == 6:
				new_state += N_SIXES
	return new_state

#  allRolls returns all possible combinations of new dice given that some dice are kept
@memoizeRolls
def allRolls(static_dice):
	local_time = time.time() * 10000.0
	ndice = 5 - len(static_dice)
	lst = []
	new_dice = []
	for n in range(1, 7):
		for x in range(0, ndice):
			lst.append(n)

	new_dice = list(set(combinations(lst, ndice)))
	new_dice_unique = []

	for element in new_dice:
		new_dice_unique.append(list(element))
	return new_dice_unique
EVERY_ROLL = allRolls([])

#  generateAllSubrolls gives all possible subrolls that you could make given some roll
@memoizeRolls
def generateAllSubrolls(roll):
	local_time = time.time() * 10000.0
	kept_dice_list = []
	for x in range(0, 6):
		for cset in combinations(roll, x):
			kept_dice_list.append(cset)
	kept_dice_list = list(set(kept_dice_list))
	return_list = []

	for subroll in kept_dice_list:
		return_list.append(list(subroll))
	return return_list

def getValidCategories(state, roll):
	local_time = time.time() * 10000.0
	cats_to_probe = []
	mroll = Roll(roll)
	if isJoker(mroll, state):
		if not (C_MASKS[mroll.sample_dice() - 1] & state):			#  if one of the Yahtzee-making dice is not a used category
			cats_to_probe.append(C_MASKS[mroll.sample_dice() - 1])	#  must use that upper category
		elif (not (C_FH & state) or not (C_SS & state) or not (C_LS & state) or not (C_3K & state) or not (C_4K & state) or not (C_C & state)):	#  if upper category filled and lower categories open...
			for cat in C_MASKS[6:12]:							#  add all of the available lower cats
				if not (cat & state):
					cats_to_probe.append(cat)
		else:
			for cat in C_MASKS[0:6]:							#  otherwise, score a zero in one of the upper cats
				if not (cat & state):
					cats_to_probe.append(cat)
	else:	# if not a joker, just add all available cats
		for cat in C_MASKS:
			if not (cat & state):
				cats_to_probe.append(cat)
	return cats_to_probe

#  apply changes to the state, roll not applied because the new state will
#  not have a roll associated with it
def getNewState(state, roll, c, marginal_upper_total):
	new_state = state
	new_state += marginal_upper_total
	new_state += c
	mroll = Roll(roll)
	if c == C_Y and mroll.is_n_kind(5) and not (new_state & YSCORED):	#  if scoring in Y and roll is Y, activate possibility for Yahtzee bonus
		new_state += YSCORED
	return new_state

@memoize
def endRollPotential(state, roll, state_potentials):
	max_potential = 0
	best_category = None
	best_next_state = None
	for c in getValidCategories(state, roll):
		marginal_score_increase, marginal_upper_total = scoreRoll(state, roll, c)
		if marginal_upper_total + (UTOT & state) >= rulebook.UPPER_BONUS_THRESHOLD:
			if (state & UTOT) < rulebook.UPPER_BONUS_THRESHOLD:			#  award bonus only if the threshold was just crossed
				marginal_score_increase += rulebook.UPPER_BONUS
			marginal_upper_total = rulebook.UPPER_BONUS_THRESHOLD - (UTOT & state)  	#  upper categories sum to 0, 1, ... 63
		new_state = getNewState(state, roll, c, marginal_upper_total)		#  get new state, assuming that we've scored in category c

		stored_potential = state_potentials[new_state + (N_REROLLS * 2)]	#  stored potential is that of the successor state
		potential_potential = marginal_score_increase + stored_potential
		if potential_potential >= max_potential:
			max_potential = potential_potential
			best_category = c
	return (max_potential, best_category)

@memoize
def afterSelectionPotential(state, kept_dice, state_potentials):
	potential = 0
	for new_dice in allRolls(kept_dice):
		reroll = new_dice + kept_dice
		reroll.sort()												#  necessary for efficient memoization
		new_state = state
		new_state -= N_REROLLS 										#  decrement number of rerolls remaining
		bsp, _ = beforeSelectionPotential(new_state, reroll, state_potentials)
		marginal_potential = probabilityOfBecomingRoll(kept_dice, new_dice) * bsp
		if marginal_potential > 0:
			potential += marginal_potential
	return potential

@memoize
def beforeSelectionPotential(state, roll, state_potentials):
	if not (state & N_REROLLS_MASK):								#  if no rerolls remaining
		return endRollPotential(state, roll, state_potentials)
	highest_potential = 0
	best_selection = None
	for kept_dice in generateAllSubrolls(roll):
		temp_potential = afterSelectionPotential(state, kept_dice, state_potentials)
		best_selection = kept_dice if temp_potential > highest_potential else best_selection
		highest_potential = max(temp_potential, highest_potential)
	return (highest_potential, best_selection)

def startTurnPotential(state, state_potentials):
	potential = 0
	current_dice = []
	for new_dice in EVERY_ROLL:
		bsp, _ = beforeSelectionPotential(state, new_dice, state_potentials)
		potential += probabilityOfBecomingRoll(current_dice, new_dice) * bsp
	return potential

def reachableState(state, uptot):
	s = set()
	if state & C_ONES:
		s.add(1)
	if state & C_TWOS:
		s.add(2)
	if state & C_THREES:
		s.add(3)
	if state & C_FOURS:
		s.add(4)
	if state & C_FIVES:
		s.add(5)
	if state & C_SIXES:
		s.add(6)
	if uptot == rulebook.UPPER_BONUS_THRESHOLD:
		for x in range(rulebook.UPPER_BONUS_THRESHOLD, rulebook.UPPER_BONUS_THRESHOLD + 6):		#  may need to increase the upper bound
			if reachableStateAux(s, x):
				return True
		return False
	return reachableStateAux(s, uptot)

def reachableStateAux(s, uptot):
	if uptot == 0:
		return True
	if len(s) == 0 and uptot > 0:
		return False
	for element in s:
		sprime = deepcopy(s)
		sprime.remove(element)
		for k in range(1, 6):
			if k * element <= uptot and reachableStateAux(sprime, uptot - (k * element)):
				return True
	return False

#  fill out end-game state potentials (characterized by potentials = 0)
def initStatePotentials():
	state_potentials = {}
	for x in range(0, (rulebook.UPPER_BONUS_THRESHOLD + 1)):
		state_potentials[ALL_STATES_MASK + x + (N_REROLLS * 2)] = 0
		state_potentials[ALL_STATES_MASK + x + YSCORED + (N_REROLLS * 2)] = 0
	return state_potentials

def processBuildDict(x, y, state, state_potentials, new_state_potentials):
	for z in range(x, y):
		if not reachableState(state, z):
			continue
		current_state = state + z
		new_state_potentials[current_state] = startTurnPotential(current_state, state_potentials)

#  determine the marginal value of each possible state
def buildDict():
	state_potentials = initStatePotentials()
	TIME = time.time() * 1.0
	completed = list(range(100))
	work_per_process = (rulebook.UPPER_BONUS_THRESHOLD + 1) // N_PROCESSES

	for alpha in range(0, 2):
		yscored_local = YSCORED if alpha == 0 else 0	# need to compute all YSCORED states first because non-YSCORED states cannot be accessed from YSCORED states
		for x in range(1, len(C_MASKS)):
			states_set = combinations(C_MASKS, len(C_MASKS) - x)
			time1 = time.time() * 1.0
			for state_lst in states_set:
				beforeSelectionPotential.clear()		# these memoized values no longer useful because state is different
				afterSelectionPotential.clear()
				endRollPotential.clear()
				state = (N_REROLLS * 2)	+ yscored_local	# all states to be looked up are "pre-loaded" with rerolls
				for mask in state_lst:					# construct the bitwise rep of state
					state += mask
				manager = mp.Manager()
				new_state_potentials = manager.dict()
				jobs = []
				overflow = (rulebook.UPPER_BONUS_THRESHOLD + 1) % N_PROCESSES
				extra_work = 0
				beginning_index = 0
				end_index = work_per_process
				for z in range(0, N_PROCESSES):			# build the dict with processes = N_PROCESSES
					if overflow > 0: end_index += 1
					overflow -= 1
					proc = mp.Process(target = processBuildDict, args = (beginning_index, end_index, state, state_potentials, new_state_potentials))
					jobs.append(proc)					# work broken up in same state in terms of UPTOT
					proc.start()
					beginning_index = end_index
					end_index += work_per_process
				for proc in jobs:
					proc.join()
				state_potentials.update(new_state_potentials)
				if len(state_potentials) / 715262 >= completed[0] / 20:
					print(str(int((completed[0] / 20) * 100)) + " percent complete")
					completed.remove(completed[0])
			print("combos of length " + str(len(C_MASKS) - x) + " completed in " + str((time.time() * 1.0 - time1) / 60) + " minutes")

	print("Finished in " + str((time.time() * 1.0 - TIME) / 60) + " minutes")

	return state_potentials

def loadDict(dict_to_load):
	state_potentials = None
	with open(dict_to_load, "rb") as pickle_file:
		state_potentials = pickle.load(pickle_file)
	print("Done loading dictionary " + dict_to_load)
	pickle_file.close()
	return state_potentials

#  allows user to obtain state potentials from the dictionary
def queryValues(state_potentials):
	raw_cats = ""
	while True:
		state = 0
		raw_cats = input("Please enter gamestate in the following format: [used categories],[roll],[n rerolls],[upper total] (Q to exit): ")
		if raw_cats == "Q" or raw_cats == 'q':
			break
		all_data = raw_cats.split(",")
		uptot = int(all_data[3].strip())
		rerolls = int(all_data[2].strip())
		roll_str = all_data[1].strip()
		roll = []
		for x in range(0, 5):
			roll.append(int(roll_str[x]))
		all_cats = all_data[0].split(" ")
		for cat in all_cats:
			if cat == "Y+":
				state += YSCORED
				state += C_Y
				break
			if cat not in CATS_TO_MASKS:
				print("Invalid gamestate.")
				break
			state += CATS_TO_MASKS[cat]
		state += uptot
		if not reachableState(state, int(uptot)):
			print("Unreachable state.")
			continue
		#print("Future expectation value associated with state: " + str(startTurnPotential(state, state_potentials)))
		print("Best move: " + str(queryMove(state_potentials, state, roll, rerolls)))


def queryValueOfModifiedGame(upperbonusthreshold, upperbonus):
	initGame(upperbonusthreshold, upperbonus)
	state_potentials = loadDict("dict_UPT" + str(upperbonusthreshold) + "_UP" + str(upperbonus))
	return startTurnPotential((N_REROLLS * 2), state_potentials)

def initGame(upperbonusthreshold, upperbonus):
	beforeSelectionPotential.clear()
	afterSelectionPotential.clear()
	endRollPotential.clear()
	rulebook.UPPER_BONUS_THRESHOLD = upperbonusthreshold
	rulebook.UPPER_BONUS = upperbonus

def queryMove(state_potentials, state, roll, rerolls):
	best_choice = None
	#state = 0
	#uptot = 0
	#gamestate_lst = gamestate.split(",")
	#for bit in gamestate_lst[:-1]:
	#	if bit == "Y+":
	#		state += CATS_TO_MASKS["Y"]
	#		state += YSCORED
	#		continue
	#	state += CATS_TO_MASKS[bit]
	#uptot = int(gamestate_lst[-1][2:])
	#state += uptot
	uptot = state & UTOT
	if not reachableState(state, uptot):
		print("Unreachable state")
		return None
	if rerolls == 0:
		value, best_choice = endRollPotential(state, roll, state_potentials)
		best_choice = C_MASKS.index(best_choice)
	else:
		state += N_REROLLS * rerolls
		_, best_choice = beforeSelectionPotential(state, roll, state_potentials)
	return best_choice

def saveDict(output_dict, state_potentials):
	f = open(output_dict, "wb")
	pickle.dump(state_potentials, f)
	f.close()
	print("Dict saved as " + str(output_dict))

def main():
	output_dict = None
	dict_to_load = None
	state_potentials = None

	for x in range(1, len(sys.argv)):
		if sys.argv[x][0] == "-":
			if sys.argv[x][1:] == "processes":
				N_PROCESSES = int(sys.argv[x + 1])
			elif sys.argv[x][1:] == "dump":
				output_dict = sys.argv[x + 1]
			elif sys.argv[x][1:] == "load":
				dict_to_load = sys.argv[x + 1]
			elif sys.argv[x][1:] == "setupperbonus":
				rulebook.UPPER_BONUS = int(sys.argv[x + 1])
			elif sys.argv[x][1:] == "setupperbonusthreshold":
				rulebook.UPPER_BONUS_THRESHOLD = int(sys.argv[x + 1])
			elif sys.argv[x][1:] == "help":
				print("Usage: pypy3/python3 query_optimal_fast.py\n [-processes <# of cores to utilize>]\n [-dump <file for EV dictionary pickling>]\n [-load <file for EV dictionary de-pickling>]\n [-setupperbonusthreshold <threshold>]\n [-setupperbonus <reward value>]")
				exit()
	if not dict_to_load == None and not output_dict == None:
		print("Unknown operation specified")
		exit()

	if dict_to_load == None:
		state_potentials = buildDict()
		print("Expectation value associated with game: " + str(startTurnPotential((N_REROLLS * 2), state_potentials)))
		print("   Exp: 254.59")
		print("Expectation value associated with rolling a Yahtzee on first turn: " + str(50 + startTurnPotential((N_REROLLS * 2) + YSCORED + C_Y, state_potentials)))
		print("   Exp: 320.84")
		print("Expectation value associated with rolling 2 3 4 4 6 on first turn: " + str(19 + startTurnPotential((N_REROLLS * 2) + C_C, state_potentials)))
		print("   Exp: 238.96")
		print(len(state_potentials))
	else:
		state_potentials = loadDict(dict_to_load)


	if output_dict != None:
		saveDict(output_dict, state_potentials)

	queryValues(state_potentials)

	#print(queryMove(state_potentials, ["1", "3", "5", "C", "UP31", "33444", "2"]))

#  standard boilerplate
if __name__ == "__main__":
    main()
    #testIt()
