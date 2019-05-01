from math import factorial
from itertools import combinations, chain, repeat, islice, count
from copy import deepcopy
import time
import sys
import multiprocessing as mp
import multiset as mset
import pickle

JOKER_BONUS = 100
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS = 35
NPIPS = 6
N_PROCESSES = 16 # Vary according to number of threads available on your CPU

class Roll:

	roll_representations = [
		0b000000000000000010000000000000000000000000,
		0b000000000000000000010000000000000000000000,
		0b000000000000000000000010000000000000000000,
		0b000000000000000000000000010000000000000000,
		0b000000000000000000000000000010000000000000,
		0b000000000000000000000000000000010000000000
	]

	def __init__(self, roll):
		''' Uses the Multiset library
		'''
		self.dice = mset.Multiset(roll)

	def count(self, n):
		''' Count all dice with n pips showing
		'''
		return self.dice[n]

	def asList(self):
		''' Returns the roll as a list
		'''
		roll = []
		for die in self.dice.items():
			roll.append(die[0])
		return roll

	def sampleD(self):
		''' Sample the dice
		'''
		for d in self.dice.items():
			return d[0]

	def total(self):
		''' Count all the pips shown across all dice
		'''
		total = 0
		for x in range(1, 7):
			total += self.dice[x] * x
		return total

	def isNKind(self, n):
		''' Determine if the roll is n of a kind
		'''
		for x in range(1, 7):
			if self.count(x) >= n:
				return True 
		return False

	def isFullHouse(self):
		''' Determine if the roll is a full house
		'''
		pair = False
		trips = False
		for x in range(1, 7):
			if (pair and trips) or self.count(x) in [1, 4, 5]:
				break
			if self.count(x) == 2:
				pair = True
			elif self.count(x) == 3:
				trips = True
		return pair and trips

	def isStraight(self, n):
		''' Determine if the roll is a small or large straight
		'''
		streakiness = 0
		for x in range(1, 7):
			if self.count(x) > 0:
				streakiness += 1
				if streakiness >= n:
					break
			else:
				streakiness = 0
		return streakiness == n

	def hash(self):
		''' Hash the roll using the binary representations above
		'''
		val = 0
		for die in self.dice.items():
			val += self.roll_representations[die[0] - 1] * die[1]
		return val

class State:
	categories = {
		"1": 0,
		"2": 1,
		"3": 2,
		"4": 3,
		"5": 4,
		"6": 5,
		"3K": 6,
		"4K": 7,
		"FH": 8,
		"SS": 9,
		"LS": 10,
		"C": 11,
		"Y": 12
	}

	''' Representing categories the bit-masking way
	'''

	cat_representations = [
		0b100000000000000000000000000000000000000000,
		0b010000000000000000000000000000000000000000,
		0b001000000000000000000000000000000000000000,
		0b000100000000000000000000000000000000000000,
		0b000010000000000000000000000000000000000000,
		0b000001000000000000000000000000000000000000,
		0b000000100000000000000000000000000000000000,
		0b000000010000000000000000000000000000000000,
		0b000000001000000000000000000000000000000000,
		0b000000000100000000000000000000000000000000,
		0b000000000010000000000000000000000000000000,
		0b000000000001000000000000000000000000000000,
		0b000000000000100000000000000000000000000000,
	]

	REROLLS_MASK = 	   0b000000000000000000000000000000001100000000
	REROLLS =          0b000000000000000000000000000000000100000000
	UPPER_TOTAL =      0b000000000000000000000000000000000011111111
	UPPER_CATEGORIES = 0b111111000000000000000000000000000000000000
	LOWER_CATEGORIES = 0b000000111111100000000000000000000000000000
	BONUS_POSSIBLE =   0b000000000000010000000000000000000000000000

	def __init__(self):
		self.val = 0

	def mark(self, dice, cat, marginal_upper_total):
		''' Add the appropriate categorical representation to the value
			representing state
		'''
		self.val += marginal_upper_total
		self.val += cat
		roll = Roll(dice)
		if cat == self.cat_representations[12]
				and roll.isNKind(5)
				and not (self.val & self.BONUS_POSSIBLE):	# activate possibility for bonus
			self.val += self.BONUS_POSSIBLE

	def isMarked(self, cat):
		'''	Returns whether or not a given category is marked
			Supports both literal and numerical representation of categories
		'''
		try:
			marked = self.categories[cat]
			return self.cat_representations[marked] & self.val
		except KeyError:
			return self.cat_representations[cat] & self.val

	def hash(self, dice):
		''' Hash the state and roll, for memoization purposes
		'''
		roll = Roll(dice)
		return self.val + roll.hash()

	def isBonusActive(self):
		''' Returns if it's possible to score a secondary Yahtzee bonus
		'''
		return self.val & self.BONUS_POSSIBLE

	def rerollsRemaining(self):
		''' Returns how many rerolls remain
		'''
		return self.val & self.REROLLS_MASK

	def upperTotal(self):
		''' Returns the upper total associated with the position
		'''
		return self.val & self.UPPER_TOTAL

	def areLowerCatsMarked(self):
		''' Returns if all the lower categories are marked
		'''
		return (self.LOWER_CATEGORIES & self.val) == self.LOWER_CATEGORIES

class memoize:
	def __init__(self, f):
		self.f = f
		self.memo = {}

	def __call__(self, *args):
		''' Remember the results of this call if it's not already
			stored in the memo
		'''
		state_roll = args[0].hash(args[1])
		if state_roll not in self.memo:
			self.memo[state_roll] = self.f(args[0], args[1], args[2])
		return self.memo[state_roll]

	def clear(self):
		''' Necessary for the large amount of calls—-otherwise, might exceed
			the maximum size of a Pythonic dictionary
		'''
		self.memo = {}

	def getLength(self):
		''' For debugging purposes
		'''
		print(len(self.memo))

class memoizeRolls:
	def __init__(self, f):
		self.f = f
		self.memo = {}

	def __call__(self, *args):
		''' Remember the results of this call if it's not already
			stored in the memo
		'''
		roll = Roll(args[0])
		roll_hash = roll.hash()
		if roll_hash not in self.memo:
			self.memo[roll_hash] = self.f(args[0])
		return self.memo[roll_hash]

def upper(n):
	def points(roll, state):
		if isJoker(roll, state):
			tot = (n * roll.count(n))
			if state.isBonusActive():
				tot += JOKER_BONUS
			return tot
		else:
			return n * roll.count(n)
	return points

def nKind(n):
	def points(roll, state):
		tot = 0
		if roll.isNKind(n):
			tot = roll.total()
			if isJoker(roll, state) and state.isMarked(roll.sampleD() - 1) and state.isBonusActive():
				tot += JOKER_BONUS
		return tot
	return points

def fullHouse(score):
	def points(roll, state):
		tot = 0
		if isJoker(roll, state) and state.isMarked(roll.sampleD() - 1):
			if state.isBonusActive():
				tot += JOKER_BONUS
			tot += score
		if roll.isFullHouse():
			tot += score
		return tot
	return points

def straight(n, score):
	def points(roll, state):
		tot = 0
		if isJoker(roll, state) and state.isMarked(roll.sampleD() - 1):
			if state.isBonusActive():
				tot += JOKER_BONUS
			tot += score
		if roll.isStraight(n):
			tot += score
		return tot
	return points

def yahtzee(score):
	def points(roll, state):
		if roll.isNKind(5):
			return score
		else:
			return 0
	return points

def isJoker(roll, state):
	return roll.isNKind(5) and state.isMarked(12)


cat_to_function = {
	State.cat_representations[0]: upper(1),
	State.cat_representations[1]: upper(2),
	State.cat_representations[2]: upper(3),
	State.cat_representations[3]: upper(4),
	State.cat_representations[4]: upper(5),
	State.cat_representations[5]: upper(6),
	State.cat_representations[6]: nKind(3),
	State.cat_representations[7]: nKind(4),
	State.cat_representations[8]: fullHouse(25),
	State.cat_representations[9]: straight(4, 30),
	State.cat_representations[10]: straight(5, 40),
	State.cat_representations[11]: nKind(1),
	State.cat_representations[12]: yahtzee(50)
}

''' Some pre-computation for speed purposes
'''
n_outcomes = {0: 1, 1: 6, 2: 36, 3: 216, 4: 1296, 5: 7776}

def probabilityOfBecomingRoll(dice, new_dice):
''' Returns the probility that some dice will become new_dice
	[number of dice to roll]! / ([number of duplicates]! * [number of
		duplicates]! * ...)
'''
	prob_denominator = n_outcomes[len(new_dice)]

	fac_denominator = 1
	for x in range(1, NPIPS + 1):
		if x in new_dice:
			fac_denominator *= factorial(new_dice.count(x))
	fac_numerator = factorial(len(new_dice))
	prob_numerator = fac_numerator // fac_denominator
	return prob_numerator / prob_denominator

@memoizeRolls
def allRolls(static_dice):
''' Returns all possible combinations of new dice, given that some dice were kept
'''
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

@memoizeRolls
def generateAllSubrolls(roll):
''' Returns all possible subrolls that could be made for the given roll
'''
	kept_dice_list = []
	for x in range(0, 6):
		for cset in combinations(roll, x):
			kept_dice_list.append(cset)
	kept_dice_list = list(set(kept_dice_list))
	return_list = []

	for subroll in kept_dice_list:
		return_list.append(list(subroll))

	return return_list

def scoreRoll(state, dice, c):
''' Returns the marginal score that one would attain by scoring the roll in
	category c AND the marginal upper total increase. It's assumed that c
	is a valid category.
'''
	roll = Roll(dice)
	score = cat_to_function[c](roll, state)
	marginal_upper_total = (score % 100) if c >= state.cat_representations[5] else 0		# if 1s, 2s, ... 6s, marginal upper total cannot include bonuses
	return score, marginal_upper_total

def getValidCategories(state, roll):
''' Returns a list of categories in which a score may be logged.
	Determined in accordance with forced Joker rules.
'''
	cats_to_probe = []
	mroll = Roll(roll)
	if isJoker(mroll, state):
		if not (state.isMarked(mroll.sampleD() - 1)):	#  if one of the Yahtzee-making dice is not a used category
			cats_to_probe.append(state.cat_representations[mroll.sampleD() - 1])	#  must use that upper category
		elif not state.areLowerCatsMarked():			#  if upper category filled and lower categories are not totally filled...
			for cat in range(6, 12):					#  add all of the available lower cats
				if not state.isMarked(cat):
					cats_to_probe.append(state.cat_representations[cat])
		else:
			for cat in range(0, 6):					#  otherwise, score a zero in one of the upper cats
				if not state.isMarked(cat):
					cats_to_probe.append(state.cat_representations[cat])
	else:	# if not a joker, just add all available cats
		for cat in range(0, 13):
			if not state.isMarked(cat):
				cats_to_probe.append(state.cat_representations[cat])
	return cats_to_probe

@memoize
def endRollPotential(state, roll, state_potentials):
''' Finds the optimal category in which the given roll should be scored.
	Returns the marginal score value and the category.
'''
	max_potential = 0
	best_category = None
	for c in getValidCategories(state, roll):
		marginal_score_increase, marginal_upper_total = scoreRoll(state, roll, c)
		if marginal_upper_total + state.upperTotal() >= UPPER_BONUS_THRESHOLD:
			if state.upperTotal() < UPPER_BONUS_THRESHOLD:				#  award bonus only if the threshold was just crossed
				marginal_score_increase += UPPER_BONUS
			marginal_upper_total = UPPER_BONUS_THRESHOLD - state.upperTotal()  	#  upper categories sum to 0, 1, ... 63
		new_state = deepcopy(state)
		new_state.mark(roll, c, marginal_upper_total)			#  get new state, assuming that we've scored in category c

		stored_potential = state_potentials[new_state.val + (new_state.REROLLS * 2)]	#  stored potential is that of the successor state
		potential_potential = marginal_score_increase + stored_potential
		if potential_potential >= max_potential:
			max_potential = potential_potential
			best_category = c
	return (max_potential, state.cat_representations.index(best_category))

@memoize
def afterSelectionPotential(state, kept_dice, state_potentials):
''' Returns the sum of the potentials of all possible immediately resultant
	game states
'''
	potential = 0
	for new_dice in allRolls(kept_dice):
		reroll = new_dice + kept_dice
		new_state = deepcopy(state)
		new_state.val -= state.REROLLS 										#  decrement number of rerolls remaining
		bsp, _ = beforeSelectionPotential(new_state, reroll, state_potentials)
		marginal_potential = probabilityOfBecomingRoll(kept_dice, new_dice) * bsp
		if marginal_potential > 0:
			potential += marginal_potential
	return potential

@memoize
def beforeSelectionPotential(state, roll, state_potentials):
''' Finds the best subset of rolled dice to keep. Returns the potential
	associated with retaining these dice and the dice themselves.
'''
	if state.rerollsRemaining() == 0:
		return endRollPotential(state, roll, state_potentials)
	highest_potential = 0
	best_selection = None
	for kept_dice in generateAllSubrolls(roll):
		temp_potential = afterSelectionPotential(state, kept_dice, state_potentials)
		best_selection = kept_dice if temp_potential > highest_potential else best_selection
		highest_potential = max(temp_potential, highest_potential)
	return (highest_potential, best_selection)

def startTurnPotential(state, state_potentials):
''' Returns the potential associated with some state.
'''
	potential = 0
	current_dice = []
	for new_dice in EVERY_ROLL:
		bsp, _ = beforeSelectionPotential(state, new_dice, state_potentials)
		potential += probabilityOfBecomingRoll(current_dice, new_dice) * bsp
	return potential

def reachableState(state, uptot):
''' Determine if the given state is reachable.
'''
	s = set()
	for x in range(0, 6):
		if state.isMarked(x):
			s.add(x + 1)
	if uptot == UPPER_BONUS_THRESHOLD:
		for x in range(UPPER_BONUS_THRESHOLD, UPPER_BONUS_THRESHOLD + 4):		#  may need to increase the upper bound
			if reachableStateAux(s, x):
				return True
		return False
	return reachableStateAux(s, uptot)

def reachableStateAux(s, uptot):
''' Actually do the heavy lifting for reachableState()
'''
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

def initStatePotentials():
''' Fills out end-game state potentials (characterized by values = 0) so that
	DP may proceed
'''
	state_potentials = {}
	state = State()
	state.val += state.REROLLS * 2
	for x in range(0, UPPER_BONUS_THRESHOLD + 1):
		state_potentials[state.val + state.UPPER_CATEGORIES + state.LOWER_CATEGORIES] = 0
		state_potentials[state.val + state.BONUS_POSSIBLE + state.UPPER_CATEGORIES + state.LOWER_CATEGORIES] = 0
		state.val += 1
	return state_potentials

def processBuildDict(x, y, state, state_potentials, new_state_potentials):
''' Build parts of the dictionary in parallel: those for which all values
	other than upper_total are static. The potentials associated with these
	states are not codependent in any way.
'''
	state.val += x
	for z in range(x, y):
		if not reachableState(state, z):
			continue
		new_state_potentials[state.val] = startTurnPotential(state, state_potentials)
		state.val += 1

def buildDict():
''' Determine the marginal value of each possible state by working backwards.
	Insert first into the dictionary state potentials in later game positions
	that would be required for the computation of earlier game positions.
'''
	TIME = time.time() * 1.0
	completed = list(range(100))
	work_per_process = (UPPER_BONUS_THRESHOLD + 1) // N_PROCESSES
	state = State()

	state_potentials = initStatePotentials()
	for alpha in range(0, 2):
		yscored_local = state.BONUS_POSSIBLE if alpha == 0 else 0	# need to compute all YSCORED states first because non-YSCORED states cannot be accessed from YSCORED states
		for x in range(1, 13):
			states_set = combinations(state.cat_representations, len(state.cat_representations) - x)
			time1 = time.time() * 1.0
			for state_lst in states_set:
				beforeSelectionPotential.clear()		# these memoized values no longer useful because state is different
				afterSelectionPotential.clear()
				endRollPotential.clear()
				state.val = (state.REROLLS * 2) + yscored_local	# all states to be looked up are "pre-loaded" with rerolls
				for mask in state_lst:					# construct the bitwise rep of state
					state.val += mask
				manager = mp.Manager()
				new_state_potentials = manager.dict()
				jobs = []
				for z in range(0, N_PROCESSES):			# build the dict with processes = N_PROCESSES
					overflow = 0
					if z == N_PROCESSES - 1:
						overflow = (UPPER_BONUS_THRESHOLD + 1) % N_PROCESSES
					proc_state = deepcopy(state)
					proc = mp.Process(target = processBuildDict, args = (z * work_per_process, z * work_per_process + work_per_process + overflow, proc_state, state_potentials, new_state_potentials))
					jobs.append(proc)					# work broken up in same state in terms of UPTOT
					proc.start()
				for proc in jobs:
					proc.join()
				state_potentials.update(new_state_potentials)
				if len(state_potentials) / 715262 >= completed[0] / 20:
					print(str(int((completed[0] / 20) * 100)) + " percent complete")
					completed.remove(completed[0])
			print("combos of length " + str(len(state.cat_representations) - x) + " completed in " + str((time.time() * 1.0 - time1) / 60) + " minutes")
			break
	print("Finished in " + str((time.time() * 1.0 - TIME) / 60) + " minutes")

	return state_potentials

def loadDict(dict_to_load):
	''' Load a pickled dictionary with name dict_to_load
	'''
	state_potentials = None
	with open(dict_to_load, "rb") as pickle_file:
		state_potentials = pickle.load(pickle_file)
	print("Done loading dictionary " + dict_to_load)
	pickle_file.close()
	return state_potentials

def queryValues(state_potentials):
''' Interface that allows user to obtain state potentials from the dictionary
'''
	raw_cats = ""
	while True:
		state = State()
		raw_cats = input("Please enter gamestate (Q to exit): ")
		if raw_cats == "Q" or raw_cats == 'q':
			break
		cats_and_uptot = raw_cats.split(",")
		uptot = cats_and_uptot[1].strip()
		all_cats = cats_and_uptot[0].split(" ")
		for cat in all_cats:
			if cat == "Y+":
				state.val += state.cat_representations[12]
				state.val += state.BONUS_POSSIBLE
				continue
			if cat not in state.categories:
				print("Invalid gamestate.")
				break
			state.val += state.cat_representations[state.categories[cat]]
		state.val += int(uptot)
		state.val += state.REROLLS * 2
		if not reachableState(state, int(cats_and_uptot[-1])):
			print("Unreachable state.")
			continue
		print(bin(state.val))
		print("Future expectation value associated with state: " + str(startTurnPotential(state, state_potentials)))

def queryMove(state_potentials, gamestate, roll, rerolls):
''' Parses comma-separated gamestates and returns the optimal move
'''
	best_choice = None
	state = State()
	uptot = 0
	gamestate_lst = gamestate.split(",")
	for bit in gamestate_lst[:-1]:
		if bit in state.categories:
			state.val += state.cat_representations[state.categories[bit]]
		elif bit == "Y+":
			state.val += state.BONUS_POSSIBLE
			state.val += state.cat_representations[12]
	uptot = int(gamestate_lst[-1][2:])
	state.val += uptot
	if not reachableState(state, uptot):
		print("Unreachable state")
		return None
	if rerolls == 0:
		_, best_choice = endRollPotential(state, roll, state_potentials)
	else:
		state.val += state.REROLLS * rerolls
		_, best_choice = beforeSelectionPotential(state, roll, state_potentials)
	return best_choice

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
	if not dict_to_load == None and not output_dict == None:
		print("Unknown operation specified")
		exit()

	if dict_to_load == None:
		state_potentials = buildDict()
		state = State()
		state.val += state.REROLLS * 2
		print("Expectation value associated with game: " + str(startTurnPotential(state, state_potentials)))
		print("   Exp: 254.59")
		state.val = state.val + state.BONUS_POSSIBLE + state.cat_representations[12]
		print("Expectation value associated with rolling a Yahtzee on first turn: " + str(50 + startTurnPotential(state, state_potentials)))
		print("   Exp: 320.84")
		state = State()
		state.val += state.REROLLS * 2
		state.val += state.cat_representations[11]
		print("Expectation value associated with rolling 2 3 4 4 6 on first turn: " + str(19 + startTurnPotential(state, state_potentials)))
		print("   Exp: 238.96")
		print(len(state_potentials))
	else:
		state_potentials = loadDict(dict_to_load)

	if output_dict != None:
		f = open(output_dict, "wb")
		pickle.dump(state_potentials, f)
		f.close()
		print("Dict saved as " + str(output_dict))

	queryValues(state_potentials)


#  standard boilerplate
if __name__ == "__main__":
    main()
    #testIt()
