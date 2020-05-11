import multiset as mset
import random

class Roll:
	def __init__(self, roll = []):
		'''	Initializes a roll to the given list
		'''
		self.dice = mset.Multiset(roll)

	def count(self, n):
		''' Count all instances of n
		'''
		return self.dice[n]

	def reroll(self):
		''' Adds uniformly random dice to the roll until the roll contains 5 dice
		'''
		while len(self.dice) < 5:
			self.dice[random.randint(1, 6)] += 1

	def subroll(self, other):
		''' Determines if the present roll is a subset of another
		'''
		for x in range(1, 7):
			if other.dice[x] < self.dice[x]:
				return False
		return True

	def as_list(self):
		''' Returns the present roll as a list
		'''
		roll = []
		for die in self.dice.items():
			for x in range(0, die[1]):
				roll.append(die[0])
		return roll

	def sample_dice(self):
		''' Returns one of the dice from the present roll
		'''
		for d in self.dice.items():
			return d[0]

	def total(self):
		''' Gives the total of all pips showing
		'''
		total = 0
		for x in range(1, 7):
			total += self.dice[x] * x
		return total

	def is_n_kind(self, n):
		for x in range(1, 7):
			if self.count(x) >= n:
				return True 
		return False

	def is_full_house(self):
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

	def is_straight(self, n):
		streakiness = 0
		for x in range(1, 7):
			if self.count(x) > 0:
				streakiness += 1
				if streakiness >= n:
					break
			else:
				streakiness = 0
		return streakiness == n

	def no_pair(self):
		for x in range(1, 7):
			if self.count(i) > 1:
				return False
		return True

	@staticmethod
	def parse(string):
		''' Returns a Roll with dice corresponding to the digits of the
			given string
		'''
		roll = Roll([])
		for char in string:
			if not char.isdigit():
				raise ValueError("given non-digit")
			num = int(char)
			roll.dice[num] += 1
		return roll

	def as_string(self):
		''' Returns the Roll as a string without any delimiters
		'''
		return "".join(map(str, self.as_list()))

	def select_all(self, nums, m = 5):
		''' Returns a Roll containing dice corresponding to the given nums--but
			only retaining a maximum of m dice for each given num
		'''
		roll = Roll([])
		for n in nums:
			if n < 1 or n > 6:
				raise ValueError("attempted to select for invalid number")
			if self.dice[n] > 0:
				roll.dice[n] = min(m, self.dice[n])
		return roll

	def select_one(self, nums):
		''' Returns a Roll containing only one occurrence of each of the given
			nums—-provided that said number is in the present roll
		'''
		roll = Roll([])
		for n in nums:
			if n < 1 or n > 6:
				raise ValueError("attempted to select for invalid number")
			if self.dice[n] > 0:
				roll.dice[n] = 1
		return roll

	def select_for_chance(self, rerolls):
		''' Returns the subroll of the present roll that maximizes the expected
			score in 		
			rerolls -- 1 or 2
		'''
		if rerolls == 2:
			return self.select_all([5, 6])
		else:
			return self.select_all([4, 5, 6])

	def select_for_full_house(self):
		''' Returns the subroll of the present roll that maximizes the chance
			of achieving a full house
		'''
		best_subroll = []
		for x in range(1, 7):
			if self.dice[x] > 1:
				best_subroll.append(x)
		return self.select_all(best_subroll, 3)

	def select_for_straight(self, sheet):
		from yahtzee_scoresheet import YahtzeeScoresheet
		if not sheet.is_marked(YahtzeeScoresheet.SMALL_STRAIGHT):
			runs = self.longest_runs()
			if len(runs[0]) >= 3:
				return self.select_one(runs[0])
			else:
				counts = [sum([(0 if sheet.is_marked(n - 1) else 1) for n in x]) for x in runs]
				run = runs[0]
				if len(runs) > 1 and (not sheet.is_marked(YahtzeeScoresheet.CHANCE) or counts[1] > counts[0]):
					run = runs[1]
				return self.select_one(run)

		else:
			low = self.select_one(range(1, 6))
			high = self.select_one(range(2, 7))
			if len(low.as_list()) > len(high.as_list()):
				return low
			else:
				return high

	def longest_runs(self):
		runs = []
		longest = 0
		curr_len = 0
		for i in range(1, 7):
			if self.count(i) > 0:
				curr_len += 1
				if curr_len == longest:
					runs.append(list(range(i - curr_len + 1, i + 1)))
				elif curr_len > longest:
					runs = [list(range(i - curr_len + 1, i + 1))]
					longest = curr_len
			else:
				curr_len = 0
		return runs

	def select_for_n_kind(self, sheet, rerolls):
		from yahtzee_scoresheet import YahtzeeScoresheet
		max_keep = 5
		if not sheet.is_marked(YahtzeeScoresheet.FOUR_KIND) and sheet.is_marked(YahtzeeScoresheet.YAHTZEE) and sheet.scores[YahtzeeScoresheet.YAHTZEE] == 0:
			max_keep = 4
		elif not sheet.is_marked(YahtzeeScoresheet.THREE_KIND) and sheet.is_marked(YahtzeeScoresheet.FOUR_KIND) and sheet.is_marked(YahtzeeScoresheet.YAHTZEE) and sheet.scores[YahtzeeScoresheet.YAHTZEE] == 0:
			max_keep = 3
		high_freq = 0
		most_freq = None
		for i in range(1, 7):
			if self.count(i) >= high_freq:
				high_freq = self.count(i)
				most_freq = i
		
		keep_nums = [most_freq]
		
		# keep 4's, 5's, and 6's if already have what we need
		# (4's only if down to last reroll)
		if ((max_keep == 3 and sheet.score(YahtzeeScoresheet.THREE_KIND, self) > 0)
			or (max_keep == 4 and sheet.score(YahtzeeScoresheet.FOUR_KIND, self) > 0)):
			for i in range(3 + rerolls, 7):
				if i != most_freq:
					keep_nums.append(i)
				else:
					max_keep = 5
				
		return self.select_all(keep_nums, max_keep)


