import query_optimal_fast as optimal_strategy

class YahtzeeScoresheet:
    ''' A standard Yahtzee scoresheet.
    '''
    # category indices
    THREE_KIND = 6
    FOUR_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    CHANCE = 11
    YAHTZEE = 12

    UPPER_BONUS_THRESHOLD = 63
    UPPER_BONUS = 35

    categories = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "3K",
        "4K",
        "FH",
        "SS",
        "LS",
        "C",
        "Y"
    ]

    def __init__(self):
        self.categories = YahtzeeScoresheet.categories

        # the functions that determine scoring
        self.rules = [
            self.upper(1),
            self.upper(2),
            self.upper(3),
            self.upper(4),
            self.upper(5),
            self.upper(6),
            self.n_kind(3),
            self.n_kind(4),
            self.full_house(25),
            self.straight(4, 30),
            self.straight(5, 40),
            self.n_kind(1),
            self.yahtzee(50)
        ]

        # the functions that update subtotals and bonuses
        self.totals = [
            self.upper_total(),
            self.lower_total(),
            self.yahtzee_bonus()
        ]
        
        self._upper_total = 0
        self._upper_bonus = 0
        self._lower_total = 0
        self._yahtzee_bonus = 0
        self._turns = 0

        self.scores = [None] * len(self.categories)


    def upper(self, num):
        def points(roll):
            return num * roll.count(num)
        return points


    def n_kind(self, count):
        def points(roll):
            if roll.is_n_kind(count):
                return roll.total()
            else:
                return 0
        return points

    
    def straight(self, count, score):
        def points(roll):
            if roll.is_straight(count) or self.is_joker(roll):
                return score
            else:
                return 0
        return points


    def full_house(self, score):
        def points(roll):
            if roll.is_full_house() or self.is_joker(roll):
                return score
            else:
                return 0
        return points

        
    def is_joker(self, roll):
        return (roll.is_n_kind(5)
                and self.scores[YahtzeeScoresheet.YAHTZEE] is not None
                # and self.scores[YahtzeeScoresheet.YAHTZEE] > 0
                # and self.scores[roll.as_list()[0] - 1] is not None
                )
        

    def yahtzee(self, score):
        def points(roll):
            if roll.is_n_kind(5):
                return score
            else:
                return 0
        return points

        
    def upper_total(self):
        def update(cat, roll, score):
            if cat < 6:
                self._upper_total += score
                if self._upper_total >= YahtzeeScoresheet.UPPER_BONUS_THRESHOLD:
                    self._upper_bonus = YahtzeeScoresheet.UPPER_BONUS
        return update


    def lower_total(self):
        def update(cat, roll, score):
            if cat >= 6:
                self._lower_total += score
        return update


    def yahtzee_bonus(self):
        def update(cat, roll, score):
            if cat != YahtzeeScoresheet.YAHTZEE and roll.is_n_kind(5) and self.scores[YahtzeeScoresheet.YAHTZEE] is not None and self.scores[YahtzeeScoresheet.YAHTZEE] > 0:
                self._yahtzee_bonus += 100
        return update

    def is_marked(self, cat):
        ''' Determines if the given category is marked on this scorsheet.

            cat -- the index of a category on this scoresheet
        '''
        if cat < 0 or cat >= len(self.scores):
            raise ValueError("invalid category index: %d" % cat)
        return self.scores[cat] is not None


    def score(self, cat, roll):
        ''' Returns the score that would be earned on this scoresheet
            by scoring the given roll in the given category.

            cat -- the index of an unused category
            roll -- a complete Yahtzee roll
        '''
        if cat < 0 or cat > YahtzeeScoresheet.YAHTZEE:
            raise ValueError("invalid category index: %d" % cat)
        if self.scores[cat] is not None:
            raise ValueError("category already used: %d" % cat)
        if self.is_joker(roll):
            if self.scores[roll.as_list()[0] - 1] == None:
                if not cat == roll.as_list()[0] - 1:
                    raise ValueError("Forced joker rules not obeyed: category %d required" % int(roll.as_list()[0] - 1))
            else:
                lower_cats = []
                for i in range(6, 12):
                    if not self.is_marked(i):
                        lower_cats.append(i)
                if len(lower_cats) == 0:
                    if not cat < 6:
                        raise ValueError("Forced joker rules not obeyed: required to score in upper category")
                else:
                    if not cat in lower_cats:
                        raise ValueError("Forced joker rules not obeyed: required to score in lower category")
        return self.rules[cat](roll)
    

    def mark(self, cat, roll):
        ''' Updates this scoresheet by scoring the given roll
            in the given category.

            cat -- the index of an unused category
            roll -- a complete Yahtzee roll
        '''
        turn_score = self.score(cat, roll)
        self.scores[cat] = turn_score
        for tot in self.totals:
            tot(cat, roll, turn_score)
        self._turns += 1

    def valid_categories(self, roll):
        valid_categories = []
        if self.is_joker(roll):
            if self.scores[roll.as_list()[0] - 1] == None:
                valid_categories.append(roll.as_list()[0] - 1)
            else:
                lower_cats = []
                for i in range(6, 12):
                    if not self.is_marked(i):
                        lower_cats.append(i)
                if len(lower_cats) == 0:
                    valid_categories = self.standard_categories(roll)
                else:
                    valid_categories = lower_cats
        else:
            valid_categories = self.standard_categories(roll)
        return valid_categories

    def standard_categories(self, roll):
        valid_categories = []
        for i in range(0, 13):
            if not self.is_marked(i):  
                valid_categories.append(i)
        return valid_categories
            
    def grand_total(self):
        ''' Returns the total score, including bonus, marked on this scoresheet.
        '''
        return self._upper_total + self._upper_bonus + self._lower_total + self._yahtzee_bonus


    def game_over(self):
        ''' Determines if this scoresheet has all categories marked.
        '''
        return self._turns == len(self.scores)

    def as_list(self):
        result = list(zip(self.categories, self.scores))
        result.append(('UPPER TOTAL', self._upper_total))
        result.append(('UPPER BONUS', self._upper_bonus))
        result.append(('YAHTZEE BONUS', self._yahtzee_bonus))
        result.append(('GRAND TOTAL', self.grand_total()))
        return result


    def as_csv(self):
        ''' Returns a string representation of this scoresheet suitable
            as input to StrategyQuery.
        '''
        free = [self.categories[i] for i in range(0, 12) if self.is_marked(i)]
        if self.scores[YahtzeeScoresheet.YAHTZEE] is not None:
            free.append("Y+" if self.scores[YahtzeeScoresheet.YAHTZEE] > 0 else "Y")
        free.append("UP%d" % min(self._upper_total, YahtzeeScoresheet.UPPER_BONUS_THRESHOLD))
        return ",".join(free)

    def as_one_hot_csv(self, roll):
        working_one_hot = ""
        for x in range(0, 12):
            if self.is_marked(x):
                working_one_hot[x] = 1.0
        if self.scores[YahtzeeScoresheet.YAHTZEE] is not None:
            index = 13 if self.scores[YahtzeeScoresheet.YAHTZEE] > 0 else 12
            working_one_hot[index] = 1.0
        working_one_hot[20] = min(self._upper_total, YahtzeeScoresheet.UPPER_BONUS_THRESHOLD) / (YahtzeeScoresheet.UPPER_BONUS_THRESHOLD * 1.0)
        for y in range(1, 7):
            ncounts = roll.count(y)
            working_one_hot[y + 13] += ncounts / 5.0
        return working_one_hot

    def as_one_hot(self, roll, rolls_remaining):
        working_one_hot = [0.0 for x in range(0, 22)]
        for i in range(0, 12):
            if self.scores[i] is not None:
                working_one_hot[i] = 1.0
        if self.scores[12] is not None:
            if self.scores[12] > 0:
                working_one_hot[13] = 1.0
            else:
                working_one_hot[12] = 1.0
        working_one_hot[-2] = self._upper_total / (YahtzeeScoresheet.UPPER_BONUS_THRESHOLD * 1.0)
        dice = roll.as_list()
        for y in range(1, 7):
            working_one_hot[y + 13] += roll.count(y) / 5.0
        working_one_hot[-1] = rolls_remaining / 2.0
        return working_one_hot

    def as_bitmask(self):
        val = 0
        for i in range(0, 13):
            if self.is_marked(i):
                val += optimal_strategy.C_MASKS[i]
        if self.scores[YahtzeeScoresheet.YAHTZEE] is not None and self.scores[YahtzeeScoresheet.YAHTZEE] > 0:
            val += optimal_strategy.YSCORED
        val += min(self._upper_total, YahtzeeScoresheet.UPPER_BONUS_THRESHOLD)
        return val

    def __len__(self):
        length = 0
        for x in range(0, 13):
            if self.is_marked(x):
                length += 1
        return length

