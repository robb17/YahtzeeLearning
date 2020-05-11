from yahtzee_scoresheet import YahtzeeScoresheet

import random

from roll import Roll

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