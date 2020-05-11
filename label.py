import random

lookup_macro = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 7, 9: 8, 10: 8, 11: 9, 12: 6}

def build_one_hot(index, length):
    one_hot = ""
    for i in range(0, length):
        if i == index:
            one_hot += "1,"
        else:
            one_hot += "0,"
    return one_hot[:-1]

def n_different_dice(roll1, roll2):
    same_pips_showing = 0
    all_pips_showing = 0
    for pip in range(1, 7):
        same_pips_showing += min(roll1.count(pip), roll2.count(pip))
        all_pips_showing += max(roll1.count(pip), roll2.count(pip))
    return all_pips_showing - same_pips_showing

def find_macro(dice_kept, roll, sheet, rerolls):
    ''' Determines which macro-category the act of keeping dice_kept most
        closely corresponds to. That is, subrolls that represent good plays
        for each category are compared against dice_kept, and that which has
        the least number of different dice is determined to be what was "gone
        for."
    '''
    best_index = -1
    min_value = 10
    indices_overlap = [10 for x in range(0, 10)]

    nUsedCats = 0
    unUsedIndex = 0
    for x in range(0, 13):
        if sheet.is_marked(x):
            nUsedCats += 1
        if not sheet.is_marked(x):
            unUsedIndex = x

    if nUsedCats == 12:                                     # if only one cat remaining, go for it
        return build_one_hot(lookup_macro[unUsedIndex], 10)

    valid_categories = sheet.valid_categories(roll)

    for x in range(0, 6):
        would_be_subroll = roll.select_all([x + 1])
        if x in valid_categories:
            indices_overlap[x] = n_different_dice(dice_kept, would_be_subroll)

    if 11 in valid_categories:
        would_be_subroll = roll.select_for_chance(rerolls)
        indices_overlap[9] = n_different_dice(dice_kept, would_be_subroll)

    if 8 in valid_categories:
        would_be_subroll = roll.select_for_full_house()
        indices_overlap[7] = n_different_dice(dice_kept, would_be_subroll)

    if 6 in valid_categories or 7 in valid_categories or 12 in valid_categories:
        would_be_subroll = roll.select_for_n_kind(sheet, rerolls)
        indices_overlap[6] = n_different_dice(dice_kept, would_be_subroll)

    if 9 in valid_categories or 10 in valid_categories:
        would_be_subroll = roll.select_for_straight(sheet)
        indices_overlap[8] = n_different_dice(dice_kept, would_be_subroll)

    ties = []
    for x in range(0, len(indices_overlap)):
        if indices_overlap[x] < min_value:
            min_value = indices_overlap[x]
            best_index = x
            ties.clear()
        if indices_overlap[x] == min_value:
            ties.append(x)

    index_priority = [0, 1, 2, 3, 4, 5]

    random.shuffle(index_priority)

    index_priority += [8, 6, 7, 9]

    for element in index_priority:
        if element in ties:
            best_index = element
            break                       # currently prioritizing upper cats > straights > nK > FH > C

    #if 6 in valid_categories and roll.is_n_kind(3) and roll.total() > 20 and len(sheet) < 9 and not roll.is_n_kind(4) and not roll.is_straight(4) and not roll.is_full_house():
    #    best_index = 6

    return build_one_hot(best_index, 10)