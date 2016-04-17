def scoreStraightFlush(hand):
    # detects if a sorted poker hand is a straight flush and returns the score
    if all(c[0] == hand[0][0] for c in hand):
        if all(hand[i][1] == hand[i+1][1]+1 for i in range(0,4)):
            return 80000000000 + 100000000*hand[0][1]
        elif hand[0][1] == 14 and hand[1][1] == 5 and hand[2][1] == 4 and hand[3][1] == 3 and hand[4][1] == 2:
            return 80500000000
        else:
            return 0
    else:
        return 0

def scoreFourKind(hand):
    # detects if a sorted poker hand is a four of a kind and returns the score
    if all(c[1] == hand[0][1] for c in hand[0:4]):
        return 70000000000 + 100000000*hand[0][1] + 1000000*hand[4][1]
    elif all(c[1] == hand[1][1] for c in hand[1:5]):
        return 70000000000 + 100000000*hand[1][1] + 1000000*hand[0][1]
    else:
        return 0

def scoreFullHouse(hand):
    # detects if a sorted poker hand is a full house and returns the score
    if hand[0][1] == hand[1][1] and hand[1][1] == hand[2][1] and hand[3][1] == hand[4][1]:
        return 60000000000 + 100000000*hand[0][1] + 1000000*hand[3][1]
    elif hand[0][1] == hand[1][1] and hand[2][1] == hand[3][1] and hand[3][1] == hand[4][1]:
        return 60000000000 + 100000000*hand[2][1] + 1000000*hand[0][1]
    else:
        return 0

def scoreFlush(hand):
    # detects if a sorted poker hand is a flush and returns the score
    if all(c[0] == hand[0][0] for c in hand):
        return 50000000000 + 100000000*hand[0][1] + 1000000*hand[1][1] + 10000*hand[2][1] + 100*hand[3][1] + hand[4][1]
    else:
        return 0

def scoreStraight(hand):
    # detects if a sorted poker hand is a straight and returns the score
    if all(hand[i][1] == hand[i+1][1]+1 for i in range(0,4)):
        return 40000000000 + 100000000*hand[0][1]
    elif hand[0][1] == 14 and hand[1][1] == 5 and hand[2][1] == 4 and hand[3][1] == 3 and hand[4][1] == 2:
        return 40500000000
    else:
        return 0

def scoreThreeKind(hand):
    # detects if a sorted poker hand is a three of a kind and returns the score
    if all(c[1] == hand[0][1] for c in hand[0:3]):
        return 30000000000 + 100000000*hand[0][1] + 1000000*hand[3][1] + 10000*hand[4][1]
    elif all(c[1] == hand[1][1] for c in hand[1:4]):
        return 30000000000 + 100000000*hand[1][1] + 1000000*hand[0][1] + 10000*hand[4][1]
    elif all(c[1] == hand[2][1] for c in hand[2:5]):
        return 30000000000 + 100000000*hand[2][1] + 1000000*hand[0][1] + 10000*hand[1][1]
    else:
        return 0

def scoreTwoPair(hand):
    # detects if a sorted poker hand is a two pair and returns the score
    if hand[0][1] == hand[1][1] and hand[2][1] == hand[3][1]:
        return 20000000000 + 100000000*hand[0][1] + 1000000*hand[2][1]  + 10000*hand[4][1]
    elif hand[0][1] == hand[1][1] and hand[3][1] == hand[4][1]:
        return 20000000000 + 100000000*hand[0][1]  + 1000000*hand[3][1]  + 10000*hand[2][1]
    elif hand[1][1] == hand[2][1] and hand[3][1] == hand[4][1]:
        return 20000000000 + 100000000*hand[1][1]  + 1000000*hand[3][1]  + 10000*hand[0][1]
    else:
        return 0

def scoreOnePair(hand):
    # detects if a sorted poker hand is a one pair and returns the score
    score = 0
    for i in range(0, 4):
        if hand[i][1] == hand[i+1][1]:
            score = 10000000000 + 100000000*hand[i][1]
            multiplier = 1000000
            for j in range(0,4):
                if j != i and j != i+1:
                    score += hand[j][1] * multiplier
                    multiplier = int(multiplier/100)
            break
    return score

def scoreHighCord(hand):
    # returns the score of a high card sorted poker hand
    return 100000000*hand[0][1] + 1000000*hand[1][1] + 10000*hand[2][1] + 100*hand[3][1] + hand[4][1]

def scoreHand(hand):
    # returns the score of a poker hand
    hand.sort(key=lambda x: x[1], reverse=True)
    score = scoreStraightFlush(hand)
    if score == 0:
        score = scoreFourKind(hand)
    if score == 0:
        score = scoreFullHouse(hand)
    if score == 0:
        score = scoreFlush(hand)
    if score == 0:
        score = scoreStraight(hand)
    if score == 0:
        score = scoreThreeKind(hand)
    if score == 0:
        score = scoreTwoPair(hand)
    if score == 0:
        score = scoreOnePair(hand)
    if score == 0:
        score = scoreHighCord(hand)
    return score