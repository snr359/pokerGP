import random
import statistics
import time
import copy
import itertools
from multiprocessing import Pool as ThreadPool
from AINodes import *
from scorePokerHand import *


memorySize = 256

boolToInt = {True: 1, False: 0}

numberPlayersPerGame = 6
handLimit = 10
initialMoney = 100
ante = 5

initialDeck = [(0, 2) ,(0, 3) ,(0, 4) ,(0, 5) ,(0, 6) ,(0, 7) ,(0, 8) ,(0, 9) ,(0, 10) ,(0, 11) ,(0, 12) ,(0, 13) ,(0, 14) ,(1, 2) ,(1, 3) ,(1, 4) ,(1, 5) ,(1, 6) ,(1, 7) ,(1, 8) ,(1, 9) ,(1, 10) ,(1, 11) ,(1, 12) ,(1, 13) ,(1, 14) ,(2, 2) ,(2, 3) ,(2, 4) ,(2, 5) ,(2, 6) ,(2, 7) ,(2, 8) ,(2, 9) ,(2, 10) ,(2, 11) ,(2, 12) ,(2, 13) ,(2, 14) ,(3, 2) ,(3, 3) ,(3, 4) ,(3, 5) ,(3, 6) ,(3, 7) ,(3, 8) ,(3, 9) ,(3, 10) ,(3, 11) ,(3, 12) ,(3, 13) ,(3, 14)]

populationSize = 300
numberGenerations = 200
numberChildrenPerGeneration = 300
numberEvaluationsPerMember = 30
parsimonyPressure = .001
maxAncestorsUsed = 50
KTournamentK = 10

mutationChance = .02
mutationTreeDepth = 5
mutationTerminateChance = .1

class AI:
    """
    The decision-making AI that is evolved
    """
    def __init__(self):
        self.baseNode = None
        self.fitnessRatings = []
        self.fitness = 0
        self.memory = [0]*memorySize
        self.parents = []
    def clearFitness(self):
        self.fitnessRatings = []
        self.fitness = 0
    def clearMemory(self):
        self.memory = [0]*memorySize
    def getDecision(self, environment):
        return self.baseNode.get(self.memory, environment)
    def replaceNode(self, oldNode, newNode):
        if oldNode is self.baseNode:
            self.baseNode = newNode
        else:
            oldNodeParent = oldNode.parent
            oldNodeParentBranchIndex = oldNode.parentBranchIndex
            oldNodeParent.branches[oldNodeParentBranchIndex] = newNode
            newNode.parent = oldNodeParent
            newNode.parentBranchIndex = oldNodeParentBranchIndex

def getAllNodes(tree):
    """
    Returns a list of all the nodes in a tree
    """
    results = list()
    results.append(tree)
    if len(tree.branches) > 0:
        for b in tree.branches:
           results.extend(getAllNodes(b))
    return results

def generateRandomDecisionTree(returnType, depthLimit, terminateChance):
    ##nonTerminals = [ifNodeNumber, ifNodeAction, equalNode, greaterThanNode, andNode, orNode, notNode, addNode, subNode, multiplyNode, divideNode, writeMemoryNode, readMemoryNode, readEnvironmentNode, buildActionNode]
    ##terminals = [constantNumberNode, randomNumberNode, constantBooleanNode, randomBooleanNode, constantActionNode, randomActionNode]
    allNodeTypes =  [ifNodeNumber, ifNodeAction, equalNode, greaterThanNode, andNode, orNode, notNode, addNode, subNode, multiplyNode, divideNode, moduloNode, writeMemoryNode, readMemoryNode, readEnvironmentNode, buildActionNode, constantNumberNode, constantBooleanNode, constantActionNode]

    newNodeType = random.choice(allNodeTypes)
    newNode = newNodeType()
    if depthLimit == 0 or random.random() < terminateChance:
        while newNode.returnType != returnType or newNode.isTerminal == False:
            newNodeType = random.choice(allNodeTypes)
            newNode = newNodeType()
    else:
        while newNode.returnType != returnType or newNode.isTerminal == True:
            newNodeType = random.choice(allNodeTypes)
            newNode = newNodeType()
    newNode = newNodeType()
    for branchIndex in range(0, len(newNode.branches)):
        newNode.branches[branchIndex] = generateRandomDecisionTree(newNode.branchTypes[branchIndex], depthLimit - 1, terminateChance)
        newNode.branches[branchIndex].parent = newNode
        newNode.branches[branchIndex].parentBranchIndex = branchIndex
    return newNode

def makeGroups(population, groupSize, minRepeats):
    """
    Produces a number of groups of individuals selected from the population, with each member appearing in a minimum number of groups
    """
    groups = []
    timesUsed = [0]*len(population)
    while any(timesUsed[i] < minRepeats for i in range(0, len(population))):
        indexes = random.sample(range(len(population)), groupSize)
        newGroup = []
        for i in indexes:
            newGroup.append(population[i])
            timesUsed[i] += 1
        groups.append(newGroup)
    return groups

def getBets(players, hands, publicCards, playersMoney, pot, playerStillInHand):
    """
    Collect the bets at a particular point in the game
    """
    bets = [0] * numberPlayersPerGame
    playerBettingIndex = 0
    lastPlayerToCollectBetsFromIndex = (playerBettingIndex - 1) % numberPlayersPerGame
    collectingBets = True
    playerStillInHandInt = [0]*numberPlayersPerGame
    while collectingBets:
        if playerStillInHand[playerBettingIndex] and playerStillInHand.count(True) > 1:

            for i in range(0, numberPlayersPerGame):
                playerStillInHandInt[i] = boolToInt[playerStillInHand[i]]

            environment = [hands[playerBettingIndex], len(publicCards), publicCards, bets, playersMoney, playerStillInHandInt, playerBettingIndex, playersMoney[playerBettingIndex], bets[playerBettingIndex], max(bets) - bets[playerBettingIndex], pot]
            thisPlayersDecision = players[playerBettingIndex].getDecision(environment)
            thisPlayersDecision = (thisPlayersDecision[0] % 3, thisPlayersDecision[1])
            amountToBet = 0

            # Get player's decision
            if thisPlayersDecision[0] == 0:
                amountToBet = max(bets) - bets[playerBettingIndex]
            elif thisPlayersDecision[0] == 1:
                amountToBet = abs(thisPlayersDecision[1] + max(bets) - bets[playerBettingIndex])
                # Cap bet to amount of money player has left
                amountToBet = min(amountToBet, playersMoney[playerBettingIndex])
                if amountToBet > 0:
                    lastPlayerToCollectBetsFromIndex = (playerBettingIndex - 1) % numberPlayersPerGame
            elif thisPlayersDecision[0] == 2:
                playerStillInHand[playerBettingIndex] = False

            if playerStillInHand[playerBettingIndex]:
                playersMoney[playerBettingIndex] -= amountToBet
                bets[playerBettingIndex] += amountToBet

                # If the player has not met the minimum bet to stay in the game, they fold automatically
                if bets[playerBettingIndex] < max(bets):
                    playerStillInHand[playerBettingIndex] = False

        # Stop collecting bets if we have reached the last player we need a bet from
        if playerBettingIndex == lastPlayerToCollectBetsFromIndex:
            collectingBets = False

        else:
            playerBettingIndex += 1
            playerBettingIndex %= numberPlayersPerGame

    return bets


def evalFitness(players):
    """
    Evaluates the fitness of the given AI players and records it in their fitness evaluations list by playing Texas Hold Em
    """
    # Initialize variables and deck
    playerStillInGame = [True]*numberPlayersPerGame
    playersMoney = [initialMoney] * numberPlayersPerGame
    for p in players:
        p.clearMemory()

    # Play poker hands until either the maximum number of hands has been played or one player has won all the money
    handsPlayed = 0
    stillPlaying = True
    while handsPlayed < handLimit and stillPlaying:
        # Initialize hand
        pot = 0
        deck = initialDeck[:]
        random.shuffle(deck)
        hands = []
        for i in range(0, numberPlayersPerGame):
            hands.append([])
        publicCards = []
        playerStillInHand = playerStillInGame[:]

        # Collect antes
        for i in range(0, numberPlayersPerGame):
            if playersMoney[i] < ante:
                pot += playersMoney[i]
                playersMoney[i] = 0
                playerStillInHand[i] = False
                playerStillInGame[i] = False
            else:
                playersMoney[i] -= ante
                pot += ante

        # Deal initial cards and get all players' bets
        for i in range(0,2):
            for h in hands:
                h.append(deck.pop())

        # Collect all initial bets
        bets = getBets(players, hands, publicCards, playersMoney, pot, playerStillInHand)

        # Add collected bets to the pot
        pot += sum(bets)

        # Draw the first three public cards
        for i in range(0, 3):
            publicCards.append(deck.pop())

        # Collect the next round of bets
        bets = getBets(players, hands, publicCards, playersMoney, pot, playerStillInHand)

        # Add collected bets to the pot
        pot += sum(bets)

        # Draw the fourth public card
        publicCards.append(deck.pop())

        # Collect the next round of bets
        bets = getBets(players, hands, publicCards, playersMoney, pot, playerStillInHand)

        # Add collected bets to the pot
        pot += sum(bets)

        # Draw the fifth public card
        publicCards.append(deck.pop())

        # Collect the last round of bets
        bets = getBets(players, hands, publicCards, playersMoney, pot, playerStillInHand)

        # Add collected bets to the pot
        pot += sum(bets)

        # Get the scores for everyone's hands by scoring every combination of possible 5-card hands for each player
        handScores = [0] * numberPlayersPerGame
        for i in range(0, numberPlayersPerGame):
            if playerStillInHand[i]:
                handCombos = list(itertools.combinations(hands[i] + publicCards, 5))
                handScores[i] = max(scoreHand(hc) for hc in handCombos)

        # Calculate the winning score and number of winners
        winningScore = max(handScores)
        numberWinners = handScores.count(winningScore)

        # Pay out winnings
        for i in range(0, numberPlayersPerGame):
            if handScores[i] == winningScore:
                if numberWinners > 1:
                    playersMoney[i] += int(pot/numberWinners)
                else:
                    playersMoney[i] += pot

        # Kick out anyone with no more money
        playerStillInGame = list(playersMoney[i] > 0 for i in range(0, numberPlayersPerGame))

        # If only one person is left with money, stop playing
        if playersMoney.count(0) == numberPlayersPerGame - 1:
            stillPlaying = False

        handsPlayed += 1

    # Rate all players' fitness by how much money they earned
    for i in range(0, numberPlayersPerGame):
        players[i].fitnessRatings.append(playersMoney[i])


def evalPopulation(population, bestAncestors, minimumEvaluations):
    """
    Evaluate the fitness of the entire population, with list of best ancestors to prevent cycling.
    Each member will be evaluated a minimum number of times
    """
    for p in population:
        p.clearFitness()
    if len(bestAncestors) > maxAncestorsUsed:
        ancestorsUsed = random.sample(bestAncestors, maxAncestorsUsed)
    else:
        ancestorsUsed = bestAncestors
    for a in ancestorsUsed:
        a.clearFitness()
    evalGroups = makeGroups(population + ancestorsUsed, numberPlayersPerGame, minimumEvaluations)
    ###if __name__ == '__main__':
    ###    evalPool = ThreadPool()
    ###    evalPool.map(evalFitness, evalGroups)
    ###    evalPool.join()
    for g in evalGroups:
        evalFitness(g)
    for p in population:
        p.fitness = statistics.mean(p.fitnessRatings) - (parsimonyPressure * len(getAllNodes(p.baseNode)))
    return

def mutate(AI, depth, terminateChance):
    """
    Replace a random node in the AI's decision tree with max depth and terminate chance
    """
    nodeToReplace = random.choice(getAllNodes(AI.baseNode))
    newNode = generateRandomDecisionTree(nodeToReplace.returnType, depth, terminateChance)
    AI.replaceNode(nodeToReplace, newNode)

def recombine(AI1, AI2, defaultDepthLimit, defaultTerminateChance):
    """
    Swap a node in AI1 with a compatible node in AI2. If no compatible node is found, both are mutated instead
    """
    nodeToReplaceInAI1 = random.choice(getAllNodes(AI1.baseNode))
    compatibleNodesInAI2 = list([n for n in getAllNodes(AI2.baseNode) if n.returnType == nodeToReplaceInAI1.returnType])
    if len(compatibleNodesInAI2) == 0:
        AI1.replaceNode(nodeToReplaceInAI1, generateRandomDecisionTree(nodeToReplaceInAI1.returnType, defaultDepthLimit, defaultTerminateChance))
        nodeToReplaceInAI2 = random.choice(getAllNodes(AI2.baseNode))
        AI2.replaceNode(nodeToReplaceInAI2, generateRandomDecisionTree(nodeToReplaceInAI2.returnType, defaultDepthLimit, defaultTerminateChance))
    else:
        nodeToReplaceInAI2 = random.choice(compatibleNodesInAI2)
        newNodeForAI1 = copy.deepcopy(nodeToReplaceInAI2)
        newNodeForAI2 = copy.deepcopy(nodeToReplaceInAI1)
        AI1.replaceNode(nodeToReplaceInAI1, newNodeForAI1)
        AI2.replaceNode(nodeToReplaceInAI2, newNodeForAI2)

def simplifyTree(AI, tree):
    """
    # Removes redundancies in AI nodes
    """
    for b in tree.branches:
        simplifyTree(AI, b)    
    if all(type(b) == constantNumberNode for b in tree.branches):
        if type(tree) == addNode:
            replacementNode = constantNumberNode()
            replacementNode.value = tree.branches[0].value + tree.branches[1].value
            replacementNode.op = "const num " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)
        elif type(tree) == subNode:
            replacementNode = constantNumberNode()
            replacementNode.value = tree.branches[0].value - tree.branches[1].value
            replacementNode.op = "const num " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)
        elif type(tree) == multiplyNode:
            replacementNode = constantNumberNode()
            replacementNode.value = tree.branches[0].value * tree.branches[1].value
            replacementNode.op = "const num " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)
        elif type(tree) == divideNode:
            replacementNode = constantNumberNode()
            left = tree.branches[0].value
            right = tree.branches[1].value
            try:
                replacementNode.value = int(left / right)
            except ZeroDivisionError:
                replacementNode.value = 2000000000
            except OverflowError:
                replacementNode.value = 1000000000
            replacementNode.op = "const num " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)
        elif type(tree) == moduloNode:
            replacementNode = constantNumberNode()
            if tree.branches[1].value == 0:
                replacementNode.value = 0
            else:
                replacementNode.value = tree.branches[0].value % tree.branches[1].value
            replacementNode.op = "const num " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)
        elif type(tree) == greaterThanNode:
            replacementNode = constantBooleanNode()
            replacementNode.value = (tree.branches[0].value > tree.branches[1].value)
            replacementNode.op = "const bool " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)
        elif type(tree) == equalNode:
            replacementNode = constantBooleanNode()
            replacementNode.value = (tree.branches[0].value == tree.branches[1].value)
            replacementNode.op = "const bool " + str(replacementNode.value)
            AI.replaceNode(tree, replacementNode)

    elif type(tree) == andNode and any(type(b) == constantBooleanNode and b.value == False for b in tree.branches):
        replacementNode = constantBooleanNode()
        replacementNode.value = False
        replacementNode.op = "const bool " + str(replacementNode.value)
        AI.replaceNode(tree, replacementNode)

    elif type(tree) == orNode and any(type(b) == constantBooleanNode and b.value == True for b in tree.branches):
        replacementNode = constantBooleanNode()
        replacementNode.value = True
        replacementNode.op = "const bool " + str(replacementNode.value)
        AI.replaceNode(tree, replacementNode)

    elif type(tree) == notNode  and type(tree.branches[0]) == constantBooleanNode:
        replacementNode = constantBooleanNode()
        replacementNode.value = not tree.branches[0].value
        replacementNode.op = "const bool " + str(replacementNode.value)
        AI.replaceNode(tree, replacementNode)

    elif (type(tree) == ifNodeAction or type(tree) == ifNodeNumber) and type(tree.branches[0]) == constantBooleanNode:
        if tree.branches[0].value:
            AI.replaceNode(tree, tree.branches[1])
        elif not tree.branches[0].value:
            AI.replaceNode(tree, tree.branches[2])

                    
def KTournamentSelection(population, numToSelect, K):
    """
    Selects a number of population members with K tournament selection, assuming fitness is already calculated
    """
    results = []
    for i in range(0, numToSelect):
        tournament = random.sample(list(p for p in population if p not in results), K)
        selectedIndividual = next(p for p in tournament if p.fitness == max(p.fitness for p in tournament))
        results.append(selectedIndividual)

    return results

def printDecisionTree(tree, numIndents=0):
    if numIndents == 0:
        print(tree.op)
    else:
        print((numIndents-1) * "    " + "|___" + tree.op)
    for b in tree.branches:
        printDecisionTree(b, numIndents + 1)

# MAIN ------------------------------------------------------------------------------------------

startTime = time.time()
random.seed(1)
# generate initial population
population = []
for i in range(0, populationSize):
    newAI = AI()
    newAI.baseNode = generateRandomDecisionTree('action', mutationTreeDepth, mutationTerminateChance)
    population.append(newAI)

print("Initial population evaluation...")

# evaluate and sort initial population
evalPopulation(population, [], numberEvaluationsPerMember)
population.sort(key=lambda p: p.fitness, reverse=True)

bestAncestors = []

for j in range(0, numberGenerations):
    print("Generation " + str(j))
    # Parent selection/children production
    # Population should already be sorted by fitness, descending, from survivor selection
    print("    Generating children...")
    for k in range(0, int(numberChildrenPerGeneration/2)):
        parent1, parent2 = KTournamentSelection(population, 2, KTournamentK)
        newChild1 = AI()
        newChild1.baseNode = copy.deepcopy(parent1.baseNode)
        newChild2 = AI()
        newChild2.baseNode = copy.deepcopy(parent2.baseNode)
        newChild1.parents = [parent1, parent2]
        newChild2.parents = [parent1, parent2]
        recombine(newChild1, newChild2, mutationTreeDepth, mutationTerminateChance)
        population.append(newChild1)
        population.append(newChild2)

    # Population mutation
    print("    Mutating...")
    for p in population:
        if random.random() < mutationChance:
            mutate(p, mutationTreeDepth, mutationTerminateChance)
            
    # Population simplification
    print("    Simplifying...")
    for p in population:
        simplifyTree(p, p.baseNode)
			
    # Fitness evaluation
    print("    Evaluating population...")
    evalPopulation(population, bestAncestors, numberEvaluationsPerMember)

    # Survivor selection
    print("    Selecting survivors...")
    population.sort(key=lambda p: p.fitness, reverse=True)
    population = population[:populationSize]
	
    # Best ancestor recording
    print("    Recording ancestor...")
    bestAncestors.append(population[0])

    
best = population[0]
printDecisionTree(best.baseNode)
print("Final fitness: " + str(best.fitness))

print("Time elapsed: " + str(time.time() - startTime))