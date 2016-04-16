import random
import statistics
import time
import copy
from multiprocessing import Pool as ThreadPool
from AINodes import *

memorySize = 256

playersPerGame = 2

populationSize = 20
numberGenerations = 10
numberChildrenPerGeneration = 20
numberEvaluationsPerMember = 30
parsimonyPressure = .00005
maxAncestorsUsed = 30

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
    def clearFitness(self):
        self.fitnessRatings = []
        self.fitness = 0
    def clearMem(self):
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


def evalFitness(players):
    """
    Evaluates the fitness of the given AI players and records it in their fitness evaluations list
    """
    stonePiles = []
    numberStonePiles = random.randint(2,8)
    for i in range(0, numberStonePiles):
        stonePiles.append(random.randint(10,100))
    environment = (len(stonePiles), stonePiles)	
    playerAtTurnIndex = 0
    numberPlayers = len(players)
    gameOn = True
    winner = None
    while(gameOn):
        pileToTakeFrom, numberStonesToTake = players[playerAtTurnIndex].getDecision(environment)
        if numberStonesToTake < 1 or pileToTakeFrom >= len(stonePiles) or pileToTakeFrom < 0:
            winner = players[playerAtTurnIndex - 1]
            gameOn = False
            break
        else:
            pileToTakeFrom %= len(stonePiles)
            numberStonesToTake = max(numberStonesToTake, 1)        
            stonePiles[pileToTakeFrom] -= numberStonesToTake
            if stonePiles[pileToTakeFrom] <= 0:
                del stonePiles[pileToTakeFrom]
            if len(stonePiles) == 0:
                winner = players[playerAtTurnIndex]
                gameOn = False
            else:
                playerAtTurnIndex = (playerAtTurnIndex + 1) % numberPlayers
    for p in players:
        if p is winner:
            p.fitnessRatings.append(1)
        else:
            p.fitnessRatings.append(0)
    return

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
    evalGroups = makeGroups(population + ancestorsUsed, playersPerGame, minimumEvaluations)
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

                    
def KTournamentSelection(population, numToSelect):
    """
    Selects a number of population members with K tournament selection, assuming fitness is already calculated
    Goes faster if population is already sorted by fitness, descending
    """
    results = []
    for i in range(0, numToSelect):
        rnd = random.random() * sum([p.fitness for p in population if p not in results])
        for p in population:
            if p not in results:
                rnd -= p.fitness
                if rnd < 0:
                    results.append(p)
                    break
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
# generate initial population
population = []
for i in range(0, populationSize):
    newAI = AI()
    newAI.baseNode = generateRandomDecisionTree('action', mutationTreeDepth, mutationTerminateChance)
    population.append(newAI)

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
        parent1, parent2 = KTournamentSelection(population, 2)
        newChild1 = copy.deepcopy(parent1)
        newChild2 = copy.deepcopy(parent2)
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
    print("    Population length: " + str(len(population)))
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