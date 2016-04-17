import random
import statistics
memorySize = 256

class ifNodeNumber:
    """
    Represents an if-then-else in the decision tree that selects a number
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['boolean', 'number', 'number']
        self.branches = [None, None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'if'
    def get(self, memory, environment):
        if self.branches[0].get(memory, environment):
            return self.branches[1].get(memory, environment)
        else:
            return self.branches[2].get(memory, environment)

class ifNodeAction:
    """
    Represents an if-then-else in the decision tree that selects an action
    """

    def __init__(self):
        self.returnType = 'action'
        self.branchTypes = ['boolean', 'action', 'action']
        self.branches = [None, None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'if'
    def get(self, memory, environment):
        if self.branches[0].get(memory, environment):
            return self.branches[1].get(memory, environment)
        else:
            return self.branches[2].get(memory, environment)

class equalNode:
    """
    Represents a number equivalence check node in the decision tree
    """
    def __init__(self):
        self.returnType = 'boolean'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '=='
    def get(self, memory, environment):
        return self.branches[0].get(memory, environment) == self.branches[1].get(memory, environment)

class greaterThanNode:
    """
    Represents a greater-than check node in the decision tree
"""
    def __init__(self):
        self.returnType = 'boolean'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '>'
    def get(self, memory, environment):
        return self.branches[0].get(memory, environment) > self.branches[1].get(memory, environment)

class andNode:
    """
    Represents a boolean and in the decision tree
    """
    def __init__(self):
        self.returnType = 'boolean'
        self.branchTypes = ['boolean', 'boolean']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'and'
    def get(self, memory, environment):
        return self.branches[0].get(memory, environment) and self.branches[1].get(memory, environment)

class orNode:
    """
    Represents a boolean or in the decision tree
    """
    def __init__(self):
        self.returnType = 'boolean'
        self.branchTypes = ['boolean', 'boolean']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'or'
    def get(self, memory, environment):
        return self.branches[0].get(memory, environment) or self.branches[1].get(memory, environment)

class notNode:
    """
    Represents a boolean not in the decision tree
    """
    def __init__(self):
        self.returnType = 'boolean'
        self.branchTypes = ['boolean']
        self.branches = [None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'not'
    def get(self, memory, environment):
        return not self.branches[0].get(memory, environment)

class addNode:
    """
    Represents addition operator in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '+'
    def get(self, memory, environment):
        return self.branches[0].get(memory, environment) + self.branches[1].get(memory, environment)

class subNode:
    """
    Represents subtraction operator in the decision tree
    """

    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '-'
    def get(self, memory, environment):
        return self.branches[0].get(memory, environment) - self.branches[1].get(memory, environment)

class multiplyNode:
    """
    Represents multiplication operator in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '*'
    def get(self, memory, environment):
        return min(self.branches[0].get(memory, environment) * self.branches[1].get(memory, environment), 1000000)

class divideNode:
    """
    Represents division operator in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '/'
    def get(self, memory, environment):
        left = self.branches[0].get(memory, environment)
        right = self.branches[1].get(memory, environment)
        try:
            result = int(left / right)
        except ZeroDivisionError:
            result = 2000000000
        except OverflowError:
            result = 1000000000
        return result
            
class moduloNode:
    """
    Represents modulo operator in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = '%'
    def get(self, memory, environment):
        left = self.branches[0].get(memory, environment)
        right = self.branches[1].get(memory, environment)
        if right == 0:
            return 0
        else:
            return left % right           

class writeMemoryNode:
    """
    Represents write memory operation in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'write'
    def get(self, memory, environment):
        index = int(self.branches[1].get(memory, environment)) % len(memory)
        value = self.branches[0].get(memory, environment)
        memory[index] = value
        return value

class readMemoryNode:
    """
    Represents read memory operation in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number']
        self.branches = [None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'read'
    def get(self, memory, environment):
        index = int(self.branches[0].get(memory, environment)) % len(memory)
        value = memory[index]
        return value

class readEnvironmentNode:
    """
    Represents read environment operation in the decision tree
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = ['number', 'number', 'number']
        self.branches = [None, None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'look'
    def get(self, memory, environment):
        a = self.branches[0].get(memory, environment) % len(environment)
        if a in (0, 2):
            if len(environment[a]) == 0:
                value = -1
            else:
                b = self.branches[1].get(memory, environment) % len(environment[a])
                c = self.branches[2].get(memory, environment) % len(environment[a][b])
                value = environment[a][b][c]
        elif a in (3, 4):
            b = self.branches[1].get(memory, environment) % len(environment[a])
            value = environment[a][b]
        else:
            value = environment[a]
        return value

class buildActionNode:
    """
    Represents a node that builds an action in the decision tree
    """
    def __init__(self):
        self.returnType = 'action'
        self.branchTypes = ['number', 'number']
        self.branches = [None, None]
        self.isTerminal = False
        self.parent = None
        self.parentBranchIndex = -1
        self.op = 'make action'
    def get(self, memory, environment):
        return int(self.branches[0].get(memory, environment)), int(self.branches[1].get(memory, environment))


class constantNumberNode:
    """
    Represents a constant number terminal node in the decision tree between 0 and 10
    """
    def __init__(self):
        self.returnType = 'number'
        self.branchTypes = []
        self.branches = []
        self.isTerminal = True
        self.parent = None
        self.parentBranchIndex = -1
        self.value = random.randint(0,10)
        self.op = 'const num ' + str(self.value)
    def get(self, memory, environment):
        return self.value

class constantBooleanNode:
    def __init__(self):
        self.returnType = 'boolean'
        self.branchTypes = []
        self.branches = []
        self.isTerminal = True
        self.parent = None
        self.parentBranchIndex = -1
        self.value = random.choice((True, False))
        self.op = 'const bool ' + str(self.value)
    def get(self, memory, environment):
        return self.value

class constantActionNode:
    def __init__(self):
        self.returnType = 'action'
        self.branchTypes = []
        self.branches = []
        self.isTerminal = True
        self.parent = None
        self.parentBranchIndex = -1
        self.value = random.randint(0,10), random.randint(0,10)
        self.op = 'const action ' + str(self.value[0]) + ' ' + str(self.value[1])
    def get(self, memory, environment):
        return self.value