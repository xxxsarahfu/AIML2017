# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        foodList = [(x, y) for x in range(successorGameState.data.layout.width) for y in range(successorGameState.data.layout.height) if newFood[x][y]]
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in foodList]
        foodDistances.sort()
        ghostDistances = [(ghostState, manhattanDistance(newPos, ghostState.getPosition())) for ghostState in successorGameState.getGhostStates()]
        ghostDistances.sort()
		
        if ghostDistances[0][1] < 3:
            if ghostDistances[0][0].scaredTimer < 1:
                gscore = -666
            else:
                gscore = 666
        else:
            gscore = 0

        if foodDistances:
            return successorGameState.getScore() + ((1 / foodDistances[0]) + gscore)
        else:
            return 7777

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.
      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
          gameState.isWin():
            Returns whether or not the game state is a winning state
          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.Maximize(gameState, 0, 0)[1] # find the highest child state with value


    def isPacman(self, agentIndex):
        return agentIndex == 0

    def calcValue(self, gameState, agentIndex, p):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        else:
            next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if self.isPacman(next_agentIndex):
                if p + 1 <= self.depth:
                    return self.Maximize(gameState, next_agentIndex, p + 1)[0]
                return self.evaluationFunction(gameState)
            else:
                return self.Minimize(gameState, next_agentIndex, p)[0]


    def Maximize(self, gameState, agentIndex, p):
        actionList = gameState.getLegalActions(agentIndex)
        maxi = (float("-inf"), None)
        for action in actionList:
            value = self.calcValue(gameState.generateSuccessor(agentIndex, action), agentIndex, p)
            if value > maxi[0]:
                maxi = (value, action)
        return maxi

    def Minimize(self, gameState, agentIndex, p):
        actionList = gameState.getLegalActions(agentIndex)
        mini = (float("inf"), None)
        for action in actionList:
            value = self.calcValue(gameState.generateSuccessor(agentIndex, action), agentIndex, p)
            if value < mini[0]:
                mini = (value, action)
        return mini


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.Maximize(gameState, (float("-inf"), None), (float("inf"), None), 0, 0)[1]


    def isPacman(self, agentIndex):
        return agentIndex == 0

    def calcValue(self, gameState, alpha, beta, agentIndex, p):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        else:
            next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if self.isPacman(next_agentIndex):
                if p + 1 <= self.depth:
                    return self.Maximize(gameState, alpha, beta, next_agentIndex, p + 1)[0]
                return self.evaluationFunction(gameState)
            else:
                return self.Minimize(gameState, alpha, beta, next_agentIndex, p)[0]


    def Maximize(self, gameState, alpha, beta, agentIndex, p):
        actionList = gameState.getLegalActions(agentIndex)
        maxi = (float("-inf"), None)
        for action in actionList:
            value = self.calcValue(gameState.generateSuccessor(agentIndex, action), alpha, beta, agentIndex, p)
            if maxi[0] < value:
                maxi = (value, action)
            if maxi[0] > beta[0]:
                return maxi
            if maxi[0] > alpha[0]:
                alpha = maxi

        return maxi

    def Minimize(self, gameState, alpha, beta, agentIndex, p):
        actionList = gameState.getLegalActions(agentIndex)
        mini = (float("inf"), None)
        for action in actionList:
            value = self.calcValue(gameState.generateSuccessor(agentIndex, action), alpha, beta, agentIndex, p)
            if mini[0] < value:
                mini = (value, action)
            if mini[0] < alpha[0]:
                return mini
            if mini[0] < beta[0]:
                beta = mini
        return mini


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.Maximize(gameState, 0, 0)[1]


    def isPacman(self, agentIndex):
        return agentIndex == 0

    def calcValue(self, gameState, agentIndex, p):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        else:
            next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if self.isPacman(next_agentIndex):
                if p + 1 <= self.depth:
                    return self.Maximize(gameState, next_agentIndex, p + 1)[0]
                return self.evaluationFunction(gameState)
            else:
                return self.Minimize(gameState, next_agentIndex, p)[0]


    def Maximize(self, gameState, agentIndex, p):
        actionList = gameState.getLegalActions(agentIndex)
        maxi = (float("-inf"), None)
        for action in actionList:
            value = self.calcValue(gameState.generateSuccessor(agentIndex, action), agentIndex, p)
            if value > maxi[0]:
                maxi = (value, action)
        return maxi

    def Minimize(self, gameState, agentIndex, p):
        actionList = gameState.getLegalActions(agentIndex)
        mini = (0, None)
        for action in actionList:
            value = (1.0 / len(actionList)) * self.calcValue(gameState.generateSuccessor(agentIndex, action), agentIndex, p)
            mini = (mini[0] + value, 0)
        return mini


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did> give weight for each factors, since the died event 
	  will minus teh score greatly, i put the weight for ghost more heavier
    """
    "*** YOUR CODE HERE ***"
    
    successorGameState = currentGameState

    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()

    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodList = [(x, y) for x in range(successorGameState.data.layout.width) for y in range(successorGameState.data.layout.height) if newFood[x][y]]
    foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in foodList]
    foodDistances.sort()
    
    ghostDistances = [(ghostState, manhattanDistance(newPos, ghostState.getPosition())) for ghostState in successorGameState.getGhostStates()]
    ghostDistances.sort()

    nearest_ghost = ghostDistances[0]

    weight_Score = 1.0
    weight_Food = 0.5
    weight_FoodDist = 1.2
    weight_Ghost = 1.5

    if nearest_ghost[0].scaredTimer > 0:
        gscore = 300 / max(1, nearest_ghost[1])
    else:
        gscore = -3 * nearest_ghost[1]
  

    if len(foodDistances) > 0:
        return ((successorGameState.getScore() * weight_Score) +(weight_Food * 1 / len(foodList)) +
                weight_FoodDist * (1 / foodDistances[0]) + weight_Ghost * gscore)
    else:
        return 7777

# Abbreviation
better = betterEvaluationFunction