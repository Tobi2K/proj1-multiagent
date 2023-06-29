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
import random, util, math

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = successorGameState.getGhostPositions()
        oldSuperFood = currentGameState.getCapsules()
        newSuperFood = successorGameState.getCapsules()
        # IDEA:
        # Try eating food and running away from ghosts (i.e., keep a certain distance)
        # Add a weighted term to the nearest food and also add a (lower weight) to closest power pellet

        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood]
        superFoodDistances = [manhattanDistance(newPos, superPos) for superPos in newSuperFood]
        # No food left after this move --> winning move
        if len(foodDistances) == 0:
            return math.inf

        # weighted reciprocal for minimum distance to food
        # this prefers moves that step closer to an eaten food
        evaluation = 10 / min(foodDistances)
        
        # if this move would eat a super pellet, eat it
        if len(oldSuperFood) > len(newSuperFood):
            return math.inf
        # if there is super food left, add a weighted reciprocal to the closest super pellet
        # this is weighted less than the food distance to prefer food over super pellets
        if len(superFoodDistances) > 0:
            evaluation += 5 / min(superFoodDistances)
        
        for index, ghostPos in enumerate(newGhostPos):
            # just ignore ghosts, if we ate a super pellet
            # NOTE: This can be optimized to go towards ghost, if scared timer > distance
            if newScaredTimes[index] > 0:
                continue
            # reduce score, if we move toward a non-scared ghost
            elif manhattanDistance(newPos, ghostPos) < 5:
                evaluation *= .5
            # what distance should pacman always try to keep
            # we settled on 2 after testing all values between 1 and 5
            # less than 2 is too risky (the ghosts can easily capture pacman)
            # more than 2 is too scared (pacman hides instead of eating food)
            # It may make sense to set this value based on the size of the grid
            elif manhattanDistance(newPos, ghostPos) < 2:
                evaluation = -math.inf

        return successorGameState.getScore() + evaluation

def scoreEvaluationFunction(currentGameState: GameState):
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

    NOTE: We follow the algorithm given in the lecture with the addition of agent checking.
    We loop through all ghosts and once we reach the last ghost (i.e., agent gameState.getNumAgents() - 1) we loop back to pacman and descend one level.
    """

    def getAction(self, gameState: GameState):
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
        return self.value(gameState, 0, 0, return_action=True)

    # NAMING OF METHODS AS IN THE LECTURE
    def value(self, gameState: GameState, depth, agent, return_action=False):
        """
        Function to asses what a states value is using minimax.
        Returns the utility if we reached our desired depth or if the current state is winning or losing.
        Args:
            gameState (GameState): The current game state.

            depth (int): keeps track of current recursion depth

            agent (int): defines whos turn it is (pacman is agent 0)

            return_action (bool): whether or not we return action or utility
        Returns:
            utility: The utility of our state. Returned if return_action is False
            action: The action that improves our objective most. Returned if return_action is True
        """
        # reached max depth
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # current state is winning or losing
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent == 0:  # pacman's turn ==> maximize
            # max_value returns a tuple of (value, action), we only want the value for recursion
            return  self.max_value(gameState, depth, agent)[0] if not return_action else self.max_value(gameState, depth, agent)[1]
        else: # a ghosts turn ==> minimize
            # min_value returns a tuple of (value, action), we only want the value for recursion
            return self.min_value(gameState, depth, agent)[0] if not return_action else self.min_value(gameState, depth, agent)[1]

    def max_value(self, gameState: GameState, depth, agent):
        """
        Function to get the best action for a MAX player.
        Args:
            gameState (GameState): The current game state.

            depth (int): keeps track of current recursion depth

            agent (int): defines whos turn it is (pacman is agent 0)
        Returns:
            v: The utility score of our best action
            best_action: The best action maximizing our utility
        """
        v = -math.inf
        best_action = Directions.STOP
        successors = gameState.getLegalActions(agent)
        random.shuffle(successors)

        for succ in successors:
            # get evaluation of state for other agents
            state_eval = self.value(gameState.generateSuccessor(agent, succ), depth, agent + 1)
            
            # if our new eval is higher than current best, we replace it and save the corresponding action
            if state_eval > v:
                v = state_eval
                best_action = succ
        
        return v, best_action

    def min_value(self, gameState: GameState, depth, agent):
        """
        Function to get the best action for a MIN player.
        Args:
            gameState (GameState): The current game state.

            depth (int): keeps track of current recursion depth

            agent (int): defines whos turn it is (pacman is agent 0)
        Returns:
            v: The utility score of our best action
            best_action: The best action minimizing our utility
        """
        v = math.inf
        best_action = Directions.STOP
        successors = gameState.getLegalActions(agent)
        random.shuffle(successors)

        for succ in successors:
            if agent == gameState.getNumAgents() - 1:
                # this is the last ghost
                # we need to restart the loop of agents (start with 0 again) and go one level deeper
                # agent index hast to be -1, because it will be incremented in the next call ==> will be 0
                agent = -1 
                depth += 1
            # get evaluation of state for other agents
            state_eval = self.value(gameState.generateSuccessor(agent, succ), depth, agent + 1)

            # if our new eval is lower than current best, we replace it and save the corresponding action
            if state_eval < v:
                v = state_eval
                best_action = succ
        
        return v, best_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)

    NOTE: We follow the algorithm given in the lecture with the addition of agent checking.
    We loop through all ghosts and once we reach the last ghost (i.e., agent gameState.getNumAgents() - 1) we loop back to pacman and descend one level.
    NOTE: This is a pretty much exact copy of minimax.
    We add alpha and beta values and only need to check if our new utility is larger/smaller than beta/alpha
    This change can be found in max_value and min_value
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.value(gameState, 0, 0, -math.inf, math.inf, return_action=True)

     # NAMING OF METHODS AS IN THE LECTURE
    def value(self, gameState: GameState, depth, agent, alpha, beta, return_action=False):
        """
        Function to asses what a states value is using alpha-beta-pruning.
        Returns the utility if we reached our desired depth or if the current state is winning or losing.
        Args:
            gameState (GameState): The current game state.

            depth (int): keeps track of current recursion depth

            agent (int): defines whos turn it is (pacman is agent 0)

            alpha (int): current best for MAX

            beta (int): current best for MIN

            return_action (bool): whether or not we return action or utility
        Returns:
            utility: The utility of our state. Returned if return_action is False
            action: The action that improves our objective most. Returned if return_action is True
        """
        # reached max depth
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # current state is winning or losing
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent == 0:  # pacman's turn ==> maximize
            # max_value returns a tuple of (value, action), we only want the value for recursion
            return  self.max_value(gameState, depth, agent, alpha, beta)[0] if not return_action else self.max_value(gameState, depth, agent, alpha, beta)[1]
        else: # a ghosts turn ==> minimize
            # min_value returns a tuple of (value, action), we only want the value for recursion
            return self.min_value(gameState, depth, agent, alpha, beta)[0] if not return_action else self.min_value(gameState, depth, agent, alpha, beta)[1]

    def max_value(self, gameState: GameState, depth, agent, alpha, beta):
        """
        Function to get the best action for a MAX player.
        Args:
            gameState (GameState): The current game state.

            depth (int): keeps track of current recursion depth

            agent (int): defines whos turn it is (pacman is agent 0)

            alpha (int): current best for MAX

            beta (int): current best for MIN
        Returns:
            v: The utility score of our best action
            best_action: The best action maximizing our utility
        """
        v = -math.inf
        best_action = Directions.STOP
        successors = gameState.getLegalActions(agent)
        
        for succ in successors:
            # get evaluation of state for other agents
            state_eval = self.value(gameState.generateSuccessor(agent, succ), depth, agent + 1, alpha, beta)
            
            # if our new eval is higher than current best, we replace it and save the corresponding action
            if state_eval > v:
                v = state_eval
                best_action = succ

            # if our currently utility is greater than beta, we can prune
            if v > beta:
                return v, best_action
            
            # update alpha
            alpha = max(alpha, v)
        
        return v, best_action

    def min_value(self, gameState: GameState, depth, agent, alpha, beta):
        """
        Function to get the best action for a MIN player.
        Args:
            gameState (GameState): The current game state.

            depth (int): keeps track of current recursion depth

            agent (int): defines whos turn it is (pacman is agent 0)

            alpha (int): current best for MAX

            beta (int): current best for MIN
        Returns:
            v: The utility score of our best action
            best_action: The best action minimizing our utility
        """
        v = math.inf
        best_action = Directions.STOP
        successors = gameState.getLegalActions(agent)
        
        for succ in successors:
            if agent == gameState.getNumAgents() - 1:
                # this is the last ghost
                # we need to restart the loop of agents (start with 0 again) and go one level deeper
                # agent index hast to be -1, because it will be incremented in the next call ==> will be 0
                agent = -1 
                depth += 1
            # get evaluation of state for other agents
            state_eval = self.value(gameState.generateSuccessor(agent, succ), depth, agent + 1, alpha, beta)

            # if our new eval is lower than current best, we replace it and save the corresponding action
            if state_eval < v:
                v = state_eval
                best_action = succ

            # if our currently utility is less than alpha, we can prune
            if v < alpha:
                return v, best_action
            
            # update beta
            beta = min(beta, v)
        
        return v, best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.value(gameState=gameState,depth=0,agent=0,return_action=True)
        
    def value(self, gameState: GameState, depth,agent, return_action=False):
        '''
        Fuction to asses what a states value is using expectimax
        Returns the utility if we reached our desired depth or if the current state is winning or losing
        Args:
        
        Returns:
            utility: The utility of our state. Returned if return_action is False
            action: The action that improves our objective most. Returned if return_action is True
        '''  
         
        #reaches max depth 
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        # current state is winning or losing        
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        if agent == 0: # pacman's turn ==> maximize
            return self.max_val(gameState,depth,agent)[0] if not return_action else self.max_val(gameState,depth,agent)[1]
        else: # a ghosts turn ==> expectimax
            return self.exp_val(gameState,depth,agent)[0] if not return_action else self.exp_val(gameState,depth,agent)[1]
    
    def max_val(self, gameState:GameState, depth, agent):
        """
        Function to get the best action for a MAX player
        Args:
            gameState (GameState): The current game state
            depth (int): keeps track of the current recursion depth
            agent (int): defines whos turn it is (pacman is agent 0)
        Return:
            v: The utility score of the best action
            best_action: The best action maximizing the utility score             
        """
        
        v = -math.inf
        best_action = Directions.STOP
        successors = gameState.getLegalActions(agentIndex=agent)
        random.shuffle(successors)
        
        for succ in successors:
            #get evaluation of the state for other agents (after pacman moved to the current successor) 
            state_eval = self.value(gameState=gameState.generateSuccessor(agentIndex=agent,action=succ), depth=depth,agent=agent+1)  

            #if our new eval is higher than the current best, we replace it and save the corresponding action 
            if state_eval > v:
                v = state_eval
                best_action = succ
        
        return v, best_action
    
    def exp_val(self,gameState:GameState, depth, agent):
        v = 0
        successors = gameState.getLegalActions(agentIndex=agent)
        p = 1/len(successors)
        random.shuffle(successors)

        for succ in successors:
            if agent == gameState.getNumAgents()-1:
                #this is the last ghost
                #we need to restart the loop of agents (starting with pacman (0) again) and go one depth deeper
                # agent index has to be -1, because it will be incremented in the next call ==> will be 0
                agent = -1
                depth += 1
            v += p * self.value(gameState=gameState.generateSuccessor(agentIndex=agent,action=succ),depth=depth,agent=agent+1)

        #cant determine best action, because its an expectimax
        return v, None
    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My evaluation function calculates a linear combination of the following:
        => The score of the current game
        => The number of food left on the map
        => The number of capsules left on the map
        => The number of legal moves for the pacman
        => The manhatten distance to the closest food
        => The manhatten distance to the closest ghost
        => The ghost state
        
    The weights are adjusted by hand but following "common sense" (and 20 minutes of try and error).
        => Score: I didnt temper with the weight for the score of the position because its a good baseline for the evaluation. 
        => Food: To prefer states with less food the amount of food on the map get subtracted from the score.
            -> Less food eaten results in a lower evaluation (same applies for capsules)
        => Capsules: Because the amount of capsules is very low they needed to be highly weighted to have an effect on the pacmans behaviour.
        => Distance to Food: Because we operate at a shallow depth the food is often out of reach for the "amount of food" to have an impact on the evaluation.
            Thats why I also chose to let the distance to the next food have an impact on the evaluation. I prefered the manhatten distance over the 
            search from proj1 because it was move accessible (already imported into the project).
        => Distance to Ghosts: I chose that the distance to the closest ghost needs to have an impact on the evaluation, but the weight in my eval is very low.
            You can see that the pacman is chasing the ghost because he gets a higher eval if the distance is close. 
        => Legal Moves: To avoid getting trapped in some chokepoint where you have to do some additional steps to escape the ghost, the pacman
            prefers squares that have many legal moves. 
        => Scared Ghosts: If the ghosts are scared the pacman can ignore the distance to the ghosts, because they cant hurt him. Chasing the ghosts resulted
            in a generally lower score with my current setup so the pacman will just focus on eating the food. If a scared ghost runs into his path he will
            not hesitate to eat it, because his general score will be higher.
    
    P.S. I have tried other strategies than the "ghost-hunting" but with the eval ghosts this has worked the best so far
        score = 1    * current_score + \
            -20  * number_of_capsules_left + \
            -3   * number_of_food_left + \
            -1   * distance_to_closest_food + \
            1    * amount_pacman_action + \
            (-1 if ghost_is_scared else 1) * distance_to_closest_ghost 
    """
    "*** YOUR CODE HERE ***"
    
    # if the game is lost of won, the position can be evaluated as +/- infinity because if cant get better/worse
    if currentGameState.isLose():
        return -math.inf
    if currentGameState.isWin():
        return math.inf
    
    
    # getting the current score of the game for the evaluation  
    current_score = scoreEvaluationFunction(currentGameState)
    
    # getting the number of food left for the evaluation
    number_of_food_left = currentGameState.getNumFood()
    
    # getting the number of capsures left as part of the evaluation 
    number_of_capsules_left = len(currentGameState.getCapsules())

    # getting the amount of legal pacman actions in the given state for the evaluation
    amount_pacman_action = len(currentGameState.getLegalPacmanActions())
    
    # if one ghost is scared all ghosts are scared right?!
    for ghost_state in currentGameState.getGhostStates():
        ghost_is_scared = True if ghost_state.scaredTimer > 0 else False
        break
        
    # getting the manhatten distance to the closest food for the evaluation
    pacman_position = currentGameState.getPacmanPosition()
    distances_to_food = [manhattanDistance(pacman_position, x) for x in currentGameState.getFood().asList()]
    distance_to_closest_food = min(distances_to_food)

    # getting the manhatten distance to the closest ghost for the evaluation
    distances_to_ghosts = [manhattanDistance(pacman_position, x) for x in currentGameState.getGhostPositions()]
    distance_to_closest_ghost = min(distances_to_ghosts)
    
    #The weights of the parameters were adjusted by hand
    score = 1    * current_score + \
            -20  * number_of_capsules_left + \
            -3   * number_of_food_left + \
            -1   * distance_to_closest_food + \
            1    * amount_pacman_action + \
            (0 if ghost_is_scared else -2.5) * distance_to_closest_ghost 

    return score

# Abbreviation
better = betterEvaluationFunction
