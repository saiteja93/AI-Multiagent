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
        #First we check if successor state is a WIN state. If yes, we return the maximum value for our evaluation function.
        if successorGameState.isWin(): return float("inf")

        #For my evaluation function, I decided to consider all the following things.
        #One - Minimum distance to food as our agent should always try to get closer to the food thats left in the grid, so In our Evaluation function it will have a negative weight.
        if len(newFood.asList()) > 0:
            MinDistancetoFood = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        else: MinDistancetoFood = 0
        #Two - Shortest distance to a non_scared ghost or scared ghost with less than 2 moves left. We want to maximise this. i.e if the action is taking us to a state which is further away from the non-scared ghost or scared ghost with less than 2 one move left, that state is more preferred. Hence its weight is positive in our evaluation function.

        MinghostDistance = min([manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates if ghost.scaredTimer <= 2])
        #checking if this distance is 0, thats the undesirable state with least reward.
        if MinghostDistance == 0: return float("-inf")
        else: MinghostDistance = -10/MinghostDistance
        #Three - the main aim of our Pacman is to devour food, which means, it should prefer going into a state where there is food than someother state without food (unless any external agent,like, the ghost, makes it too)


        return successorGameState.getScore() - 2* MinDistancetoFood + MinghostDistance - 60 * len(newFood.asList())

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
    
      #Your minimax agent (question 2)
    

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
      """


        #since, we start at the top with the max agent(Pacman) going first. We initialise the probe, Agent_ID to 0 and call the helper function.
        probe = 0
        Agent_ID = 0
        val = self.helper(gameState, Agent_ID, probe)
        #The return statements are all done in the form of a list , with val[0] being the value and val[1] being the optimum action at each level.
        return val[1]



        #The sole purpose of the helper function is to take care of the terminal conditions.
    def helper(self, gameState, Agent_ID, probe):
        """

        :param gameState: The current gameState of the game.
        :param Agent_ID: The current Agent_ID of the agent calling the helper function
        :param probe: the depth of the tree explored
        :return: calls the respective max_agent or the min_agent based on the Agent_ID.
        """
        #Since in the max_agent and the min_agent functions we increase the agent_id by 1, and call be min_agent or the max_agent based on the resultant Agent_ID, we need to make sure that the Agent_ID doesnt go out of bounds than the number of agents in the game. Since, the pacman gets Agent_ID of 0, the rest are numbered from 1,number of agents -1. So, when we are incrementing the Agent_ID by 1, when it reaches the value of number of agents, we reinitialise it 0, there by calling the max_agents. This means that, the min_agents have all played their turn and its Max_agent's turn.
        if Agent_ID == gameState.getNumAgents():
            Agent_ID = 0
            probe += 1
        #If we reach a state, where there are no legal actions, then we return the evaluation function value.
        # This case took me a long time to figure out as I didnt think there would be state within self.depth which do not have legalActions.
        if not gameState.getLegalActions(Agent_ID): return [self.evaluationFunction(gameState)]
        #As per the problem statement, once we reach the specified depth, we return the score from the evaluation function.
        if probe == self.depth: return [self.evaluationFunction(gameState)]
        #This part is the crux of the helper function. It uses the Agent_ID and calls the respective functions (max_agent, min_agent)
        if Agent_ID == 0:
            return self.max_agent(gameState, Agent_ID, probe)
        else: return self.min_agent(gameState, Agent_ID,probe)

    def max_agent(self,gameState, Agent_ID , probe):
        """

        :param gameState: The current gameState of the game.
        :param Agent_ID: The current Agent_ID of the agent calling the helper function
        :param probe: the depth of the tree explored.
        :return: returns a list of [value,action], which is the 'value' for optimum 'action' from the given gameState.
        """
        #As per the algorithm for minimax, we initialise the value to -infinity and then increase its value to the maximum of all values of the min_agents.
        value = [float("-inf")]

        #For each action the max_agent takes, the next min_agent plays its turn after it and from each of their value, we update the initial value to get to the maximum and also store the max_Action that leads to the max_value in a list and return it.
        for legal in gameState.getLegalActions(Agent_ID):

            max_val = self.helper(gameState.generateSuccessor(Agent_ID,legal),Agent_ID +1, probe)
            #Updating the value incase its greater than the previous max_value.
            if max_val[0] >= value[0]:
                value = [max_val[0],legal]

        return value


    def min_agent(self, gameState, Agent_ID, probe):
        """

        :param gameState: The current gameState of the game.
        :param Agent_ID: The current Agent_ID of the agent calling the helper function
        :param probe: the depth of the tree explored.
        :return: returns a list of [value,action], which is the 'value' for optimum 'action' from the given gameState.
        """
        # As per the algorithm for minimax, we initialise the value to infinity and then deccrease its value to the minimum of all values that go after it till it gets its turn again.
        value = [float("inf")]
        # For each action the min_agent takes, the agent with the next Agent_ID plays its turn after it and from each of their value, we update the initial value to get to the maximum and also store the max_Action that leads to the max_value in a list and return it.
        for legal in gameState.getLegalActions(Agent_ID):
            min_val = self.helper(gameState.generateSuccessor(Agent_ID, legal), Agent_ID + 1, probe)

            #Updating the values incase, if the previous minimum is greater than the current minimum.
            if min_val[0] <= value[0]:
                value = [min_val[0],legal]


        return value



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
      Your minimax agent with alpha-beta pruning (question 3)
    """

  def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # since, we start at the top with the max agent(Pacman) going first. We initialise the probe, Agent_ID to 0 and call the helper function
        probe = 0
        Agent_ID = 0
        #We initialize the value of alpha and beta to -(infinity), (infinity) respectively.
        alpha = float("-inf")
        beta = float("inf")
        val = self.helper(gameState, Agent_ID, probe,alpha,beta)
        # The return statements are all done in the form of a list , with val[0] being the value and val[1] being the optimum action at each level.
        return val[1]

  def helper(self, gameState, Agent_ID, probe,alpha,beta):
      """

      :param gameState: The present game state of the game
      :param Agent_ID: The agent who's turn it is. (0 - Pacman, the rest are ghosts)
      :param probe: The depth of the tree currently explored
      :param alpha: The current value of alpha
      :param beta: The current value of beta
      :return: The list of [value,action] at each game state which give the optimal action and the value obtained by that optimal state.
      """
      # Since in the max_agent and the min_agent functions we increase the agent_id by 1, and call be min_agent or the max_agent based on the resultant Agent_ID, we need to make sure that the Agent_ID doesnt go out of bounds than the number of agents in the game. Since, the pacman gets Agent_ID of 0, the rest are numbered from 1,number of agents -1. So, when we are incrementing the Agent_ID by 1, when it reaches the value of number of agents, we reinitialise it 0, there by calling the max_agents. This means that, the min_agents have all played their turn and its Max_agent's turn.
      #The code in this part of the function is exactly similar to the minimax agent.
      if Agent_ID == gameState.getNumAgents():
          Agent_ID = 0
          probe += 1
          # If we reach a state, where there are no legal actions, then we return the evaluation function value.
          # This case took me a long time to figure out as I didnt think there would be state within self.depth which do not have legalActions.

      if len(gameState.getLegalActions(Agent_ID)) ==0: return [self.evaluationFunction(gameState)]
      # As per the problem statement, once we reach the specified depth, we return the score from the evaluation function.
      if probe == self.depth: return [self.evaluationFunction(gameState)]
      # This part is the crux of the helper function. It uses the Agent_ID and calls the respective functions (max_agent, min_agent)
      if Agent_ID == 0:
          return self.max_agent(gameState, Agent_ID, probe,alpha,beta)
      else:
          return self.min_agent(gameState, Agent_ID, probe,alpha,beta)


  def max_agent(self, gameState, Agent_ID, probe,alpha,beta):
      """

      :param gameState: The present game state of the game
      :param Agent_ID: The agent who's turn it is. (0 - Pacman, the rest are ghosts)
      :param probe: The depth of the tree currently explored
      :param alpha: The current value of alpha
      :param beta: The current value of beta
      :return: The list of [value,action] at each game state which give the optimal action and the value obtained by that optimal state.
      """

      # As per the algorithm for minimax, we initialise the value to -infinity and then increase its value to the maximum of all values of the min_agents.
      value = [float("-inf")]
      #The major difference from the minimax agent written above to this one is we prune if the value ever gets higher than beta. This makes our computation faster.
      for action in gameState.getLegalActions(Agent_ID):
          max_val = self.helper(gameState.generateSuccessor(Agent_ID, action), Agent_ID + 1, probe,alpha,beta)
          if max_val[0] >= value[0]:
              value = [max_val[0],action]

          if value[0] > beta: return value
          #We update the value of alpha, using the following formula
          alpha = max(value[0],alpha)

      return value

  def min_agent(self, gameState, Agent_ID, probe,alpha,beta):
      """
      :param gameState: The present game state of the game
      :param Agent_ID: The agent who's turn it is. (0 - Pacman, the rest are ghosts)
      :param probe: The depth of the tree currently explored
      :param alpha: The current value of alpha
      :param beta: The current value of beta
      :return: The list of [value,action] at each game state which give the optimal action and the value obtained by that optimal state.
      """

      value = [float("inf")]

      #The only difference between the minimax function and this, we return if the value ever gets lower than alpha. This increases our computation power.
      for action in gameState.getLegalActions(Agent_ID):

          min_val = self.helper(gameState.generateSuccessor(Agent_ID, action), Agent_ID + 1, probe,alpha,beta)
          if min_val[0] <= value[0]:
              value = [min_val[0],action]

          if value[0] < alpha:
              return value
          #We update the value of beta, using the following formula
          beta = min(value[0], beta)

      return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    #Since, we were not taught the expectimaxAgent in class, I used the Minimax agent in this section and worked more on the evaluation function instead.
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
      """


        #since, we start at the top with the max agent(Pacman) going first. We initialise the probe, Agent_ID to 0 and call the helper function.
        probe = 0
        Agent_ID = 0
        val = self.helper(gameState, Agent_ID, probe)
        #The return statements are all done in the form of a list , with val[0] being the value and val[1] being the optimum action at each level.
        return val[1]



        #The sole purpose of the helper function is to take care of the terminal conditions.
    def helper(self, gameState, Agent_ID, probe):
        """

        :param gameState: The current gameState of the game.
        :param Agent_ID: The current Agent_ID of the agent calling the helper function
        :param probe: the depth of the tree explored
        :return: calls the respective max_agent or the min_agent based on the Agent_ID.
        """
        #Since in the max_agent and the min_agent functions we increase the agent_id by 1, and call be min_agent or the max_agent based on the resultant Agent_ID, we need to make sure that the Agent_ID doesnt go out of bounds than the number of agents in the game. Since, the pacman gets Agent_ID of 0, the rest are numbered from 1,number of agents -1. So, when we are incrementing the Agent_ID by 1, when it reaches the value of number of agents, we reinitialise it 0, there by calling the max_agents. This means that, the min_agents have all played their turn and its Max_agent's turn.
        if Agent_ID == gameState.getNumAgents():
            Agent_ID = 0
            probe += 1
        #If we reach a state, where there are no legal actions, then we return the evaluation function value.
        # This case took me a long time to figure out as I didnt think there would be state within self.depth which do not have legalActions.
        if not gameState.getLegalActions(Agent_ID): return [self.evaluationFunction(gameState)]
        #As per the problem statement, once we reach the specified depth, we return the score from the evaluation function.
        if probe == self.depth: return [self.evaluationFunction(gameState)]
        #This part is the crux of the helper function. It uses the Agent_ID and calls the respective functions (max_agent, min_agent)
        if Agent_ID == 0:
            return self.max_agent(gameState, Agent_ID, probe)
        else: return self.min_agent(gameState, Agent_ID,probe)

    def max_agent(self,gameState, Agent_ID , probe):
        """

        :param gameState: The current gameState of the game.
        :param Agent_ID: The current Agent_ID of the agent calling the helper function
        :param probe: the depth of the tree explored.
        :return: returns a list of [value,action], which is the 'value' for optimum 'action' from the given gameState.
        """
        #As per the algorithm for minimax, we initialise the value to -infinity and then increase its value to the maximum of all values of the min_agents.
        value = [float("-inf")]

        #For each action the max_agent takes, the next min_agent plays its turn after it and from each of their value, we update the initial value to get to the maximum and also store the max_Action that leads to the max_value in a list and return it.
        for legal in gameState.getLegalActions(Agent_ID):

            max_val = self.helper(gameState.generateSuccessor(Agent_ID,legal),Agent_ID +1, probe)
            #Updating the value incase its greater than the previous max_value.
            if max_val[0] >= value[0]:
                value = [max_val[0],legal]

        return value


    def min_agent(self, gameState, Agent_ID, probe):
        """

        :param gameState: The current gameState of the game.
        :param Agent_ID: The current Agent_ID of the agent calling the helper function
        :param probe: the depth of the tree explored.
        :return: returns a list of [value,action], which is the 'value' for optimum 'action' from the given gameState.
        """
        # As per the algorithm for minimax, we initialise the value to infinity and then deccrease its value to the minimum of all values that go after it till it gets its turn again.
        value = [float("inf")]
        # For each action the min_agent takes, the agent with the next Agent_ID plays its turn after it and from each of their value, we update the initial value to get to the maximum and also store the max_Action that leads to the max_value in a list and return it.
        for legal in gameState.getLegalActions(Agent_ID):
            min_val = self.helper(gameState.generateSuccessor(Agent_ID, legal), Agent_ID + 1, probe)

            #Updating the values incase, if the previous minimum is greater than the current minimum.
            if min_val[0] <= value[0]:
                value = [min_val[0],legal]


        return value



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    #The factors I'm taking into consideration are,
    #One - Minimum food distance
    getFoodpos = currentGameState.getFood().asList()
    current_pacman_pos = currentGameState.getPacmanPosition()

    if currentGameState.getNumFood() != 0:
        Min_Distance = min([manhattanDistance(current_pacman_pos,food) for food in getFoodpos])
    else: return float("inf")
    #Two: number of Food Pellets remaning - the aim is to pick the move in the successor state's that reduces the number of Food Pellets.
    total_remaining_food = currentGameState.getNumFood()
    #Three: now, the Ghost positions. We consider all of them.

    GPositions = currentGameState.getGhostPositions()
    MinghostDistance = min([manhattanDistance(ghost, current_pacman_pos) for ghost in GPositions])
    # checking if this distance is 0, thats the undesirable state with least reward.
    if MinghostDistance <= 2:
        return float("-inf")

    #Four: The presence of capsules that we haven't considered for the first evaluation function.
    capsule_count = len(currentGameState.getCapsules())
    return  (-2)*Min_Distance - (600) * total_remaining_food + currentGameState.getScore() + 10*capsule_count





# Abbreviation
better = betterEvaluationFunction
