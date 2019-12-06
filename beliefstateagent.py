# Complete this class for all parts of the project

from pacman_module.game import Agent
import numpy as np
from pacman_module import util
import scipy.stats
from pacman_module.util import *
import csv
class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of( walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance
        #measurements variables to be errased before submission##########
        self.nbTurns = 0
        #entropy measurements
        self.entropyArray = [(0,0.0)] * 100
        
        #position
        self.realGhostPosition = [(0,0)] * 100
        self.positionCalculated = [(0.0,0.0)] * 100
        self.bias = [0.0, 0.0] * 100

        #measurements variables to be errased before submission END##########
    def sensorModel(self, noisyDist, beliefState, pacmanPos):

        for i in range(self.walls.width):
            for j in range(self.walls.height):
                if not self.walls[i][j]:
                    beliefState[i][j] = beliefState[i][j] * (1-scipy.stats.norm(0, self.sensor_variance).cdf(abs(manhattanDistance(pacmanPos, (i,j))-noisyDist)))    
                else:
                    beliefState[i][j] = 0.0
        return beliefState


    def normalizeProba(self, beliefState):
        sum = 0.0
        for i in range(self.walls.width):
            for j in range(self.walls.height):
                sum += beliefState[i][j] 

        if sum == 0.0:
            return beliefState

        for i in range(self.walls.width):
            for j in range(self.walls.height):
                beliefState[i][j] /= sum
        return beliefState


    def getProba(self, pacmanPos, cellPrev, cellNow):

        if(self.walls[cellNow[0]][cellNow[1]]):
            return 0
        else:
            distNow = manhattanDistance(cellNow, pacmanPos)
            distPrev = manhattanDistance(cellPrev, pacmanPos)
            if(distPrev > distNow):
                return 1
            else:
                if(self.ghost_type == 'confused'):
                    return 1
                if(self.ghost_type == 'scared'):
                    return 2
                if(self.ghost_type == 'afraid'):
                    return 2**3


    def ghostModel(self, pacmanPos, cellPos):

        cellXPos = cellPos[0]
        cellYPos = cellPos[1]
        probability = [0.0, 0.0, 0.0, 0.0]
        if(not self.walls[cellXPos-1][cellYPos]):
            probability[0] = self.getProba(pacmanPos, cellPos, (cellXPos-1, cellYPos))
        if(not self.walls[cellXPos][cellYPos]):
            probability[1] = self.getProba(pacmanPos, cellPos, (cellXPos, cellYPos + 1))
        if(not self.walls[cellXPos-1][cellYPos]):
            probability[2] = self.getProba(pacmanPos, cellPos, (cellXPos+1, cellYPos))
        if(not self.walls[cellXPos-1][cellYPos]):
            probability[3] = self.getProba(pacmanPos, cellPos, (cellXPos, cellYPos - 1))

        sum = 0.0
        for i in range(len(probability)) :
            sum += probability[i]
        
        if sum == 0.0 :
            return probability
        else:
            for i in range(len(probability)) :
                probability[i] /= sum

        return probability 


    def transitionModel(self, beliefState, pacmanPos):
        temp = beliefState.copy()
        for i in range(0, self.walls.width):
            for j in range(0, self.walls.height):
                temp[i][j] = 0.0

        for i in range(1, self.walls.width-1):
            for j in range(1, self.walls.height-1):

                if self.walls[i][j]:
                    temp[i][j] += 0.0
                else:

                    proba = self.ghostModel(pacmanPos, (i,j))
                    temp[i-1][j] += proba[0] * beliefState[i][j]
                    temp[i][j+1] += proba[1] * beliefState[i][j]
                    temp[i+1][j] += proba[2] * beliefState[i][j]
                    temp[i][j-1] += proba[3] * beliefState[i][j]

        return temp

    def update_belief_state(self, evidences, pacman_position):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        beliefStates = self.beliefGhostStates

        # XXX: Your code here
        #time.sleep(1)
        for i in range(len(beliefStates)):
            
            beliefStates[i] = self.transitionModel(beliefStates[i], pacman_position)
            beliefStates[i] = self.sensorModel(evidences[i], beliefStates[i], pacman_position)
            beliefStates[i] = self.normalizeProba(beliefStates[i])

        # XXX: End of your code

        self.beliefGhostStates = beliefStates

        return beliefStates

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for p in positions:
            true_distance = util.manhattanDistance(p, pacman_position)
            noisy_distances.append(
                np.random.normal(loc=true_distance,
                                 scale=np.sqrt(self.sensor_variance)))

        return noisy_distances

    def getShanonEntropy(self, beliefState):
        
        entropy = 0.0

        for i in range(1, self.walls.width):
            for j in range(1, self.walls.height):
                if(beliefState[i][j] == 0.0):
                    continue
                entropy += beliefState[i][j] * np.log(beliefState[i][j])

        return -entropy


    #still to be finished
    def getEstimatedCoordinates(self, beliefState):
        estimatedX = 0.0
        estimatedY = 0.0

        for i in range(1, self.walls.width):
            for j in range(1, self.walls.height):
                estimatedX += beliefState[i][j] * i
                estimatedY += beliefState[i][j] * j

        return(estimatedX, estimatedY)

    def getPositionBias(self, state, believedPosition, ghostIndex):
        (x,y) = state.getGhostPosition(ghostIndex)
        xBias = abs(x- believedPosition[0])
        yBias = abs(y - believedPosition[1])

        return(xBias, yBias)

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        if self.nbTurns == 100:
            #write csv
            with open('recordedStats.csv', 'w', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow(["Turn", "Entropy", "Ghost X position", "Ghost Y position", "Estimated X", "Estimated Y", "X bias", "Y bias"])
                for i in range(0 , 99):
                    writer.writerow([i, self.entropyArray[i][1], self.realGhostPosition[i][0],self.realGhostPosition[i][1],self.positionCalculated[i][0],self.positionCalculated[i][1],self.bias[i][0],self.bias[i][1]])

        if self.nbTurns < 100:
            self.entropyArray[self.nbTurns] = (self.nbTurns, self.getShanonEntropy(belief_states[0]))
            self.realGhostPosition[self.nbTurns] = state.getGhostPosition(1)
            self.positionCalculated[self.nbTurns] = self.getEstimatedCoordinates(belief_states[0])
            self.bias[self.nbTurns] = self.getPositionBias(state, self.positionCalculated[self.nbTurns] , 1)
            self.nbTurns += 1
        else:
            print("done")
        

        pass

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        newBeliefStates = self.update_belief_state(self._get_evidence(state),
                                                   state.getPacmanPosition())
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates