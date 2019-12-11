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

    def sensorModel(self, noisyDist, beliefState, pacmanPos):
        """
        The sensor model updating the probability distribution
        with the latest evidence variables given the previous
        values.

        Arguments:
        ----------
        - `beliefState`:
           N*M numpy matrix of probabilities
           where N and M are respectively width and height
           of the maze layout
         `noisyDist`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacmanPos`: 2D coordinates position
          of pacman in the  matrix

        Return:
        -------
        - A belief state representing a probability distribution.
        """
        for i in range(self.walls.width):
            for j in range(self.walls.height):
                if not self.walls[i][j]:
                    d = abs(manhattanDistance(pacmanPos, (i, j))-noisyDist)
                    sample = scipy.stats.norm(0, self.sensor_variance).cdf(d)
                    beliefState[i][j] = beliefState[i][j] * (1-sample)
                else:
                    beliefState[i][j] = 0.0
        return beliefState

    def normalizeProba(self, beliefState):
        """
        Normalises a belief state based on its content

        Arguments:
        ----------
        - `beliefState`:
           N*M numpy matrix of probabilities
           where N and M are respectively width and height
           of the maze layout.
        -------
        - The normalised belief state
        """
        sum = 0.0
        for i in range(self.walls.width):
            for j in range(self.walls.height):
                sum += beliefState[i][j]

        for i in range(self.walls.width):
            for j in range(self.walls.height):
                beliefState[i][j] /= sum
        return beliefState

    def getProba(self, pacmanPos, cellPrev, cellNew):
        """
        Determines the probability for a ghost to move from the cell cellPrev
        to the cell cellNew based on pacman's position and the walls with in
        the game

        Arguments:
        ----------
        - pacmanPos :  the Cartesian position of pacman
        - cellPrev: the origin cell
        - cellNew: the destination cell

        Return:
        -------
        - The probability of the move from cellPrev to cellNew for a ghost
        """
        if(self.walls[cellNew[0]][cellNew[1]]):
            return 0
        else:
            distNew = manhattanDistance(cellNew, pacmanPos)
            distPrev = manhattanDistance(cellPrev, pacmanPos)
            if(distPrev > distNew):
                return 1
            else:
                if(self.ghost_type == 'confused'):
                    return 1
                if(self.ghost_type == 'scared'):
                    return 2
                if(self.ghost_type == 'afraid'):
                    return 2**3

    def ghostModel(self, pacmanPos, cellPos):
        """
        Gathers the probabilities for a ghost to move to each
        adjacent cell from cellPos based on Pacmans' position

        Arguments:
        ----------
        - cellPos: The current cell
        - `pacmanPos`: 2D coordinates position
          of pacman

        Return:
        -------
        - An array containing all the probabilities to move to the adjacent cells.
          adjacent cells are stored clockwise in the array starting with the left cell
        """
        cellXPos = cellPos[0]
        cellYPos = cellPos[1]
        probability = [0.0, 0.0, 0.0, 0.0]
        probability[0] = self.getProba(pacmanPos, cellPos,
                                       (cellXPos-1, cellYPos))
        probability[1] = self.getProba(pacmanPos, cellPos,
                                       (cellXPos, cellYPos + 1))
        probability[2] = self.getProba(pacmanPos, cellPos,
                                       (cellXPos+1, cellYPos))
        probability[3] = self.getProba(pacmanPos, cellPos,
                                       (cellXPos, cellYPos - 1))

        sum = 0.0
        for i in range(len(probability)):
            sum += probability[i]
        if sum == 0.0:
            return probability
        else:
            for i in range(len(probability)):
                probability[i] /= sum
        return probability

    def transitionModel(self, beliefState, pacmanPos):
        """
        The transition model predicting the probability distribution
        over the latest state variables given the previous beliefstate.

        Arguments:
        ----------
        - `beliefState`:N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout
        - `pacmanPosition`: 2D coordinates position
          of pacman in the current beliefState
          where 't' is the current time step

        Return:
        -------
        - A belief state representing a probability distribution.
        """
        temp = beliefState.copy()
        for i in range(0, self.walls.width):
            for j in range(0, self.walls.height):
                temp[i][j] = 0.0

        for i in range(1, self.walls.width-1):
            for j in range(1, self.walls.height-1):
                if self.walls[i][j]:
                    temp[i][j] += 0.0
                else:
                    proba = self.ghostModel(pacmanPos, (i, j))
                    temp[i-1][j] += proba[0]*beliefState[i][j]
                    temp[i][j+1] += proba[1]*beliefState[i][j]
                    temp[i+1][j] += proba[2]*beliefState[i][j]
                    temp[i][j-1] += proba[3]*beliefState[i][j]

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
        for i in range(len(beliefStates)):
            beliefStates[i] = self.transitionModel(beliefStates[i],
                                                   pacman_position)
            beliefStates[i] = self.sensorModel(evidences[i], beliefStates[i],
                                               pacman_position)
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
