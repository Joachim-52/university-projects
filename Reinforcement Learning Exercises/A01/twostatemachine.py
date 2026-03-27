import mdp
from enum import Enum

class TwoStateMachine(mdp.MDP):
    """
    A simple Markov Process with two states and two actions.
    A toy can be upright or prone, with actions to walk or stand up.
    """

    Actions = Enum("Actions", "stand walk")
    States = Enum("States", "upright prone")

    def __init__(self):
        self._rewards = {
            (TwoStateMachine.States.upright,
             TwoStateMachine.Actions.walk,
             TwoStateMachine.States.upright): 1,
            (TwoStateMachine.States.upright,
             TwoStateMachine.Actions.walk,
             TwoStateMachine.States.prone): 0,
            (TwoStateMachine.States.prone,
             TwoStateMachine.Actions.stand,
             TwoStateMachine.States.upright): 0,
        }

        self._probs = {
            (TwoStateMachine.States.upright,
             TwoStateMachine.Actions.walk,
             TwoStateMachine.States.upright): 0.8,
            (TwoStateMachine.States.upright,
             TwoStateMachine.Actions.walk,
             TwoStateMachine.States.prone): 0.2,
            (TwoStateMachine.States.prone,
             TwoStateMachine.Actions.stand,
             TwoStateMachine.States.upright): 1.0,
        }

    def R(self, s1, a, s2):
        """Return reward for transition s1 -[a]-> s2."""
        return self._rewards[(s1, a, s2)]

    def P(self, s1, a, s2):
        """Return probability for transition s1 -[a]-> s2."""
        return self._probs[(s1, a, s2)]

    def applicable_actions(self, s):
        """Return actions available in state s."""
        aa = []
        for s2 in TwoStateMachine.States:
            for a in TwoStateMachine.Actions:
                if (s, a, s2) in self._rewards:
                    aa.append(a)
        return set(aa)

    def successor_states(self, s, a):
        """Return all states reachable from s using action a."""
        return {s2 for (s1, a1, s2) in self._rewards if s1 == s and a1 == a}

    def states(self):
        """Return all states in the system."""
        return set(self.States)

    def analytic(self, gamma):
        """
        Compute value function analytically.

        Parameters
        ----------
        gamma : float in ]0,1[
            Discount factor
        Returns
        -------
        dict of state : float
            Values for each state
        """
        p11 = self._probs[(TwoStateMachine.States.upright,
                           TwoStateMachine.Actions.walk,
                           TwoStateMachine.States.upright)]
        r11 = self._rewards[(TwoStateMachine.States.upright,
                             TwoStateMachine.Actions.walk,
                             TwoStateMachine.States.upright)]
        p12 = self._probs[(TwoStateMachine.States.upright,
                           TwoStateMachine.Actions.walk,
                           TwoStateMachine.States.prone)]
        v1 = p11 * r11 / (1 - p11 * gamma - p12 * gamma * gamma)
        v2 = gamma * v1
        return {TwoStateMachine.States.upright: v1,
                TwoStateMachine.States.prone: v2}

if __name__ == "__main__":
    tsm = TwoStateMachine()
    print("Computing tsm.analytic(0.5)")
    va = tsm.analytic(0.5)
    print("Value of state upright: {0}".format(va[TwoStateMachine.States.upright]))

    # Analytic solution by hand:
    # p11 = 0.8, r11 = 1, p12 = 0.2, gamma = 0.5
    # v1 = p11 * r11 / (1 - p11*gamma - p12*gamma^2)
    #    = 0.8 * 1 / (1 - 0.8*0.5 - 0.2*0.25)
    #    = 0.8 / (1 - 0.4 - 0.05)
    #    = 0.8 / 0.55
    #    ≈ 1.4545
    # v2 = gamma * v1 = 0.5 * 1.4545 ≈ 0.7273
    gamma = 0.5
    p11, r11, p12 = 0.8, 1, 0.2
    v1_analytic = p11 * r11 / (1 - p11 * gamma - p12 * gamma ** 2)
    v2_analytic = gamma * v1_analytic
    print("Correct value (hand-computed): {0:.4f}".format(v1_analytic))
    print("Match: {0}".format(abs(va[TwoStateMachine.States.upright] - v1_analytic) < 1e-10))
