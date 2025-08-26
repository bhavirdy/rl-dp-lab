###
# Group Members
# Name:Student Number
# Name:Student Number
# Name:Student Number
# Name:Student Number
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt

def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0.0
            # sum over actions
            for a, a_prob in enumerate(policy[s]):
                # sum over possible next states
                if a in env.P[s]:
                    for prob, next_state, reward, done in env.P[s][a]:
                        v += a_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(s, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            if a in env.P[s]:
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
        return action_values

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    policy = np.ones((n_states, n_actions)) / n_actions
    V = np.zeros(n_states)
    
    policy_stable = False
    while not policy_stable:
        # 1. policy eval
        V = policy_evaluation_fn(env, policy)

        # 2. policy improvement
        policy_stable = True
        for s in range(n_states):
            old_action = np.argmax(policy[s])

            action_values = one_step_lookahead(s, V)
            best_action = np.argmax(action_values)

            policy[s] = np.zeros(n_actions)
            policy[s][best_action] = 1.0

            if best_action != old_action:
                policy_stable = False

    return policy, V

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(s, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            if a in env.P[s]:
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
        return action_values

    n_actions = env.action_space.n
    n_states = env.observation_space.n

    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            action_values = one_step_lookahead(s, V)
            v = np.max(action_values)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        action_values = one_step_lookahead(s, V)
        best_action = np.argmax(action_values)
        policy[s] = np.zeros(n_actions)
        policy[s][best_action] = 1.0

    return policy, V

def main():
    # -----------------------------
    # Create Gridworld environment
    # -----------------------------
    env = GridworldEnv(
        shape=[5, 5],
        terminal_states=[24],
        terminal_reward=0,
        step_reward=-1
    )
    state = env.reset()
    print("\nEnvironment:")
    env.render()
    print("")

    # -----------------------------
    # Generate random policy
    # -----------------------------
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    random_policy = np.ones((n_states, n_actions)) / n_actions

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # -----------------------------
    # Evaluate random policy
    # -----------------------------
    v = policy_evaluation(env, random_policy)

    # Print state values as grid
    print("State values (V) as grid:")
    print(np.round(v.reshape(env.shape), 2))

    # Test evaluated policy
    expected_v = np.array([
        -106.81, -104.81, -101.37, -97.62, -95.07,
        -104.81, -102.25, -97.69, -92.40, -88.52,
        -101.37, -97.69, -90.74, -81.78, -74.10,
        -97.62, -92.40, -81.78, -65.89, -47.99,
        -95.07, -88.52, -74.10, -47.99, 0.0
    ])
    try:
        np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
        print("\n✅ Policy evaluation test PASSED! Values match expected results.\n")
    except AssertionError as e:
        print("❌ Policy evaluation test FAILED!")
        print(e)
        print("")

    # -----------------------------
    # Policy iteration
    # -----------------------------
    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")

    # use policy iteration to compute optimal policy and state values
    policy, v = policy_iteration(env)

    # print best action for each state (grid shape)
    action_symbols = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    best_actions = np.argmax(policy, axis=1).reshape(env.shape)
    print("Best actions (policy) as grid:")
    for row in best_actions:
        print(' '.join(action_symbols[a] for a in row))
    
    # print state values as grid
    print("\nState values (V) as grid:")
    print(v.reshape(env.shape))

    # test for policy iteration value function
    expected_v = np.array([
        -8., -7., -6., -5., -4.,
        -7., -6., -5., -4., -3.,
        -6., -5., -4., -3., -2.,
        -5., -4., -3., -2., -1.,
        -4., -3., -2., -1., 0.
    ])
    try:
        np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
        print("\n✅ Policy iteration test PASSED! Values match expected results.\n")
    except AssertionError as e:
        print("❌ Policy iteration test FAILED!")
        print(e)
        print("")

    # -----------------------------
    # Value iteration
    # -----------------------------
    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # use value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)

    # print best action for each state (grid shape)
    action_symbols = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    best_actions = np.argmax(policy, axis=1).reshape(env.shape)
    print("Best actions (policy) as grid:")
    for row in best_actions:
        print(' '.join(action_symbols[a] for a in row))

    # print state values as grid
    print("\nState values (V) as grid:")
    print(v.reshape(env.shape))

    # test for value iteration value function
    expected_v = np.array([
        -8., -7., -6., -5., -4.,
        -7., -6., -5., -4., -3.,
        -6., -5., -4., -3., -2.,
        -5., -4., -3., -2., -1.,
        -4., -3., -2., -1., 0.
    ])
    try:
        np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
        print("\n✅ Value iteration test PASSED! Values match expected results.\n")
    except AssertionError as e:
        print("❌ Value iteration test FAILED!")
        print(e)
        print("")

if __name__ == "__main__":
    main()
