

MAX_STEPS = 500


def test_model(env, agent, num_initial_states=10):
    initial_states = [env.reset() for i in range(num_initial_states)]

    num_success = 0.
    for initial_state in initial_states:
        state = env.reset_specific(initial_state[0], initial_state[1])
        num_iter = 0    # bound number of iterations to prevent infinite loop
        env.render()
        is_done = False
        while not is_done and num_iter < MAX_STEPS:
            # act on environment
            state, reward, is_done, _ = env.step(agent.policy.get_action(state))
            env.render()
            if is_done:
                num_success += 1

            num_iter += 1
    env.close()

    success_rate = num_success / num_initial_states

    return success_rate


