from gym.envs.registration import register

register(
        id="overcookedEnv-v1",
        entry_point="gym_cooking.envs:OvercookedEnvironment",
        )
