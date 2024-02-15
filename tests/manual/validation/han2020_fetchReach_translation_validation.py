"""FetchReachCost Environment Translation Validation Script.

This script validates the translation of the 'FetchReachCost' environment from the
'Actor-critic-with-stability-guarantee' repository to the 'stable_gym' package. It does
this by comparing the output of the 'step' method in the translated environment with the
output from the original implementation.

The original output should be placed in a CSV file located in the same directory as this
script. This CSV file is generated by a corresponding script in the
'Actor-critic-with-stability-guarantee' repository.

For detailed usage instructions, please refer to the README.md in this directory.
"""

import stable_gym  # NOTE: Ensures that the latest version of the environment is used. # noqa: F401, E501
import numpy as np
from prettytable import PrettyTable
import textwrap
import gymnasium as gym
import os
import pandas as pd
import math

STEPS = 10
CHECK_TOLERANCE = 1e-2
SEED = 0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_accuracy(number):
    """Get the number of decimal places of a number.

    Args:
        number (float): The number to get the decimal places of.

    Returns:
        int: The number of decimal places of the number.
    """
    number_str = str(number)
    parts = number_str.split(".")

    # If there's no decimal point, the number is an integer.
    if len(parts) == 1:
        return 0  # An integer has zero decimal places.

    # Count the number of digits after the decimal point and return.
    decimal_places = len(parts[1])
    return decimal_places


if __name__ == "__main__":
    print("=== FetchReachCost Environment Translation Validation===")
    print(
        textwrap.dedent(
            f"""
            Welcome to the FetchReachCost environment translation validation script. This
            script initializes the stable-gym FetchReachCost environment and executes
            '{STEPS}' steps to validate in against the original implementation in the
            'Actor-critic-with-stability-guarantee' repository.

            Compare the output of this script with the equivalent script in the
            'Actor-critic-with-stability-guarantee' repository. Matching outputs
            indicate  correct translations.

            Note: These script only compares step outputs, not reset outputs. This is
            due to non-deterministic behaviors between different numpy and gym versions
            used in both packages.
            """
        )
    )

    # Initialize FetchReachCost environment.
    # NOTE: The state is set directly due to non-deterministic behaviour across different
    # numpy and gym versions.
    env_cost = gym.make(
        "FetchReachCost",
        action_space_dtype=np.float32,
        observation_space_dtype=np.float32,
    )
    env_cost.unwrapped.initial_time = 0.4000000000000003
    env_cost.unwrapped.initial_qpos = np.array(
        [
            4.04899887e-01,
            4.80000000e-01,
            2.79906896e-07,
            -2.10804408e-05,
            1.80448057e-10,
            6.00288106e-02,
            9.67580396e-03,
            -8.28231087e-01,
            -3.05625957e-03,
            1.44397975e00,
            2.53423937e-03,
            9.55099996e-01,
            5.96093593e-03,
            1.97805133e-04,
            7.15193042e-05,
        ]
    )
    env_cost.unwrapped.initial_qvel = np.array(
        [
            -8.20972730e-10,
            -5.42827776e-13,
            3.01801009e-07,
            -2.06118511e-05,
            1.60548710e-11,
            7.22090854e-05,
            7.12945378e-04,
            9.39598373e-04,
            -1.38720810e-03,
            -1.63417143e-03,
            1.15341321e-03,
            1.24855991e-03,
            -9.73707041e-04,
            1.18331413e-04,
            -5.71138070e-05,
        ]
    )
    env_cost.reset(seed=SEED)
    env_cost.unwrapped.goal = np.array([1.37384575, 0.81794948, 0.54779444])

    # Create a pretty table to display the results in.
    obs_cols = [
        f"Obs{i+1}" for i in range(env_cost.observation_space["observation"].shape[0])
    ]
    achieved_goal_cols = [f"AchievedGoal{dim}" for dim in ["x", "y", "z"]]
    desired_goal_cols = [f"DesiredGoal{dim}" for dim in ["x", "y", "z"]]
    table = PrettyTable()
    table.field_names = [
        "Step",
        *obs_cols,
        "Reward",
        "Done",
        *achieved_goal_cols,
        *desired_goal_cols,
        "IsSuccess",
    ]

    # Perform N steps for the stable-gym environment comparison.
    # NOTE: Use the same action as in the stable-gym package.
    df = pd.DataFrame(
        columns=[
            "Step",
            *obs_cols,
            "Reward",
            "Done",
            *achieved_goal_cols,
            *desired_goal_cols,
            "IsSuccess",
        ]
    )
    for i in range(STEPS):
        delta = (
            (env_cost.action_space.high - env_cost.action_space.low)[0] / STEPS
        ) * i
        action = np.array(
            [
                env_cost.action_space.low[0] + delta,
                env_cost.action_space.high[1] - delta,
                env_cost.action_space.low[2] + delta,
                env_cost.action_space.high[3] - delta,
            ],
            dtype=np.float32,
        )
        observation, reward, terminated, truncated, info = env_cost.step(action)

        # Store the results in a table.
        done = terminated or truncated
        table.add_row(
            [
                i,
                *observation["observation"],
                reward,
                done,
                *observation["achieved_goal"],
                *observation["desired_goal"],
                info["is_success"],
            ]
        )
        obs_dict = {
            f"Obs{i+1}": [obs] for i, obs in enumerate(observation["observation"])
        }
        achieved_goal_dict = {
            f"AchievedGoal{dim}": [obs]
            for dim, obs in zip(["x", "y", "z"], observation["achieved_goal"])
        }
        desired_goal_dict = {
            f"DesiredGoal{dim}": [obs]
            for dim, obs in zip(["x", "y", "z"], observation["desired_goal"])
        }
        data = {
            "Step": [np.int64(i)],
            "Reward": [reward],
            "Done": [done],
            **obs_dict,
            **achieved_goal_dict,
            **desired_goal_dict,
            "IsSuccess": [
                info["is_success"],
            ],
        }
        new_row = pd.DataFrame(data)
        df = pd.concat([df, new_row], ignore_index=True)

    # Save the results to a CSV file.
    df.to_csv(
        os.path.join(
            SCRIPT_DIR, "results/stableGym_fetchReachCost_translation_validation.csv"
        )
    )
    env_cost.close()
    print(
        "Stable gym fetchReachCost comparison table generated and stored in "
        "'results/stableGym_fetchReachCost_translation_validation.csv'."
    )

    # Print the results of the stable-gym environment steps.
    print("\nStable gym FetchReach comparison table:")
    print(f"{table}\n")

    # Check if the reference CSV file exists.
    if not os.path.isfile(
        os.path.join(SCRIPT_DIR, "results/fetchReach_translation_validation.csv")
    ):
        print(
            "\nNo 'results/fetchReach_translation_validation.csv' file found. Please "
            "run the same script in the 'Actor-critic-with-stability-guarantee' "
            "repository to generate the file and place it in the 'results' folder "
            "found alongside this script to get a comparison result."
        )
        exit()

    # Load the reference fil CSV file.
    df2 = pd.read_csv(
        os.path.join(SCRIPT_DIR, "results/fetchReach_translation_validation.csv"),
    )

    # Print the reference CSV file results as a pretty table.
    table2 = PrettyTable()
    table2.field_names = [
        "Step",
        *obs_cols,
        "Reward",
        "Done",
        *achieved_goal_cols,
        *desired_goal_cols,
        "IsSuccess",
    ]
    df2_tmp = df2.round(7)
    for i, row in df2_tmp.iterrows():
        table2.add_row(row)
    print("\nReference CSV file table:")
    print(f"{table2}\n")

    # Compare the results.
    print(
        "\nComparing stable-gym step results with results in the "
        "'results/fetchReachCost_translation_validation.csv' file."
    )
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    boolean_cols = df.select_dtypes(include=["bool"]).columns
    numeric_close = np.allclose(
        df[numeric_cols].values,
        df2[numeric_cols].values,
        atol=CHECK_TOLERANCE,
        equal_nan=True,
    )
    boolean_equal = (df[boolean_cols] == df2[boolean_cols]).all().all()
    accuracy = min(get_accuracy(df2["Obs1"][0]), get_accuracy(df["Obs1"][0]))
    if numeric_close and boolean_equal:
        print(
            "✅ Test Passed: Results are consistent up to a precision of "
            f"{min(accuracy, abs(math.log10(CHECK_TOLERANCE)))} decimal places."
        )
    else:
        raise ValueError("❌ Test Failed: Results do not match the expected values.")
