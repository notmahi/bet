import hydra
import json
import pathlib
import numpy as np
from pathlib import Path

# BLOCKPUSH COMPUTE METRICS


def block_distance(block, rollout_i, step_i, data):
    """Compute the distance of the block from the previous position.
    Args:
        block: name of the block
        rollout_i: index of the rollout
        step_i: index of the step
        data: dictionary containing the observations
    Returns:
        distance: distance of the block from the previous position
    """
    return np.linalg.norm(data["%s_translation" % block][rollout_i, step_i])


def block_target_distance(block, target, rollout_i, step_i, data):
    """Compute the distance of the block from the target
    Args:
        block: name of the block
        target: name of the target
        rollout_i: index of the rollout
        step_i: index of the step
        data: dictionary containing the observations
    Returns:
        distance: distance of the block from the target
    """
    return np.linalg.norm(
        data["%s_translation" % block][rollout_i, step_i]
        - data["%s_translation" % target][rollout_i, step_i]
    )


def compute_blockpush_metrics(data, tolerance=0.05):
    """Compute the metrics for the blockpush task as described in the paper
    Args:
        data: dictionary containing the observations
        tolerance: tolerance for the push metric, threshold for the reach metric is 1e-3 by default,
        check original repo for more details: https://github.com/notmahi/bet.
    Returns:
        metrics: dictionary containing the metrics for both tables.
    """

    # Table 1 variables initialization
    # References:
    # https://arxiv.org/abs/2206.11251
    # version v2, page 7.

    # Reach
    # Two possible ways of computing the reach:
    # 1. Compute the distance between the position of the block a step t and the position of the block at step t+1.
    # 2. Compute the distance between the block and the arm and check if it is less than a threshold.

    # Method 1 is the one used in the original imlpementation.

    R1 = 0
    R2 = 0
    R1_2 = 0
    R2_2 = 0

    # Push
    # Two possible ways of computing the push:
    # 1. Compute the distance between the block and the target, update the metric
    # if it's lower than the threshold at some step t.
    # 2. Compute the distance between the block and the target, update the metric
    # if it's lower than the threshold at the last step of the rollout.

    # Method 1 is the one used in the original imlpementation.

    P1 = 0
    P2 = 0
    P1_2 = 0
    P2_2 = 0

    # Table 2 variables initialization
    # References:
    # https://arxiv.org/abs/2206.11251
    # version v2, page 7.

    first_reached_red = 0
    first_reached_green = 0

    red_block_red_target = 0
    red_block_green_target = 0
    green_block_red_target = 0
    green_block_green_target = 0

    red_pushed = 0
    green_pushed = 0

    # Coordinates of the blocks, targets and arm at each step, recorded during evaluation.
    obs = collections.OrderedDict(
        block1_translation=data[:, :, 0:2],
        block1_orientation=data[:, :, 2],
        block2_translation=data[:, :, 3:5],
        block2_orientation=data[:, :, 5],
        effector_translation=data[:, :, 6:8],
        effector_target_translation=data[:, :, 8:10],
        target1_translation=data[:, :, 10:12],
        target1_orientation=data[:, :, 12],
        target2_translation=data[:, :, 13:15],
        target2_orientation=data[:, :, 15],
    )

    # Iterate over the rollouts
    for rollout_i, rollout in enumerate(data):

        # Table 1 flags initialization

        # Flags to check if the respective goals have been achieved
        R1_achieved, R2_achieved = False, False
        P1_achieved, P2_achieved = False, False
        # Flags to check if one of the blocks is in one of the targets
        b1t1, b1t2, b2t1, b2t2 = False, False, False, False

        # Initialize the first distance between the blocks as -inf
        distaceb1b1_old = float("-inf")
        distaceb2b2_old = float("-inf")
        # Flags to check if the blocks have moved a certain distance from the previous position
        distanceb1b1_achieved, distanceb2b2_achieved = False, False

        # Flags to check if the respective goals have been achieved for the second method
        R1_2_achieved, R2_2_achieved = False, False
        # Flags to check if the arm is close to one of the blocks
        b1arm, b2arm = False, False

        # For each step in a rollout
        for step_i, step_vector in enumerate(data[rollout_i, :, :]):
            # If both goals have been achieved, break the loop
            # if P1_achieved and P2_achieved:
            #    break
            # If the rollout is finished, break the loop
            if step_i == len(data[rollout_i, :, :]) - 1:
                break

            # Table 1
            # 1. Reach (block is being moved from previous position)

            # Compute the distance between the blocks and the previous position
            distanceb1b1_now = block_distance("block1", rollout_i, step_i, obs)
            distanceb2b2_now = block_distance("block2", rollout_i, step_i, obs)

            # If the distance is the first one, set the old distance to the current one
            if distaceb1b1_old == float("-inf"):
                distaceb1b1_old = distanceb1b1_now
            if distaceb2b2_old == float("-inf"):
                distaceb2b2_old = distanceb2b2_now

            # First check if the first block has been moved
            # If so update R1 and the correpsponding flag
            if not R1_achieved and np.abs(distanceb1b1_now - distaceb1b1_old) > 1e-3:
                # Table 1
                R1 += 1
                distanceb1b1_achieved = True
                R1_achieved = True
                # Table 2
                first_reached_red += 1

            # The same for the second block (R1 isupdated regarding of which of the blocks has been moved first)
            elif not R1_achieved and np.abs(distanceb2b2_now - distaceb2b2_old) > 1e-3:
                # Table 1
                R1 += 1
                distanceb2b2_achieved = True
                R1_achieved = True
                # Table 2
                first_reached_green += 1

            # If at least one of the blocks has been moved, check if the other one has been moved
            if R1_achieved and not R2_achieved:
                # If the block one has been moved first, check if the second one has been moved
                if distanceb1b1_achieved:
                    if np.abs(distanceb2b2_now - distaceb2b2_old) > 1e-3:
                        R2 += 1
                        R2_achieved = True

                # If the block two has been moved first, check if the first one has been moved
                elif distanceb2b2_achieved:
                    if np.abs(distanceb1b1_now - distaceb1b1_old) > 1e-3:
                        R2 += 1
                        R2_achieved = True

            # 2. Reach (block interacts with arm)

            # Compute the distance between the blocks and the arm
            distance_arm_block1 = block_target_distance(
                "block1", "effector", rollout_i, step_i, obs
            )
            distance_arm_block2 = block_target_distance(
                "block2", "effector", rollout_i, step_i, obs
            )

            # If no block has been reached yet:
            if not R1_2_achieved:
                # If the arm is close to the first block, update R1 and the corresponding flag
                if not b1arm and distance_arm_block1 < tolerance:
                    R1_2 += 1
                    b1arm = True
                    R1_2_achieved = True
                # If the arm is close to the second block, update R1 and the corresponding flag
                elif not b2arm and distance_arm_block2 < tolerance:
                    R1_2 += 1
                    b2arm = True
                    R1_2_achieved = True

            # If the first block has been reached, check if the second one has been reached
            if R1_2_achieved and not R2_2_achieved:
                # If the arm is close to the first block, update R2 and the corresponding flag
                if b1arm and distance_arm_block2 < tolerance:
                    R2_2 += 1
                    R2_2_achieved = True
                # If the arm is close to the second block, update R2 and the corresponding flag
                elif b2arm and distance_arm_block1 < tolerance:
                    R2_2 += 1
                    R2_2_achieved = True

            # Push

            # Compute the distance between the blocks and the targets
            distanceb1t1 = block_target_distance(
                "block1", "target1", rollout_i, step_i, obs
            )
            distanceb1t2 = block_target_distance(
                "block1", "target2", rollout_i, step_i, obs
            )
            distanceb2t1 = block_target_distance(
                "block2", "target1", rollout_i, step_i, obs
            )
            distanceb2t2 = block_target_distance(
                "block2", "target2", rollout_i, step_i, obs
            )

            # 1. Push (block enters the target)

            # If no block has been pushed to the target yet
            # and the distance between block one and the target one is less than the tolerance, update P1 and the corresponding flag
            if not P1_achieved and distanceb1t1 < tolerance:
                # Table 1
                P1 += 1
                P1_achieved = True
                b1t1 = True
                # Table 2
                red_pushed += 1
                red_block_red_target += 1
            # Same if block one is pushed to target two
            elif not P1_achieved and distanceb1t2 < tolerance:
                # Table 1
                P1 += 1
                P1_achieved = True
                b1t2 = True
                # Table 2
                red_pushed += 1
                red_block_green_target += 1
            # Same if block two is pushed to target one
            elif not P1_achieved and distanceb2t1 < tolerance:
                # Table 1
                P1 += 1
                P1_achieved = True
                b2t1 = True
                # Table 2
                green_pushed += 1
                green_block_red_target += 1
            # Same if block two is pushed to target two
            elif not P1_achieved and distanceb2t2 < tolerance:
                # Table 1
                P1 += 1
                P1_achieved = True
                b2t2 = True
                # Table 2
                green_pushed += 1
                green_block_green_target += 1

            # If at least one of the blocks has been pushed to the target, check if the other one has been pushed to the target
            if P1_achieved and not P2_achieved:
                # If block one has been pushed to the target first, check if block two has been pushed to target two
                if (
                    b1t1
                    and block_target_distance(
                        "block2", "target2", rollout_i, step_i, obs
                    )
                    < tolerance
                ):
                    # Table 1
                    P2 += 1
                    P2_achieved = True
                    # Table 2
                    green_block_green_target += 1
                    green_pushed += 1
                    # break
                # Same if block one has been pushed to target two, check if block two has been pushed to target one
                elif (
                    b1t2
                    and block_target_distance(
                        "block2", "target1", rollout_i, step_i, obs
                    )
                    < tolerance
                ):
                    # Table 1
                    P2 += 1
                    P2_achieved = True
                    # Table 2
                    green_block_red_target += 1
                    green_pushed += 1
                    # break
                # Same if block two has been pushed to target one, check if block one has been pushed to target two
                elif (
                    b2t1
                    and block_target_distance(
                        "block1", "target2", rollout_i, step_i, obs
                    )
                    < tolerance
                ):
                    # Table 1
                    P2 += 1
                    P2_achieved = True
                    # Table 2
                    red_block_green_target += 1
                    red_pushed += 1
                    # break
                # Same if block two has been pushed to target two, check if block one has been pushed to target one
                elif (
                    b2t2
                    and block_target_distance(
                        "block1", "target1", rollout_i, step_i, obs
                    )
                    < tolerance
                ):
                    # Table 1
                    P2 += 1
                    P2_achieved = True
                    # Table 2
                    red_block_red_target += 1
                    red_pushed += 1
                    # break

            """
            # 2. Push (block is in the target at the final step)

            # Get the next step vector and check if the current step is the last one
            step_vector_next = data[rollout_i, step_i+1, :]
            if np.sum(step_vector_next) == 0 or step_i == len(data[rollout_i]) - 2:
                # Store the distances in a list
                distances = [distanceb1t1, distanceb1t2, distanceb2t1, distanceb2t2]
                print(len([x for x in distances if x < tolerance]))
                # If only one distance is less than the tolerance, update P1_2
                if len([x for x in distances if x < tolerance]) >= 1:
                    P1_2+=1
                # If two distances are less than the tolerance, update P2_2
                if len([x for x in distances if x < tolerance]) == 2:
                    P2_2+=1
                else:
                    print('Error in the computation of P2_2')
                break
            """

        # Compute the probabilities
        if rollout_i == len(data) - 1:
            PR1 = R1 / len(data)
            PR2 = R2 / len(data)
            PR1_2 = R1_2 / len(data)
            PR2_2 = R2_2 / len(data)
            PP1 = P1 / len(data)
            PP2 = P2 / len(data)
            PP1_2 = P1_2 / len(data)
            PP2_2 = P2_2 / len(data)
            Pfirst_reached_red = first_reached_red / len(data)
            Pfirst_reached_green = first_reached_green / len(data)
            Pred_block_red_target = red_block_red_target / len(data)
            Pred_block_green_target = red_block_green_target / len(data)
            Pgreen_block_red_target = green_block_red_target / len(data)
            Pgreen_block_green_target = green_block_green_target / len(data)

            # Store a dictionary with the probabilities
            probability_metrics = {
                "R1 (block-block distance)": PR1,
                "R2 (block-block distance)": PR2,
                "R1 (block-arm distance)": PR1_2,
                "R2 (block-arm distance)": PR2_2,
                "P1 (block enters the target)": PP1,
                "P2 (block enters the target) ": PP2,
                "P1 (block stays in target)": PP1_2,
                "P2 (block stays in target)": PP2_2,
                "First block reached Red": Pfirst_reached_red,
                "First block reached Green": Pfirst_reached_green,
                "Red block reached Red target": Pred_block_red_target,
                "Red block reached Green target": Pred_block_green_target,
                "Green block reached Red target": Pgreen_block_red_target,
                "Green block reached Green target": Pgreen_block_green_target,
            }

            # Store a dictionary with the absolute counts
            absolute_metrics = {
                "R1 (block-block distance)": R1,
                "R2 (block-block distance)": R2,
                "R1 (block-arm distance)": R1_2,
                "R2 (block-arm distance)": R2_2,
                "P1 (block enters the target)": P1,
                "P2 (block enters the target)": P2,
                "P1 (block stays in target)": P1_2,
                "P2 (block stays in target)": P2_2,
                "First block reached Red": first_reached_red,
                "First block reached Green": first_reached_green,
                "Red block reached Red target": red_block_red_target,
                "Red block reached Green target": red_block_green_target,
                "Green block reached Red target": green_block_red_target,
                "Green block reached Green target": green_block_green_target,
                "Red block pushed": red_pushed,
                "Green block pushed": green_pushed,
            }

            return probability_metrics, absolute_metrics


# KITCHEN COMPUTE METRICS

next_goal = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.88,
        -0.01,
        0,
        0,
        -0.92,
        -0.01,
        -0.69,
        -0.05,
        0.37,
        0,
        1.45,
        -0.75,
        -0.23,
        0.75,
        1.62,
        0.99,
        0,
        0,
        -0.06,
    ]
)

REMOVE_TASKS_WHEN_COMPLETE = True

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.5

tasks_to_complete = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]


def compute_kitchen_sequences(rollouts, treshhold=0.3):
    """Compute the tasks completed for the kitchen environment.
    Args:
        rollouts (np.ndarray): Array of shape (num_demos, rollout_length, dimensions)
            containing the observations for each rollout.
    Returns:
            tot_completions: List of lenght num_demos containing lists representing the achieved goals for each rollout chronologically.
            tot_mappings: List of lenght num_demos containing strings with the
            mapping of the tasks completed to their char initials for each rollout.
            tot_timesteps: List of lenght num_demos containing strings with the
            timesteps of the tasks completed for each rollout."""

    index_to_string = {0: "b", 1: "t", 2: "l", 3: "s", 4: "h", 5: "m", 6: "k"}

    tot_completions = []
    tot_mappings = []
    tot_timesteps = []
    for demon_i, demon in enumerate(rollouts):
        rollout_completion = []
        rollout_mapping = ""
        timestep_of_completion = ""
        for step_i, step in enumerate(rollouts[demon_i]):
            qp = rollouts[demon_i][step_i][:9]
            obj_qp = rollouts[demon_i][step_i][9:30]
            idx_offset = len(qp)
            for task in tasks_to_complete:
                element_idx = OBS_ELEMENT_INDICES[task]
                distance = np.linalg.norm(
                    obj_qp[element_idx - idx_offset] - next_goal[element_idx]
                )
                if distance < treshhold:
                    if task not in rollout_completion:
                        rollout_completion.append(task)
                        rollout_mapping += index_to_string[
                            tasks_to_complete.index(task)
                        ]
                        timestep_of_completion += str(step_i) + "_"
        if len(rollout_mapping) <= 7:
            rollout_mapping += "_" * (7 - len(rollout_mapping))
        tot_completions.append(rollout_completion)
        tot_timesteps.append(timestep_of_completion)
        tot_mappings.append(rollout_mapping)
    return tot_completions, tot_mappings, tot_timesteps


def compute_kitchen_metrics(rollout_elements):
    """Compute the metrics for the kitchen environment.
    Args:
        rollout_elements (list): List of lists containing the tasks completed for each rollout chronologically.
    Returns:
            Metrics for each of the tasks,"""

    task_completed_1 = 0
    task_completed_2 = 0
    task_completed_3 = 0
    task_completed_4 = 0
    task_completed_5 = 0
    task_completed_6 = 0
    task_completed_7 = 0

    for rollout in rollout_elements:
        if len(rollout) >= 1:
            task_completed_1 += 1
        if len(rollout) >= 2:
            task_completed_2 += 1
        if len(rollout) >= 3:
            task_completed_3 += 1
        if len(rollout) >= 4:
            task_completed_4 += 1
        if len(rollout) >= 5:
            task_completed_5 += 1
        if len(rollout) >= 6:
            task_completed_6 += 1
        if len(rollout) >= 7:
            task_completed_7 += 1

    P1 = task_completed_1 / len(rollout_elements)
    P2 = task_completed_2 / len(rollout_elements)
    P3 = task_completed_3 / len(rollout_elements)
    P4 = task_completed_4 / len(rollout_elements)
    P5 = task_completed_5 / len(rollout_elements)
    P6 = task_completed_6 / len(rollout_elements)
    P7 = task_completed_7 / len(rollout_elements)

    return {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6, "P7": P7}


def compute_task_entropy(task_mappings):
    """
    Compute the entropy of a list of task mappings.
    Args:
        task_mappings (list): list of task mappings
    Returns:
        count_seq (dict): dictionary with the number of times each sequence appears
        e (float): entropy
    """
    count_seq = {}
    for sequence in task_mappings:
        if sequence in count_seq:
            count_seq[sequence] += 1
        else:
            count_seq[sequence] = 1

    e = 0
    for key, value in count_seq.items():
        p = value / len(task_mappings)
        e += -p * np.log2(p)

    return count_seq, e


# PIPELINE


# @hydra.main(version_base="1.2", config_path="configs", config_name="config_compute_metrics")
def main(path):
    # Config should just take the folder of an evaluation job:
    # e.g. eval_runs/eval_blockpush/reproduction/ablation_focal_loss/0

    # SHould log in wandb the configs below
    # Should relog the `train_config:`
    # Should log the `eval_config:`
    # Should log tags in wandb about which job ran e.g.
    # can be gotten from   save_path: /opt/project/eval_runs/eval_blockpush/reproduction/ablation_focal_loss/0/snapshot_0
    # ablation_focal_loss -> tags: ablation, focal, loss (just a split)

    # Should compute the metrics
    # should log the metrics to wandb
    # Should save the logs in a file. (serialize a dictionary.)

    # iterate over a folder, for each subfolder, load the rollout and compute the entropy.
    # use pathlib to iterate over the subfolders

    # get the path from the config
    metrics = {}
    p = Path(path)

    for subfolder in p.glob("*"):
        # if there is a snapshot folder (has snaphot in the name) print the full name
        if "snapshot" in str(subfolder):
            print(subfolder)
            # load the observations (has obs in the name)
            for file in subfolder.glob("*"):
                if "obs" in str(file):
                    print(file)
                    # load the rollout
                    rollout = np.load(file, allow_pickle=True)
                    print(rollout.shape)
                    if "blockpush" in path:
                        prob_dict, count_dict = compute_blockpush_metrics(
                            rollout, tolerance=0.05
                        )
                        # store the metrics in a dictionary
                        metrics[str(subfolder)] = (prob_dict, count_dict)
                        with open("blockpush_metrics.json", "w") as f:
                            json.dump(metrics, f, indent=4)
                    elif "kitchen" in path:
                        elements, mappings, timesteps = compute_kitchen_sequences(
                            rollout, treshhold=0.3
                        )
                        # compute the entropy
                        count_seq, e = compute_task_entropy(mappings)

                        # compute table 1 metrics
                        table1_metrics = compute_kitchen_metrics(elements)
                        # store the metrics in a dictionary
                        metrics[str(subfolder)] = (e, table1_metrics)
                        with open("kitchen_metrics.json", "w") as f:
                            json.dump(metrics, f, indent=4)
                    else:
                        print("Environment not fount or not supported")
                        return

    return metrics


if __name__ == "__main__":
    main()
