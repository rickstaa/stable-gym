"""Functions that are used in multiple simzoo environments.
"""

import re

import numpy as np
from gym.utils import colorize as gym_colorize


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    .. seealso::
        This function wraps the :meth:`gym.utils.colorize` function to make sure that it
        also works with empty empty color strings.

    Args:
        string (str): The string you want to colorize.
        color (str): The color you want to use.
        bold (bool, optional): Whether you want the text to be bold text has to be bold.
        highlight (bool, optional):  Whether you want to highlight the text. Defaults to
            ``False``.

    Returns:
        str: Colorized string.
    """
    if color:  # If not empty
        return gym_colorize(string, color, bold, highlight)
    else:
        return string


def get_flattened_values(input_obj):
    """Retrieves all the values that are present in a nested dictionary and appends them
    to a list. Its like a recursive version of the :meth:`dict.values()` method.

    Args:
        input_obj (dict): The input dictionary from which you want
            to retrieve all the values.

    Returns:
        list: A list containing all the values that were present in the nested
            dictionary.
    """
    flat_values = []
    if isinstance(input_obj, dict):
        for item in input_obj.values():
            if isinstance(item, dict):
                for it in item.values():
                    flat_values.extend(get_flattened_values(it))
            else:
                flat_values.append(item)
    else:
        flat_values.append(input_obj)
    return flat_values


def get_flattened_keys(input_obj, include_root=False):
    """Retrieves all the keys that are present in a nested dictionary and appends them
    to a list. Its like a recursive version of the :meth:`dict.keys()` method.

    Args:
        input_obj (dict): The input dictionary from which you want
            to retrieve all the keys.
        include_root (bool): Whether you want to include the root level keys. Defaults
            to ``False``.

    Returns:
        list: A list containing all the keys that were present in the nested
            dictionary.
    """
    flat_keys = []
    if isinstance(input_obj, dict):
        if include_root:
            flat_keys.extend(input_obj.keys())
        for key, val in input_obj.items():
            if isinstance(val, dict):
                flat_keys.extend(get_flattened_keys(val))
            else:
                flat_keys.append(key)
    else:
        flat_keys.append(input_obj)
    return flat_keys


def abbreviate(input_item, length=1, max_length=4, capitalize=True):
    """Creates unique abbreviations for a string or list of strings.

    Args:
        input_item (union[str, list]): The string of list of strings which you want to
            abbreviate.
        length (int, optional): The desired length of the abbreviation. Defaults to
            ``1``.
        max_length (int, optional): The maximum length of the abbreviation. Defaults to
            4.
        capitalize (bool, optional): Whether the abbrevaitions should be capitalized.
            Defaults to True.

    Returns:
        list: List with abbreviations.
    """
    if isinstance(input_item, list):
        items = []
        abbreviations = []
        for it in input_item:
            unique = False
            length_tmp = length
            suffix = ""
            while not unique:
                abbreviation = (
                    it[:length_tmp].capitalize() + str(suffix)
                    if capitalize
                    else it[:length_tmp] + suffix
                )
                if abbreviation not in abbreviations:  # Check if unique
                    abbreviations.append(abbreviation)
                    items.append(it)
                    unique = True
                else:
                    prev_item = items[abbreviations.index(abbreviation)]
                    if it == prev_item:  # Allow if item was equal
                        abbreviations.append(abbreviation)
                        items.append(it)
                        unique = True
                    else:  # Use longer abbreviation otherwise
                        if length_tmp < max_length:
                            length_tmp += 1
                        else:
                            suffix = get_lowest_next_int(abbreviations)
        return abbreviations
    else:
        return input_item[:length].capitalize() if capitalize else input_item[:length]


def get_lowest_next_int(input_item):
    """Retrieves the lowest next integer that is not present in a string or float list.

    Args:
        input_item (union[int, str, list]): The input for which you want to determine
            the next lowest interger.

    Returns:
        int: The next lowest integer.
    """
    if isinstance(input_item, list):
        input_ints = [
            (
                round(float(re.sub("[^0-9.]", "", item)))
                if (re.sub("[^0-9.]", "", item) != "")
                else ""
            )
            if isinstance(item, str)
            else item
            for item in input_item
        ]  # Trim all non-numeric chars
        input_ints = [item for item in input_ints if item != ""]
    else:
        input_ints = [
            (
                (
                    round(float(re.sub("[^0-9.]", "", input_item)))
                    if (re.sub("[^0-9.]", "", input_item) != "")
                    else ""
                )
                if isinstance(input_item, str)
                else input_item
            )
        ]
    input_ints = input_ints if input_ints else [0]
    return list(set(input_ints) ^ set(range(min(input_ints), max(input_ints) + 2)))[0]


def friendly_list(input_list, apostrophes=False):
    """Transforms a list to a human friendly format (seperated by commas and ampersand).

    Args:
        input_list (list): The input list.
        apostrophes(bool, optional): Whether the list items should be encapsuled with
            apostrophes. Defaults to ``False``.

    Returns:
        str: Human friendly list string.
    """
    input_list = (
        ["'" + item + "'" for item in input_list] if apostrophes else input_list
    )
    return " & ".join(", ".join(input_list).rsplit(", ", 1))


def strip_underscores(text, position="all"):
    """Strips leading and/or trailing underscores from a string.

    Args:
        text (str): The input string.
        position (str, optional): From which position underscores should be removed.
            Options are 'leading', 'trailing' & 'both'. Defaults to "both".

    Returns:
        str: String without the underscores.
    """
    if position.lower() == "leading":
        while text.startswith("_"):
            text = text[1:]
    elif position.lower() == "trailing":
        while text.endswith("_"):
            text = text[:-1]
    else:
        text = text.strip("_")
    return text


def inject_value(input_item, value, round_accuracy=2, order=False, axis=0):
    """Injects a value into a list or dictionary if it is not yet present.

    Args:
        input_item (union[list,dict]): The input list or dictionary.
        value (float): The value you want to inject.
        round_accuracy (int, optional): The accuracy used for checking whether a value
            is present. Defaults to 2.
        order (bool, optional): Whether the list should be ordered when returned.
            Defaults to ``false``.
        axis (int, optional): The axis along which you want to inject the value. Only
            used when the input is a numpy array. Defaults to ``0``.

    Returns:
        union[list,dict]: The list or dictionary that contains the value.
    """
    order_op = (
        lambda *args, **kwargs: sorted(*args, **kwargs)
        if order
        else list(*args, **kwargs)
    )
    if isinstance(input_item, dict):
        return {
            k: inject_value(
                v, value=value, round_accuracy=round_accuracy, order=order, axis=axis
            )
            for k, v in input_item.items()
        }
    elif isinstance(input_item, np.ndarray) and input_item.ndim > 1:
        transpose_matrix = np.eye(input_item.ndim, dtype=np.int16)
        return np.transpose(
            np.array(
                [
                    order_op([value] + [it for it in item if it != value])
                    for item in np.transpose(input_item, transpose_matrix[axis])
                ]
            ),
            transpose_matrix[axis],
        )
    else:
        return order_op([value] + [item for item in input_item if item != value])
