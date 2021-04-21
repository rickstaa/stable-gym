"""Functions that are used in multiple simzoo environments.
"""
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    This function was originally written by John Schulman.

    Args:
        string (str): The string you want to colorize.
        color (str): The color you want to use.
        bold (bool, optional): Whether you want the text to be bold text has to be bold.
        highlight (bool, optional):  Whether you want to highlight the text. Defaults to
            False.

    Returns:
        str: Colorized string.
    """
    if color:
        attr = []
        num = color2num[color]
        if highlight:
            num += 10
        attr.append(str(num))
        if bold:
            attr.append("1")
        return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)
    else:
        return string
