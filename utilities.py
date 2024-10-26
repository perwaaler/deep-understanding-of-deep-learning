RESET_COLOR = "\033[0m"

COLOR_TO_ANSI_MAP = {  # All are in bold font
    "grey": "\033[1;30m",
    "grey_bold": "\033[1;30m",
    "green": "\033[1;92m",
    "blue": "\033[1;94m",
    "orange": "\033[1;33m",
    "red": "\033[1;91m",
    "magenta": "\033[1;95m",  # Added magenta
    "teal": "\033[1;96m",  # Added teal (cyan)
    "brown": "\033[38;5;94m",  # Changed brown to a darker yellow/brown
    "pink": "\033[1;35m",  # Added pink (magenta with bold)
    "lime": "\033[1;32m",  # Added lime (green with bold)
    "reset": "\033[0m",
}


# %% PRINTING UTILITES


def color_text(*args, c="grey"):
    """Returns text in the desired colour (grey, green, or blue)."""
    colored_strings = []
    for arg in args:
        colored_strings.append(f"{COLOR_TO_ANSI_MAP[c]}{arg}{RESET_COLOR}")
    return colored_strings


def red_text(*args):
    """Returns text in red."""
    return "\n".join([color_text(str(text), "red") for text in args])


def color_print(*args, c="green"):
    """Prints text in the specified color."""
    print(color_text(*args, c=c))


def silent_print(*args):
    """Prints text in console in dark grey."""
    print(*color_text(*args, c="grey"))


def orange_print(*args):
    """Prints each argument on a new line in orange."""
    print(*color_text(*args, c="orange"))


def green_print(*args):
    """Prints each argument on a new line in green."""
    print(*color_text(*args, c="green"))


def blue_print(*args):
    """Prints each argument on a new line in blue."""
    print(*color_text(*args, c="blue"))


def teal_print(*args):
    """Prints each argument on a new line in blue."""
    print(*color_text(*args, c="teal"))


def magenta_print(*args):
    """Prints each argument on a new line in magenta color."""
    print(*color_text(*args, c="magenta"))


def red_print(*args):
    """Prints each argument on a new line in red."""
    print(*color_text(*args, c="red"))


def lime_print(*args):
    """Prints each argument on a new line in red."""
    print(*color_text(*args, c="lime"))


def pink_print(*args):
    """Prints each argument on a new line in red."""
    print(*color_text(*args, c="pink"))


def calc_acc(*model_inputs, labels, model):
    """Calculates Accuracy for a torch model"""
    predictions = (model(*model_inputs) >= 0.5).float()
    return (predictions == labels).float().mean()


