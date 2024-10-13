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
    "brown": "\033[1;33m",  # Added brown (same as orange, closest approximation)
    "reset": "\033[0m",
}

# %% PRINTING UTILITES


def color_text(text: str, color="grey"):
    """Returns text in the desired colour (grey, green, or blue)."""
    return f"{COLOR_TO_ANSI_MAP[color]}{text}{RESET_COLOR}"
def red_text(*args):
    """Returns text in red."""
    return "\n".join([color_text(str(text), "red") for text in args])

def color_print(*args, color="green"):
    """Prints text in the specified color."""
    for text in args:
        print(color_text(str(text), color))

def silent_print(*args):
    """Prints text in console in dark grey."""
    for arg in args:
        print(color_text(str(arg), "grey"))

def orange_print(*args):
    """Prints each argument on a new line in orange."""
    for text in args:
        print(color_text(str(text), "orange"))

def green_print(*args):
    """Prints each argument on a new line in green."""
    for text in args:
        print(color_text(str(text), "green"))

def blue_print(*args):
    """Prints each argument on a new line in blue."""
    for text in args:
        print(color_text(str(text), "blue"))

def magenta_print(*args):
    """Prints each argument on a new line in magenta color."""
    for text in args:
        print(color_text(str(text), "magenta"))

def red_print(*args):
    """Prints each argument on a new line in red."""
    for text in args:
        print(color_text(str(text), "red"))
