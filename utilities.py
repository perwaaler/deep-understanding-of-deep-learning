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


def red_text(text: str):
    """Returns text in red."""
    return color_text(text, "red")


def color_print(text: str, color="green"):
    """Prints text in the specified color."""
    print(color_text(text, color))


def silent_print(text: str):
    """Prints text in console in dark grey."""
    print(color_text(text, "grey"))


def orange_print(text: str):
    """Prints text (warning) in console in a color close to orange (bold
    yellow)."""
    print(color_text(text, "orange"))


def green_print(text: str):
    """Prints text (warning) in console in a color close to orange (bold
    yellow)."""
    print(color_text(text, "green"))


def blue_print(text: str):
    """Prints text (warning) in console in a color close to orange (bold
    yellow)."""
    print(color_text(text, "blue"))


def magenta_print(text: str):
    """Prints text (warning) in console in a color close to orange (bold
    yellow)."""
    print(color_text(text, "magenta"))


def red_print(text):
    """Prints the ValueError message in red."""
    print(color_text(text, "red"))
