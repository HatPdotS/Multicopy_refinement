#!/usr/bin/env python3
"""Test the split function"""

def split_respecting_quotes_old(line):
    line_new = ''
    in_quotes = False
    for character in line:
        if character == "'" or character == '"':
            in_quotes = not in_quotes
        if in_quotes and character == ' ':
            continue
        line_new += character
    return line_new.split()

def split_respecting_quotes_new(line):
    """
    Split a line by whitespace, but preserve quoted strings intact.
    Handles both single and double quotes.
    """
    line_new = ''
    in_quotes = False
    quote_char = None
    for character in line:
        if (character == "'" or character == '"') and not in_quotes:
            # Starting a quoted section
            in_quotes = True
            quote_char = character
        elif character == quote_char and in_quotes:
            # Ending a quoted section
            in_quotes = False
            quote_char = None
        elif in_quotes and character == ' ':
            # Skip spaces inside quotes
            continue
        line_new += character
    return line_new.split()

test_line = 'GTP     GTP      "GUANOSINE-5\'-TRIPHOSPHATE"     NON-POLYMER     44     32     .'

print("Original line:")
print(repr(test_line))
print("\nOld function result:")
result_old = split_respecting_quotes_old(test_line)
print(f"{len(result_old)} items: {result_old}")

print("\nNew function result:")
result_new = split_respecting_quotes_new(test_line)
print(f"{len(result_new)} items: {result_new}")

print("\nExpected: 7 items")
