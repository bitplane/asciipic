ASCII_PRINTABLE = "".join(chr(i) for i in range(32, 127))

ASCII_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:;!?@#$%&*+-=/<>()[]{}|\\\"'`~^_"

# Braille patterns: U+2800 to U+28FF (256 characters, 2x4 dot grid)
BRAILLE = "".join(chr(i) for i in range(0x2800, 0x2900))

# Block elements: fills, halves, quadrants, shades
BLOCKS = " ░▒▓█▀▄▌▐▖▗▘▙▚▛▜▝▞▟"

# Box drawing and line characters
LINES = "─│┌┐└┘├┤┬┴┼╱╲╳"

# ASCII characters useful for texture and edges
TEXTURE_ASCII = " .,:;!'-/\\xX*+=#@"

# Combined visual charset for neural colour rendering (deduplicated, order-preserving)
VISUAL = "".join(dict.fromkeys(BLOCKS + BRAILLE + LINES + TEXTURE_ASCII))
