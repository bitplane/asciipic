ASCII_PRINTABLE = "".join(chr(i) for i in range(32, 127))

ASCII_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:;!?@#$%&*+-=/<>()[]{}|\\\"'`~^_"

# Braille patterns: U+2800 to U+28FF (256 characters, 2x4 dot grid)
BRAILLE = "".join(chr(i) for i in range(0x2800, 0x2900))

# Block elements: U+2580-U+259F (fills, eighths, halves, quadrants, shades)
BLOCKS = " " + "".join(chr(i) for i in range(0x2580, 0x25A0))

# Box drawing and line characters
LINES = "─│┌┐└┘├┤┬┴┼╱╲╳"

# Symbols for Legacy Computing: diagonal fills, sextants, wedges (U+1FB00-U+1FB3B)
LEGACY_COMPUTING = "".join(chr(i) for i in range(0x1FB00, 0x1FB3C))

# ASCII characters useful for texture and edges
TEXTURE_ASCII = " .,:;!'-/\\xX*+=#@"

# Combined visual charset for neural colour rendering (deduplicated, order-preserving)
VISUAL = "".join(dict.fromkeys(BLOCKS + BRAILLE + LINES + LEGACY_COMPUTING + TEXTURE_ASCII))
