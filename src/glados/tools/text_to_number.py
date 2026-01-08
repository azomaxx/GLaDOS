"""Text-to-number conversion for spoken number words."""

import re
from typing import Dict, Union

# Mapping of number words to their numeric values
NUMBER_WORDS: Dict[str, Union[int, float]] = {
    # Basic numbers
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    
    # Teens
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    
    # Tens
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    
    # Large numbers
    "hundred": 100, "thousand": 1000, "million": 1000000,
    
    # Fractions
    "half": 0.5, "quarter": 0.25, "third": 0.333, "fourth": 0.25,
}

def text_to_number(text: str) -> Union[int, float, None]:
    """
    Convert spoken number text to numeric value.
    
    Args:
        text: Text containing number words
        
    Returns:
        Numeric value or None if no numbers found
    """
    if not text:
        return None
    
    # Convert to lowercase and extract number-related words
    text_lower = text.lower()
    
    # Simple direct mappings first
    if text_lower in NUMBER_WORDS:
        return NUMBER_WORDS[text_lower]
    
    # Extract digits directly if present
    digit_match = re.search(r'\d+(\.\d+)?', text)
    if digit_match:
        number_str = digit_match.group()
        return float(number_str) if '.' in number_str else int(number_str)
    
    # Handle compound numbers like "twenty one", "thirty five"
    words = text_lower.split()
    total = 0
    current = 0
    
    for word in words:
        if word in NUMBER_WORDS:
            value = NUMBER_WORDS[word]
            
            if value >= 100:
                # For hundred, thousand, etc.
                if current == 0:
                    current = 1
                current *= value
            elif value >= 20:
                # For tens like twenty, thirty, etc.
                current += value
            else:
                # For single digits
                current += value
        elif word == "and":
            continue
        else:
            # Non-number word, finalize current and reset
            total += current
            current = 0
    
    total += current
    return total if total > 0 else None

# Test cases
if __name__ == "__main__":
    test_cases = [
        "thirteen", "13", "twenty one", "thirty five", 
        "volume down by 13 percent", "set volume to fifty",
        "decrease by ten percent", "increase by twenty five"
    ]
    
    for case in test_cases:
        result = text_to_number(case)
        print(f"'{case}' -> {result}")
