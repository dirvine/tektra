#\!/usr/bin/env python3
"""Generate a simple test image for multimodal testing."""

from PIL import Image, ImageDraw, ImageFont
import os

# Create a simple test image
width, height = 800, 600
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Draw some shapes
draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=3)
draw.ellipse([250, 50, 400, 200], fill='blue', outline='black', width=3)
draw.polygon([(500, 50), (600, 200), (450, 200)], fill='green', outline='black', width=3)

# Add text
try:
    # Try to use a default font
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
except:
    # Fall back to default font
    font = ImageFont.load_default()

draw.text((50, 300), "Test Image for Multimodal AI", fill='black', font=font)
draw.text((50, 400), "Contains: Rectangle, Circle, Triangle", fill='gray', font=font)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), "test_image.png")
image.save(output_path)
print(f"Test image saved to: {output_path}")
