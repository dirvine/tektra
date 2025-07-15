#!/usr/bin/env python3
"""
Demo script for Enhanced Markdown Renderer

This script demonstrates the enhanced markdown rendering capabilities
including syntax highlighting, copy-to-clipboard functionality,
table rendering, and improved visual styling.
"""

import asyncio
from src.tektra.gui.markdown_renderer import EnhancedMarkdownRenderer

def demo_enhanced_markdown():
    """Demonstrate the enhanced markdown renderer capabilities."""
    
    print("🎨 Enhanced Markdown Renderer Demo")
    print("=" * 50)
    
    # Create renderer instance
    renderer = EnhancedMarkdownRenderer()
    
    # Sample markdown content with all features
    sample_markdown = """# Enhanced Markdown Renderer Demo

This document demonstrates all the **enhanced features** of our markdown renderer.

## Code Blocks with Syntax Highlighting

Here's a Python example:

```python
def fibonacci(n):
    \"\"\"Calculate fibonacci number using recursion.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
```

And here's some JavaScript:

```javascript
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

console.log("5! =", factorial(5));
```

## Tables

| Language | Type | Paradigm |
|----------|------|----------|
| Python | Interpreted | Multi-paradigm |
| JavaScript | Interpreted | Multi-paradigm |
| Rust | Compiled | Systems |
| Go | Compiled | Concurrent |

## Lists

### Unordered List
- **Enhanced styling** with better spacing
- *Italic text* support
- `Inline code` formatting
- [Links](https://example.com) with hover effects

### Ordered List
1. First item with **bold** text
2. Second item with *italic* text
3. Third item with `inline code`
4. Fourth item with [a link](https://github.com)

## Blockquotes

> "The best way to predict the future is to invent it."
> - Alan Kay

> This is a multi-line quote that demonstrates
> how our enhanced renderer handles longer
> quoted content with proper formatting.

## Inline Formatting

This paragraph contains **bold text**, *italic text*, `inline code`, 
and [links](https://example.com) to demonstrate all inline formatting options.

## Final Notes

The enhanced markdown renderer provides:
- ✅ Syntax highlighting for code blocks
- ✅ Copy-to-clipboard functionality
- ✅ Improved table rendering
- ✅ Enhanced visual styling
- ✅ Better typography hierarchy
- ✅ Accessibility support
"""

    print("\n📝 Sample Markdown Content:")
    print("-" * 30)
    print(sample_markdown[:200] + "..." if len(sample_markdown) > 200 else sample_markdown)
    
    print("\n🔧 Parsing markdown into blocks...")
    blocks = renderer._parse_blocks(sample_markdown)
    
    print(f"✅ Parsed {len(blocks)} blocks:")
    for i, block in enumerate(blocks, 1):
        block_type = block['type']
        content_preview = str(block['content'])[:50].replace('\n', ' ')
        print(f"  {i:2d}. {block_type:12s} - {content_preview}...")
    
    print("\n🎨 Rendering markdown to widgets...")
    try:
        # Note: This would normally create Toga widgets, but we can't run them in a script
        # So we'll just demonstrate the parsing and some of the processing
        
        # Test syntax highlighting
        if renderer._apply_syntax_highlighting:
            python_code = 'def hello():\n    print("Hello, World!")'
            highlighted = renderer._apply_syntax_highlighting(python_code, 'python')
            print(f"\n🔍 Syntax highlighting demo:")
            print(f"Original: {python_code}")
            print(f"Enhanced: {highlighted}")
        
        # Test inline formatting
        inline_text = "This has **bold**, *italic*, `code`, and [link](url) text."
        processed = renderer._process_inline_formatting(inline_text)
        print(f"\n✨ Inline formatting demo:")
        print(f"Original: {inline_text}")
        print(f"Enhanced: {processed}")
        
        print("\n✅ Enhanced Markdown Renderer Demo Complete!")
        print("\nFeatures demonstrated:")
        print("  • Syntax highlighting with Pygments")
        print("  • Enhanced visual styling")
        print("  • Table parsing and rendering")
        print("  • Improved list formatting")
        print("  • Better inline text processing")
        print("  • Copy-to-clipboard functionality (in GUI)")
        print("  • Animation integration support")
        
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        return False
    
    return True

async def demo_copy_functionality():
    """Demonstrate the copy-to-clipboard functionality."""
    print("\n📋 Copy-to-Clipboard Demo")
    print("-" * 30)
    
    renderer = EnhancedMarkdownRenderer()
    
    # Test clipboard functionality
    test_code = """def example():
    return "Hello, World!\""""
    
    print(f"Testing clipboard copy with code:")
    print(test_code)
    
    try:
        success = await renderer._copy_to_clipboard(test_code)
        if success:
            print("✅ Code copied to clipboard successfully!")
        else:
            print("❌ Clipboard copy failed (this is expected in some environments)")
    except Exception as e:
        print(f"❌ Clipboard error: {e}")

def main():
    """Main demo function."""
    print("🚀 Starting Enhanced Markdown Renderer Demo\n")
    
    # Run the main demo
    success = demo_enhanced_markdown()
    
    if success:
        print("\n🔄 Running async clipboard demo...")
        try:
            asyncio.run(demo_copy_functionality())
        except Exception as e:
            print(f"❌ Async demo error: {e}")
    
    print("\n🎯 Demo completed!")

if __name__ == "__main__":
    main()