"""
Tests for Enhanced Markdown Renderer

This module tests the enhanced markdown rendering capabilities including
syntax highlighting, copy-to-clipboard functionality, table rendering,
and improved visual styling.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

import toga
from toga.style import Pack

from src.tektra.gui.markdown_renderer import (
    EnhancedMarkdownRenderer,
    get_markdown_renderer,
    get_enhanced_markdown_renderer,
    PYGMENTS_AVAILABLE
)


class TestEnhancedMarkdownRenderer:
    """Test cases for the EnhancedMarkdownRenderer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = EnhancedMarkdownRenderer()
        self.mock_animation_manager = Mock()
        self.renderer_with_animations = EnhancedMarkdownRenderer(self.mock_animation_manager)
    
    def test_initialization(self):
        """Test renderer initialization."""
        assert self.renderer is not None
        assert self.renderer.animation_manager is None
        assert self.renderer.syntax_theme == "default"
        assert self.renderer.copy_feedback_duration == 2.0
        assert isinstance(self.renderer.interactive_elements, dict)
        assert len(self.renderer.interactive_elements) == 0
    
    def test_initialization_with_animation_manager(self):
        """Test renderer initialization with animation manager."""
        assert self.renderer_with_animations.animation_manager is not None
        assert self.renderer_with_animations.animation_manager == self.mock_animation_manager
    
    def test_enhanced_styles_creation(self):
        """Test that enhanced styles are properly created."""
        styles = self.renderer.styles
        
        # Test header styles
        for i in range(1, 7):
            assert f"h{i}" in styles
            assert styles[f"h{i}"].font_weight == "bold"
        
        # Test code block styles
        assert "code_block_container" in styles
        assert "code_block_header" in styles
        assert "code_block_content" in styles
        assert "code_language_label" in styles
        assert "copy_button" in styles
        assert "copy_button_success" in styles
        
        # Test table styles
        assert "table_container" in styles
        assert "table_header_row" in styles
        assert "table_row" in styles
        assert "table_header_cell" in styles
        assert "table_cell" in styles
        
        # Test other styles
        assert "inline_code" in styles
        assert "paragraph" in styles
        assert "list_container" in styles
        assert "quote_container" in styles
    
    def test_parse_blocks_headers(self):
        """Test parsing of header blocks."""
        markdown = """# Header 1
## Header 2
### Header 3
#### Header 4
##### Header 5
###### Header 6"""
        
        blocks = self.renderer._parse_blocks(markdown)
        
        assert len(blocks) == 6
        for i, block in enumerate(blocks, 1):
            assert block['type'] == f'h{i}'
            assert block['content'] == f'Header {i}'
    
    def test_parse_blocks_code_blocks(self):
        """Test parsing of code blocks."""
        markdown = """```python
def hello():
    print("Hello, World!")
```

```javascript
console.log("Hello, World!");
```

```
plain text code
```"""
        
        blocks = self.renderer._parse_blocks(markdown)
        
        assert len(blocks) == 3
        
        # Python code block
        assert blocks[0]['type'] == 'code_block'
        assert blocks[0]['language'] == 'python'
        assert 'def hello():' in blocks[0]['content']
        
        # JavaScript code block
        assert blocks[1]['type'] == 'code_block'
        assert blocks[1]['language'] == 'javascript'
        assert 'console.log' in blocks[1]['content']
        
        # Plain text code block
        assert blocks[2]['type'] == 'code_block'
        assert blocks[2]['language'] == 'text'
        assert blocks[2]['content'] == 'plain text code'
    
    def test_parse_blocks_tables(self):
        """Test parsing of table blocks."""
        markdown = """| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |
| Jane | 30  | LA   |"""
        
        blocks = self.renderer._parse_blocks(markdown)
        
        assert len(blocks) == 1
        assert blocks[0]['type'] == 'table'
        assert len(blocks[0]['content']) == 4  # Header + separator + 2 data rows
    
    def test_parse_blocks_lists(self):
        """Test parsing of list blocks."""
        markdown = """- Item 1
- Item 2
- Item 3

1. Numbered item 1
2. Numbered item 2"""
        
        blocks = self.renderer._parse_blocks(markdown)
        
        assert len(blocks) == 2
        
        # Unordered list
        assert blocks[0]['type'] == 'list'
        assert len(blocks[0]['content']) == 3
        assert blocks[0]['content'][0] == 'Item 1'
        
        # Ordered list
        assert blocks[1]['type'] == 'list'
        assert len(blocks[1]['content']) == 2
        assert blocks[1]['content'][0] == 'Numbered item 1'
    
    def test_parse_blocks_quotes(self):
        """Test parsing of quote blocks."""
        markdown = """> This is a quote
> with multiple lines
> of text"""
        
        blocks = self.renderer._parse_blocks(markdown)
        
        assert len(blocks) == 1
        assert blocks[0]['type'] == 'quote'
        assert 'This is a quote' in blocks[0]['content']
        assert 'multiple lines' in blocks[0]['content']
    
    def test_parse_blocks_paragraphs(self):
        """Test parsing of paragraph blocks."""
        markdown = """This is a paragraph
with multiple lines.

This is another paragraph."""
        
        blocks = self.renderer._parse_blocks(markdown)
        
        assert len(blocks) == 2
        assert blocks[0]['type'] == 'paragraph'
        assert blocks[1]['type'] == 'paragraph'
        assert 'multiple lines' in blocks[0]['content']
        assert blocks[1]['content'] == 'This is another paragraph.'
    
    def test_render_code_block_without_language(self):
        """Test rendering code block without language specification."""
        code = "print('Hello, World!')"
        widget = self.renderer._render_code_block(code, 'text')
        
        assert isinstance(widget, toga.Box)
        # Should have only code content, no header
        assert len(widget.children) == 1
    
    def test_render_code_block_with_language(self):
        """Test rendering code block with language specification."""
        code = "print('Hello, World!')"
        widget = self.renderer._render_code_block(code, 'python')
        
        assert isinstance(widget, toga.Box)
        # Should have header and code content
        assert len(widget.children) == 2
    
    def test_render_header(self):
        """Test rendering headers."""
        for i in range(1, 7):
            header_text = f"Header {i}"
            widget = self.renderer._render_header(header_text, f'h{i}')
            
            assert isinstance(widget, toga.Label)
            assert header_text in widget.text
    
    def test_render_paragraph(self):
        """Test rendering paragraphs."""
        text = "This is a paragraph with **bold** and *italic* text."
        widget = self.renderer._render_paragraph(text)
        
        assert isinstance(widget, toga.Label)
        assert widget.text is not None
    
    def test_render_list(self):
        """Test rendering lists."""
        items = ["Item 1", "Item 2", "Item 3"]
        widget = self.renderer._render_list(items)
        
        assert isinstance(widget, toga.Box)
        assert len(widget.children) == 3
        
        for i, child in enumerate(widget.children):
            assert isinstance(child, toga.Label)
            assert f"Item {i+1}" in child.text
    
    def test_render_quote(self):
        """Test rendering quotes."""
        quote_text = "This is a quote"
        widget = self.renderer._render_quote(quote_text)
        
        assert isinstance(widget, toga.Box)
        assert len(widget.children) == 1
        assert isinstance(widget.children[0], toga.Label)
        assert quote_text in widget.children[0].text
    
    def test_render_table(self):
        """Test rendering tables."""
        table_rows = [
            "| Name | Age |",
            "|------|-----|",
            "| John | 25  |",
            "| Jane | 30  |"
        ]
        widget = self.renderer._render_table(table_rows)
        
        assert isinstance(widget, toga.Box)
        # Should have header row + 2 data rows
        assert len(widget.children) == 3
    
    def test_render_table_empty(self):
        """Test rendering empty table."""
        widget = self.renderer._render_table([])
        
        assert isinstance(widget, toga.Box)
        assert len(widget.children) == 0
    
    def test_process_inline_formatting(self):
        """Test processing of inline formatting."""
        text = "This has **bold**, *italic*, `code`, and [link](url) text."
        processed = self.renderer._process_inline_formatting(text)
        
        # Should contain Unicode formatting indicators
        assert '𝗕' in processed  # Bold indicator
        assert '𝘐' in processed  # Italic indicator
        assert '⟨' in processed and '⟩' in processed  # Code brackets
        assert '🔗' in processed  # Link indicator
    
    @pytest.mark.skipif(not PYGMENTS_AVAILABLE, reason="Pygments not available")
    def test_apply_syntax_highlighting_python(self):
        """Test syntax highlighting for Python code."""
        code = "def hello():\n    print('Hello, World!')"
        highlighted = self.renderer._apply_syntax_highlighting(code, 'python')
        
        # Should contain visual indicators for different token types
        assert '🔵' in highlighted or code in highlighted  # Keywords or fallback
    
    def test_apply_syntax_highlighting_no_language(self):
        """Test syntax highlighting without language."""
        code = "some code"
        highlighted = self.renderer._apply_syntax_highlighting(code, '')
        
        # Should return original code
        assert highlighted == code
    
    def test_apply_syntax_highlighting_unknown_language(self):
        """Test syntax highlighting with unknown language."""
        code = "some code"
        highlighted = self.renderer._apply_syntax_highlighting(code, 'unknown_lang')
        
        # Should return original code
        assert highlighted == code
    
    @pytest.mark.asyncio
    async def test_copy_to_clipboard_success(self):
        """Test successful clipboard copy."""
        with patch('platform.system', return_value='Darwin'):
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'', b'')
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process
                
                result = await self.renderer._copy_to_clipboard("test text")
                assert result is True
    
    @pytest.mark.asyncio
    async def test_copy_to_clipboard_failure(self):
        """Test failed clipboard copy."""
        with patch('platform.system', return_value='Darwin'):
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b'', b'error')
                mock_process.returncode = 1
                mock_subprocess.return_value = mock_process
                
                result = await self.renderer._copy_to_clipboard("test text")
                assert result is False
    
    @pytest.mark.asyncio
    async def test_handle_copy_code_success(self):
        """Test handling successful code copy."""
        mock_button = Mock(spec=toga.Button)
        mock_button.text = "📋 Copy"
        mock_button.style = self.renderer.styles["copy_button"]
        
        with patch.object(self.renderer, '_copy_to_clipboard', return_value=True):
            await self.renderer._handle_copy_code(mock_button, "test code")
            
            # Button text should be reset after the operation
            assert mock_button.text == "📋 Copy"
    
    @pytest.mark.asyncio
    async def test_handle_copy_code_failure(self):
        """Test handling failed code copy."""
        mock_button = Mock(spec=toga.Button)
        mock_button.text = "📋 Copy"
        
        with patch.object(self.renderer, '_copy_to_clipboard', return_value=False):
            await self.renderer._handle_copy_code(mock_button, "test code")
            
            # Button text should be reset after showing error
            assert mock_button.text == "📋 Copy"
    
    def test_render_markdown_complete(self):
        """Test rendering complete markdown document."""
        markdown = """# Main Title

This is a paragraph with **bold** and *italic* text.

## Code Example

```python
def hello():
    print("Hello, World!")
```

## List Example

- Item 1
- Item 2
- Item 3

## Table Example

| Name | Age |
|------|-----|
| John | 25  |

> This is a quote

Another paragraph at the end."""
        
        widget = self.renderer.render_markdown(markdown)
        
        assert isinstance(widget, toga.Box)
        # Should have multiple child widgets for different blocks
        assert len(widget.children) > 5
    
    def test_render_simple_message_with_markdown(self):
        """Test rendering simple message with markdown content."""
        message = "Here's some `code` and **bold** text."
        widget = self.renderer.render_simple_message(message)
        
        assert isinstance(widget, toga.Box)
        assert len(widget.children) >= 1
    
    def test_render_simple_message_plain_text(self):
        """Test rendering simple message with plain text."""
        message = "This is just plain text."
        widget = self.renderer.render_simple_message(message)
        
        assert isinstance(widget, toga.Box)
        assert len(widget.children) == 1
        assert isinstance(widget.children[0], toga.Label)
    
    def test_clear_interactive_elements(self):
        """Test clearing interactive elements."""
        # Add some mock elements
        self.renderer.interactive_elements["test1"] = "element1"
        self.renderer.interactive_elements["test2"] = "element2"
        
        assert len(self.renderer.interactive_elements) == 2
        
        self.renderer.clear_interactive_elements()
        
        assert len(self.renderer.interactive_elements) == 0
    
    def test_get_interactive_element_count(self):
        """Test getting interactive element count."""
        assert self.renderer.get_interactive_element_count() == 0
        
        self.renderer.interactive_elements["test"] = "element"
        assert self.renderer.get_interactive_element_count() == 1


class TestMarkdownRendererGlobalFunctions:
    """Test cases for global markdown renderer functions."""
    
    def test_get_markdown_renderer(self):
        """Test getting global markdown renderer."""
        renderer = get_markdown_renderer()
        
        assert isinstance(renderer, EnhancedMarkdownRenderer)
        assert renderer.animation_manager is None
        
        # Should return same instance on subsequent calls
        renderer2 = get_markdown_renderer()
        assert renderer is renderer2
    
    def test_get_markdown_renderer_with_animation_manager(self):
        """Test getting global markdown renderer with animation manager."""
        mock_animation_manager = Mock()
        renderer = get_markdown_renderer(mock_animation_manager)
        
        assert isinstance(renderer, EnhancedMarkdownRenderer)
        # Note: The global renderer is created once, so animation manager
        # won't be set if renderer already exists
    
    def test_get_enhanced_markdown_renderer(self):
        """Test getting enhanced markdown renderer."""
        renderer = get_enhanced_markdown_renderer()
        
        assert isinstance(renderer, EnhancedMarkdownRenderer)
        assert renderer.animation_manager is None
    
    def test_get_enhanced_markdown_renderer_with_animation_manager(self):
        """Test getting enhanced markdown renderer with animation manager."""
        mock_animation_manager = Mock()
        renderer = get_enhanced_markdown_renderer(mock_animation_manager)
        
        assert isinstance(renderer, EnhancedMarkdownRenderer)
        assert renderer.animation_manager == mock_animation_manager


class TestMarkdownRendererIntegration:
    """Integration tests for markdown renderer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_animation_manager = Mock()
        self.mock_animation_manager.micro_interaction_manager = Mock()
        self.renderer = EnhancedMarkdownRenderer(self.mock_animation_manager)
    
    def test_code_block_with_animation_manager(self):
        """Test code block rendering with animation manager."""
        code = "print('Hello, World!')"
        widget = self.renderer._render_code_block(code, 'python')
        
        assert isinstance(widget, toga.Box)
        # Should have header with copy button
        assert len(widget.children) == 2
    
    def test_complex_markdown_rendering(self):
        """Test rendering complex markdown with all features."""
        markdown = """# Complex Document

This document contains **all** the *features* we support.

## Code Blocks

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

```javascript
function factorial(n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
```

## Tables

| Function | Language | Complexity |
|----------|----------|------------|
| fibonacci | Python | O(2^n) |
| factorial | JavaScript | O(n) |

## Lists

### Ordered List
1. First item
2. Second item
3. Third item

### Unordered List
- Bullet point 1
- Bullet point 2
- Bullet point 3

## Quotes

> "The best way to predict the future is to invent it."
> - Alan Kay

## Links and Inline Code

Check out the `fibonacci` function at [GitHub](https://github.com).

## Final Paragraph

This paragraph contains `inline code`, **bold text**, *italic text*, 
and [a link](https://example.com) to demonstrate all inline formatting."""
        
        widget = self.renderer.render_markdown(markdown)
        
        assert isinstance(widget, toga.Box)
        # Should have many child widgets for all the different blocks
        assert len(widget.children) >= 10
        
        # Verify we have different types of widgets
        widget_types = [type(child).__name__ for child in widget.children]
        assert 'Label' in widget_types  # Headers, paragraphs
        assert 'Box' in widget_types    # Code blocks, tables, lists