"""
Enhanced Markdown Renderer for Chat Messages

This module provides advanced markdown rendering capabilities for chat messages,
including syntax highlighting, copy-to-clipboard functionality, improved table
rendering, and enhanced visual styling.
"""

import re
import asyncio
import platform
import subprocess
from typing import Dict, List, Tuple, Any, Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from loguru import logger

# Import syntax highlighting
try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import get_formatter_by_name
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False
    logger.warning("Pygments not available - syntax highlighting disabled")


class EnhancedMarkdownRenderer:
    """
    Enhanced markdown renderer with syntax highlighting, copy functionality,
    improved table rendering, and better visual styling.
    
    Supports:
    - Headers (# ## ### #### ##### ######)
    - Bold and italic text with better visual indicators
    - Code blocks with syntax highlighting and copy buttons
    - Inline code with improved styling
    - Lists (unordered and ordered) with better formatting
    - Tables with proper alignment and styling
    - Links with hover effects and accessibility
    - Blockquotes with enhanced visual treatment
    - Line breaks and paragraphs with proper spacing
    """

    def __init__(self, animation_manager=None):
        """Initialize the enhanced markdown renderer."""
        self.animation_manager = animation_manager
        self.styles = self._create_enhanced_styles()
        self.syntax_theme = "default"  # Can be customized
        self.copy_feedback_duration = 2.0  # Duration for copy feedback
        
        # Track interactive elements for cleanup
        self.interactive_elements = {}
        
        logger.debug("Enhanced Markdown Renderer initialized")

    def _create_enhanced_styles(self) -> Dict[str, Pack]:
        """Create enhanced styles for different markdown elements."""
        return {
            # Enhanced headers with better typography hierarchy
            "h1": Pack(
                font_size=24, font_weight="bold", margin_bottom=16, margin_top=8,
                color="#1a202c", text_align="left"
            ),
            "h2": Pack(
                font_size=20, font_weight="bold", margin_bottom=12, margin_top=6,
                color="#2d3748", text_align="left"
            ),
            "h3": Pack(
                font_size=18, font_weight="bold", margin_bottom=10, margin_top=5,
                color="#4a5568", text_align="left"
            ),
            "h4": Pack(
                font_size=16, font_weight="bold", margin_bottom=8, margin_top=4,
                color="#4a5568", text_align="left"
            ),
            "h5": Pack(
                font_size=14, font_weight="bold", margin_bottom=6, margin_top=3,
                color="#718096", text_align="left"
            ),
            "h6": Pack(
                font_size=12, font_weight="bold", margin_bottom=4, margin_top=2,
                color="#718096", text_align="left"
            ),
            
            # Enhanced code block styling
            "code_block_container": Pack(
                background_color="#f7fafc",
                padding=0,
                margin=(8, 0),
                direction=COLUMN
            ),
            "code_block_header": Pack(
                background_color="#e2e8f0",
                padding=(6, 12),
                direction=ROW
            ),
            "code_block_content": Pack(
                font_family="monospace",
                background_color="#f7fafc",
                padding=12,
                color="#2d3748",
                font_size=13
            ),
            "code_language_label": Pack(
                font_size=11,
                color="#4a5568",
                font_weight="bold",
                font_family="monospace"
            ),
            "copy_button": Pack(
                font_size=11,
                padding=(4, 8),
                background_color="#4299e1",
                color="#ffffff",
                font_weight="bold"
            ),
            "copy_button_success": Pack(
                font_size=11,
                padding=(4, 8),
                background_color="#48bb78",
                color="#ffffff",
                font_weight="bold"
            ),
            
            # Enhanced inline code
            "inline_code": Pack(
                font_family="monospace",
                background_color="#edf2f7",
                padding=(2, 4),
                color="#d53f8c",
                font_size=13
            ),
            
            # Enhanced paragraph styling
            "paragraph": Pack(
                margin_bottom=12,
                color="#2d3748",
                font_size=14,
                text_align="left"
            ),
            
            # Enhanced list styling
            "list_container": Pack(
                direction=COLUMN,
                margin_bottom=12,
                padding_left=8
            ),
            "list_item": Pack(
                margin_bottom=6,
                color="#2d3748",
                font_size=14,
                text_align="left"
            ),
            "ordered_list_item": Pack(
                margin_bottom=6,
                color="#2d3748",
                font_size=14,
                text_align="left"
            ),
            
            # Enhanced link styling
            "link": Pack(
                color="#3182ce",
                font_weight="normal"
            ),
            "link_hover": Pack(
                color="#2c5282",
                font_weight="normal"
            ),
            
            # Enhanced blockquote styling
            "quote_container": Pack(
                background_color="#f7fafc",
                padding=16,
                margin=(8, 0),
                direction=COLUMN
            ),
            "quote_content": Pack(
                color="#4a5568",
                font_size=14,
                font_style="italic"
            ),
            
            # Table styling
            "table_container": Pack(
                direction=COLUMN,
                margin=(8, 0),
                background_color="#ffffff"
            ),
            "table_header_row": Pack(
                direction=ROW,
                background_color="#f7fafc",
                padding=(8, 0)
            ),
            "table_row": Pack(
                direction=ROW,
                padding=(6, 0)
            ),
            "table_header_cell": Pack(
                padding=(8, 12),
                font_weight="bold",
                color="#2d3748",
                font_size=13,
                flex=1
            ),
            "table_cell": Pack(
                padding=(6, 12),
                color="#4a5568",
                font_size=13,
                flex=1
            )
        }

    def render_markdown(self, markdown_text: str) -> toga.Box:
        """
        Render markdown text into a Toga widget container.
        
        Args:
            markdown_text: The markdown text to render
            
        Returns:
            toga.Box: Container with rendered content
        """
        container = toga.Box(style=Pack(direction=COLUMN, padding=5))
        
        # Split into blocks (paragraphs, code blocks, headers, etc.)
        blocks = self._parse_blocks(markdown_text)
        
        for block in blocks:
            widget = self._render_block(block)
            if widget:
                container.add(widget)
        
        return container

    def _parse_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into blocks.
        
        Args:
            text: Markdown text
            
        Returns:
            List of block dictionaries with type and content
        """
        blocks = []
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Skip empty lines
            if not line.strip():
                i += 1
                continue
            
            # Code blocks
            if line.strip().startswith('```'):
                language = line.strip()[3:].strip() or 'text'
                code_lines = []
                i += 1
                
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                
                blocks.append({
                    'type': 'code_block',
                    'content': '\n'.join(code_lines),
                    'language': language
                })
                i += 1  # Skip closing ```
                continue
            
            # Tables
            if '|' in line and line.count('|') >= 2:
                table_rows = []
                # Collect table rows
                while i < len(lines) and '|' in lines[i] and lines[i].count('|') >= 2:
                    table_rows.append(lines[i])
                    i += 1
                
                if table_rows:
                    blocks.append({
                        'type': 'table',
                        'content': table_rows
                    })
                    continue
            
            # Headers
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                level = len(header_match.group(1))
                content = header_match.group(2)
                blocks.append({
                    'type': f'h{level}',
                    'content': content
                })
                i += 1
                continue
            
            # Lists
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                list_items = []
                while i < len(lines) and (re.match(r'^\s*[-*+]\s+', lines[i]) or re.match(r'^\s*\d+\.\s+', lines[i])):
                    # Remove list markers (bullets or numbers)
                    if re.match(r'^\s*[-*+]\s+', lines[i]):
                        item_content = re.sub(r'^\s*[-*+]\s+', '', lines[i])
                    else:  # numbered list
                        item_content = re.sub(r'^\s*\d+\.\s+', '', lines[i])
                    list_items.append(item_content)
                    i += 1
                
                blocks.append({
                    'type': 'list',
                    'content': list_items
                })
                continue
            
            # Quotes
            if line.strip().startswith('>'):
                quote_lines = []
                while i < len(lines) and lines[i].strip().startswith('>'):
                    quote_content = lines[i].strip()[1:].strip()
                    quote_lines.append(quote_content)
                    i += 1
                
                blocks.append({
                    'type': 'quote',
                    'content': '\n'.join(quote_lines)
                })
                continue
            
            # Regular paragraphs
            paragraph_lines = [line]
            i += 1
            
            # Collect continuation lines
            while i < len(lines) and lines[i].strip() and not self._is_special_line(lines[i]):
                paragraph_lines.append(lines[i])
                i += 1
            
            blocks.append({
                'type': 'paragraph',
                'content': '\n'.join(paragraph_lines)
            })
        
        return blocks

    def _is_special_line(self, line: str) -> bool:
        """Check if a line starts a special markdown element."""
        line = line.strip()
        return (
            line.startswith('#') or
            line.startswith('```') or
            line.startswith('>') or
            re.match(r'^\s*[-*+]\s+', line) or
            re.match(r'^\s*\d+\.\s+', line)
        )

    def _render_block(self, block: Dict[str, Any]) -> toga.Widget:
        """
        Render a single markdown block.
        
        Args:
            block: Block dictionary with type and content
            
        Returns:
            toga.Widget: Rendered widget
        """
        block_type = block['type']
        content = block['content']
        
        if block_type == 'code_block':
            return self._render_code_block(content, block.get('language', 'text'))
        elif block_type.startswith('h'):
            return self._render_header(content, block_type)
        elif block_type == 'paragraph':
            return self._render_paragraph(content)
        elif block_type == 'list':
            return self._render_list(content)
        elif block_type == 'quote':
            return self._render_quote(content)
        elif block_type == 'table':
            return self._render_table(content)
        
        return None

    def _render_code_block(self, code: str, language: str) -> toga.Box:
        """Render an enhanced code block with syntax highlighting and copy functionality."""
        container = toga.Box(style=self.styles["code_block_container"])
        
        # Create header with language label and copy button
        if language and language != 'text':
            header = toga.Box(style=self.styles["code_block_header"])
            
            # Language label
            lang_label = toga.Label(
                f"📝 {language.upper()}",
                style=self.styles["code_language_label"]
            )
            header.add(lang_label)
            
            # Spacer to push copy button to the right
            spacer = toga.Box(style=Pack(flex=1))
            header.add(spacer)
            
            # Copy button with enhanced styling
            copy_button = toga.Button(
                "📋 Copy",
                style=self.styles["copy_button"],
                on_press=lambda x: asyncio.create_task(self._handle_copy_code(x, code))
            )
            
            # Set up micro-interactions for the copy button if animation manager is available
            if self.animation_manager:
                try:
                    micro_manager = self.animation_manager.micro_interaction_manager
                    copy_button_id = f"copy_button_{id(copy_button)}"
                    
                    element_id = micro_manager.setup_button_interactions(
                        copy_button,
                        button_id=copy_button_id,
                        interaction_config={
                            "hover_scale": 1.05,
                            "press_scale": 0.95,
                            "hover_duration": 0.15,
                            "press_duration": 0.1,
                            "spring_back_duration": 0.2,
                            "enable_spring_back": True
                        }
                    )
                    
                    self.interactive_elements[copy_button_id] = element_id
                    
                except Exception as e:
                    logger.debug(f"Could not set up micro-interactions for copy button: {e}")
            
            header.add(copy_button)
            container.add(header)
        
        # Apply syntax highlighting if available
        highlighted_code = self._apply_syntax_highlighting(code, language)
        
        # Create code content
        code_label = toga.Label(
            highlighted_code,
            style=self.styles["code_block_content"]
        )
        container.add(code_label)
        
        return container

    def _render_header(self, content: str, header_type: str) -> toga.Label:
        """Render a header."""
        # Process inline formatting
        formatted_content = self._process_inline_formatting(content)
        
        return toga.Label(
            formatted_content,
            style=self.styles[header_type]
        )

    def _render_paragraph(self, content: str) -> toga.Label:
        """Render a paragraph with inline formatting."""
        formatted_content = self._process_inline_formatting(content)
        
        return toga.Label(
            formatted_content,
            style=self.styles["paragraph"]
        )

    def _render_list(self, items: List[str]) -> toga.Box:
        """Render a list."""
        container = toga.Box(style=Pack(direction=COLUMN, margin_bottom=10))
        
        for item in items:
            formatted_item = self._process_inline_formatting(item)
            item_label = toga.Label(
                f"• {formatted_item}",
                style=self.styles["list_item"]
            )
            container.add(item_label)
        
        return container

    def _render_quote(self, content: str) -> toga.Box:
        """Render an enhanced blockquote."""
        container = toga.Box(style=self.styles["quote_container"])
        
        formatted_content = self._process_inline_formatting(content)
        quote_label = toga.Label(
            f"❝ {formatted_content}",
            style=self.styles["quote_content"]
        )
        container.add(quote_label)
        
        return container

    def _apply_syntax_highlighting(self, code: str, language: str) -> str:
        """
        Apply syntax highlighting to code using Pygments.
        
        Args:
            code: Code content to highlight
            language: Programming language
            
        Returns:
            str: Highlighted code (plain text with basic formatting for Toga)
        """
        if not PYGMENTS_AVAILABLE or not language or language == 'text':
            return code
        
        try:
            # Get lexer for the language
            lexer = get_lexer_by_name(language, stripall=True)
            
            # For Toga, we can't use HTML formatting, so we'll use a simple text formatter
            # that adds basic visual indicators for different token types
            tokens = list(lexer.get_tokens(code))
            highlighted_lines = []
            current_line = ""
            
            for token_type, value in tokens:
                # Add simple visual indicators for different token types
                if 'Keyword' in str(token_type):
                    current_line += f"🔵{value}"  # Blue circle for keywords
                elif 'String' in str(token_type):
                    current_line += f"🟢{value}"  # Green circle for strings
                elif 'Comment' in str(token_type):
                    current_line += f"🔘{value}"  # Gray circle for comments
                elif 'Number' in str(token_type):
                    current_line += f"🟡{value}"  # Yellow circle for numbers
                else:
                    current_line += value
                
                # Handle line breaks
                if '\n' in value:
                    lines = value.split('\n')
                    if len(lines) > 1:
                        current_line += lines[0]
                        highlighted_lines.append(current_line)
                        for line in lines[1:-1]:
                            highlighted_lines.append(line)
                        current_line = lines[-1]
            
            if current_line:
                highlighted_lines.append(current_line)
            
            return '\n'.join(highlighted_lines)
            
        except (ClassNotFound, Exception) as e:
            logger.debug(f"Could not highlight code for language '{language}': {e}")
            return code
    
    async def _handle_copy_code(self, button: toga.Button, code: str) -> None:
        """
        Handle copying code to clipboard with visual feedback.
        
        Args:
            button: The copy button that was pressed
            code: Code content to copy
        """
        try:
            # Copy to clipboard using platform-specific method
            success = await self._copy_to_clipboard(code)
            
            if success:
                # Provide visual feedback
                original_text = button.text
                original_style = button.style
                
                # Update button to show success
                button.text = "✅ Copied!"
                button.style = self.styles["copy_button_success"]
                
                # Animate button press if animation manager is available
                if self.animation_manager:
                    try:
                        await self.animation_manager.animate_button_press(button)
                    except Exception as e:
                        logger.debug(f"Could not animate copy button: {e}")
                
                # Reset button after delay
                await asyncio.sleep(self.copy_feedback_duration)
                button.text = original_text
                button.style = original_style
                
                logger.info(f"Code copied to clipboard ({len(code)} characters)")
            else:
                # Show error feedback
                original_text = button.text
                button.text = "❌ Failed"
                await asyncio.sleep(1.0)
                button.text = original_text
                
        except Exception as e:
            logger.error(f"Error copying code to clipboard: {e}")
    
    async def _copy_to_clipboard(self, text: str) -> bool:
        """
        Copy text to clipboard using platform-specific methods.
        
        Args:
            text: Text to copy
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                process = await asyncio.create_subprocess_exec(
                    "pbcopy",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate(input=text.encode())
                return process.returncode == 0
                
            elif system == "linux":
                # Try xclip first, then xsel
                for cmd in ["xclip", "xsel"]:
                    try:
                        if cmd == "xclip":
                            process = await asyncio.create_subprocess_exec(
                                "xclip", "-selection", "clipboard",
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                        else:  # xsel
                            process = await asyncio.create_subprocess_exec(
                                "xsel", "--clipboard", "--input",
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                        
                        await process.communicate(input=text.encode())
                        if process.returncode == 0:
                            return True
                    except FileNotFoundError:
                        continue
                return False
                
            elif system == "windows":
                process = await asyncio.create_subprocess_exec(
                    "clip",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate(input=text.encode())
                return process.returncode == 0
                
        except Exception as e:
            logger.debug(f"Clipboard copy failed: {e}")
            return False
        
        return False
    
    def _process_inline_formatting(self, text: str) -> str:
        """
        Process inline formatting like bold, italic, links, and inline code.
        
        Args:
            text: Text to process
            
        Returns:
            str: Processed text with enhanced formatting indicators
        """
        # Process links first [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'🔗 \1', text)
        
        # Convert markdown to enhanced Unicode equivalents
        text = re.sub(r'\*\*(.*?)\*\*', r'𝗕 \1', text)  # Bold (using bold Unicode)
        text = re.sub(r'\*(.*?)\*', r'𝘐 \1', text)      # Italic (using italic Unicode)
        text = re.sub(r'`(.*?)`', r'⟨\1⟩', text)        # Inline code with brackets
        
        return text
    
    def _render_table(self, table_rows: List[str]) -> toga.Box:
        """
        Render a table with proper styling and alignment.
        
        Args:
            table_rows: List of table row strings
            
        Returns:
            toga.Box: Rendered table container
        """
        container = toga.Box(style=self.styles["table_container"])
        
        if not table_rows:
            return container
        
        # Parse table rows
        parsed_rows = []
        for row in table_rows:
            # Remove leading/trailing pipes and split by pipe
            cells = [cell.strip() for cell in row.strip().strip('|').split('|')]
            parsed_rows.append(cells)
        
        # Skip separator row (usually second row with dashes)
        if len(parsed_rows) > 1 and all(cell.strip().replace('-', '').replace(':', '') == '' 
                                       for cell in parsed_rows[1]):
            header_row = parsed_rows[0]
            data_rows = parsed_rows[2:]
        else:
            header_row = parsed_rows[0] if parsed_rows else []
            data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
        
        # Render header row
        if header_row:
            header_container = toga.Box(style=self.styles["table_header_row"])
            for cell in header_row:
                formatted_cell = self._process_inline_formatting(cell)
                cell_label = toga.Label(
                    formatted_cell,
                    style=self.styles["table_header_cell"]
                )
                header_container.add(cell_label)
            container.add(header_container)
        
        # Render data rows
        for row in data_rows:
            row_container = toga.Box(style=self.styles["table_row"])
            for cell in row:
                formatted_cell = self._process_inline_formatting(cell)
                cell_label = toga.Label(
                    formatted_cell,
                    style=self.styles["table_cell"]
                )
                row_container.add(cell_label)
            container.add(row_container)
        
        return container
    
    def clear_interactive_elements(self) -> None:
        """Clear tracked interactive elements for cleanup."""
        self.interactive_elements.clear()
        logger.debug("Interactive elements cleared")
    
    def get_interactive_element_count(self) -> int:
        """Get the number of tracked interactive elements."""
        return len(self.interactive_elements)

    def render_simple_message(self, message: str, role: str = "assistant") -> toga.Box:
        """
        Render a simple message with basic markdown support.
        
        Args:
            message: Message content
            role: Message role (user, assistant, system)
            
        Returns:
            toga.Box: Rendered message container
        """
        container = toga.Box(style=Pack(direction=COLUMN, padding=8))
        
        # Detect if this might be markdown content
        has_markdown = any([
            '```' in message,
            re.search(r'^#{1,6}\s+', message, re.MULTILINE),
            re.search(r'\*\*(.*?)\*\*', message),
            re.search(r'`(.*?)`', message),
            re.search(r'^\s*[-*+]\s+', message, re.MULTILINE)
        ])
        
        if has_markdown:
            # Render as markdown
            markdown_widget = self.render_markdown(message)
            container.add(markdown_widget)
        else:
            # Render as plain text with basic formatting
            formatted_text = self._process_inline_formatting(message)
            text_label = toga.Label(
                formatted_text,
                style=Pack(
                    color="#333333",
                    font_size=14,
                    text_align="left"
                )
            )
            container.add(text_label)
        
        return container


# Backward compatibility - keep the old class name as an alias
MarkdownRenderer = EnhancedMarkdownRenderer

# Global renderer instance
_markdown_renderer = None

def get_markdown_renderer(animation_manager=None) -> EnhancedMarkdownRenderer:
    """Get the global markdown renderer instance."""
    global _markdown_renderer
    if _markdown_renderer is None:
        _markdown_renderer = EnhancedMarkdownRenderer(animation_manager)
    return _markdown_renderer

def get_enhanced_markdown_renderer(animation_manager=None) -> EnhancedMarkdownRenderer:
    """Get an enhanced markdown renderer instance with animation support."""
    return EnhancedMarkdownRenderer(animation_manager)