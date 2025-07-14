"""
Modern Theme System for Tektra AI Assistant

This module provides a clean, professional theme system with light and dark modes.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ColorScheme:
    """Color scheme for a theme."""
    # Primary colors
    primary: str
    primary_dark: str
    primary_light: str
    
    # Accent colors
    accent: str
    accent_dark: str
    accent_light: str
    
    # Background colors
    background: str
    surface: str
    card: str
    
    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str
    
    # Status colors
    success: str
    warning: str
    error: str
    info: str
    
    # Border and divider colors
    border: str
    divider: str
    
    # Shadow colors
    shadow: str


@dataclass
class Theme:
    """Complete theme definition."""
    name: str
    colors: ColorScheme
    typography: Dict[str, Any]
    spacing: Dict[str, int]
    borders: Dict[str, Any]
    shadows: Dict[str, str]


# Modern Light Theme
LIGHT_THEME = Theme(
    name="Modern Light",
    colors=ColorScheme(
        # Primary colors - Modern blue
        primary="#0066cc",
        primary_dark="#0052a3",
        primary_light="#3385e0",
        
        # Accent colors - Fresh green
        accent="#00a862",
        accent_dark="#008a4f",
        accent_light="#33c77f",
        
        # Background colors - Clean and bright
        background="#ffffff",
        surface="#f8f9fa",
        card="#ffffff",
        
        # Text colors - Strong contrast
        text_primary="#1a1a1a",
        text_secondary="#6c757d",
        text_disabled="#adb5bd",
        
        # Status colors - Clear and modern
        success="#28a745",
        warning="#ffc107",
        error="#dc3545",
        info="#17a2b8",
        
        # Border and divider colors - Light
        border="#dee2e6",
        divider="#e9ecef",
        
        # Shadow color
        shadow="rgba(0, 0, 0, 0.08)"
    ),
    typography={
        "heading1": {
            "size": 24,
            "weight": "bold",
            "line_height": 1.2
        },
        "heading2": {
            "size": 20,
            "weight": "bold",
            "line_height": 1.3
        },
        "heading3": {
            "size": 18,
            "weight": "normal",
            "line_height": 1.4
        },
        "body1": {
            "size": 16,
            "weight": "normal",
            "line_height": 1.5
        },
        "body2": {
            "size": 14,
            "weight": "normal",
            "line_height": 1.5
        },
        "caption": {
            "size": 12,
            "weight": "normal",
            "line_height": 1.4
        },
        "button": {
            "size": 14,
            "weight": "normal",
            "line_height": 1.2,
            "text_transform": "uppercase"
        }
    },
    spacing={
        "xs": 4,
        "sm": 8,
        "md": 16,
        "lg": 24,
        "xl": 32,
        "xxl": 48
    },
    borders={
        "radius_sm": 4,
        "radius_md": 8,
        "radius_lg": 16,
        "radius_xl": 24,
        "width": 1
    },
    shadows={
        "sm": "0 1px 3px rgba(0, 0, 0, 0.12)",
        "md": "0 3px 6px rgba(0, 0, 0, 0.16)",
        "lg": "0 10px 20px rgba(0, 0, 0, 0.19)",
        "xl": "0 14px 28px rgba(0, 0, 0, 0.25)"
    }
)


# Modern Dark Theme (for future implementation)
DARK_THEME = Theme(
    name="Modern Dark",
    colors=ColorScheme(
        # Primary colors - Lighter blue for dark backgrounds
        primary="#42a5f5",
        primary_dark="#1976d2",
        primary_light="#64b5f6",
        
        # Accent colors - Bright teal
        accent="#26c6da",
        accent_dark="#00acc1",
        accent_light="#4dd0e1",
        
        # Background colors - Dark grays
        background="#121212",
        surface="#1e1e1e",
        card="#2c2c2c",
        
        # Text colors - Light on dark
        text_primary="#ffffff",
        text_secondary="#b3b3b3",
        text_disabled="#666666",
        
        # Status colors - Adjusted for dark background
        success="#66bb6a",
        warning="#ffa726",
        error="#ef5350",
        info="#42a5f5",
        
        # Border and divider colors - Subtle dark
        border="#333333",
        divider="#2a2a2a",
        
        # Shadow color
        shadow="rgba(0, 0, 0, 0.3)"
    ),
    typography=LIGHT_THEME.typography,  # Same typography
    spacing=LIGHT_THEME.spacing,  # Same spacing
    borders=LIGHT_THEME.borders,  # Same borders
    shadows={
        "sm": "0 1px 3px rgba(0, 0, 0, 0.3)",
        "md": "0 3px 6px rgba(0, 0, 0, 0.4)",
        "lg": "0 10px 20px rgba(0, 0, 0, 0.5)",
        "xl": "0 14px 28px rgba(0, 0, 0, 0.6)"
    }
)


class ThemeManager:
    """Manages theme switching and application."""
    
    def __init__(self, initial_theme: str = "light"):
        """
        Initialize theme manager.
        
        Args:
            initial_theme: "light" or "dark"
        """
        self.themes = {
            "light": LIGHT_THEME,
            "dark": DARK_THEME
        }
        self.current_theme_name = initial_theme
        self.current_theme = self.themes[initial_theme]
        
    def get_theme(self) -> Theme:
        """Get the current theme."""
        return self.current_theme
        
    def switch_theme(self, theme_name: str):
        """
        Switch to a different theme.
        
        Args:
            theme_name: "light" or "dark"
        """
        if theme_name in self.themes:
            self.current_theme_name = theme_name
            self.current_theme = self.themes[theme_name]
        else:
            raise ValueError(f"Unknown theme: {theme_name}")
            
    def get_style(self, element: str) -> Dict[str, Any]:
        """
        Get style dictionary for a specific element.
        
        Args:
            element: Element type (e.g., "button", "card", "input")
            
        Returns:
            Style dictionary for Toga Pack
        """
        theme = self.current_theme
        colors = theme.colors
        
        styles = {
            "app_background": {
                "background_color": colors.background
            },
            "header": {
                "background_color": colors.surface,
                "padding": theme.spacing["md"],
                "border_bottom": f"{theme.borders['width']}px solid {colors.border}"
            },
            "card": {
                "background_color": colors.card,
                "padding": theme.spacing["md"],
                "margin": theme.spacing["sm"],
                "border": f"{theme.borders['width']}px solid {colors.border}",
                "border_radius": theme.borders["radius_md"]
            },
            "button_primary": {
                "background_color": colors.primary,
                "color": "#ffffff",
                "padding": (theme.spacing["sm"], theme.spacing["md"]),
                "border_radius": theme.borders["radius_sm"],
                "font_size": theme.typography["button"]["size"],
                "font_weight": theme.typography["button"]["weight"]
            },
            "button_secondary": {
                "background_color": colors.surface,
                "color": colors.primary,
                "border": f"{theme.borders['width']}px solid {colors.primary}",
                "padding": (theme.spacing["sm"], theme.spacing["md"]),
                "border_radius": theme.borders["radius_sm"],
                "font_size": theme.typography["button"]["size"],
                "font_weight": theme.typography["button"]["weight"]
            },
            "input": {
                "background_color": colors.surface,
                "color": colors.text_primary,
                "border": f"{theme.borders['width']}px solid {colors.border}",
                "padding": theme.spacing["sm"],
                "border_radius": theme.borders["radius_sm"],
                "font_size": theme.typography["body2"]["size"]
            },
            "label": {
                "color": colors.text_primary,
                "font_size": theme.typography["body2"]["size"],
                "margin_bottom": theme.spacing["xs"]
            },
            "heading1": {
                "color": colors.text_primary,
                "font_size": theme.typography["heading1"]["size"],
                "font_weight": theme.typography["heading1"]["weight"],
                "margin_bottom": theme.spacing["md"]
            },
            "heading2": {
                "color": colors.text_primary,
                "font_size": theme.typography["heading2"]["size"],
                "font_weight": theme.typography["heading2"]["weight"],
                "margin_bottom": theme.spacing["sm"]
            },
            "body_text": {
                "color": colors.text_primary,
                "font_size": theme.typography["body1"]["size"],
                "line_height": theme.typography["body1"]["line_height"]
            },
            "caption": {
                "color": colors.text_secondary,
                "font_size": theme.typography["caption"]["size"],
            },
            "status_indicator": {
                "padding": (theme.spacing["xs"], theme.spacing["sm"]),
                "border_radius": theme.borders["radius_xl"],
                "font_size": theme.typography["caption"]["size"],
                "font_weight": "normal"
            },
            "message_user": {
                "background_color": colors.primary,
                "color": "#ffffff",
                "padding": theme.spacing["sm"],
                "border_radius": theme.borders["radius_md"],
                "margin": (theme.spacing["xs"], 0),
            },
            "message_assistant": {
                "background_color": colors.surface,
                "color": colors.text_primary,
                "border": f"{theme.borders['width']}px solid {colors.border}",
                "padding": theme.spacing["sm"],
                "border_radius": theme.borders["radius_md"],
                "margin": (theme.spacing["xs"], 0),
            },
            "sidebar": {
                "background_color": colors.surface,
                "border_right": f"{theme.borders['width']}px solid {colors.border}",
                "padding": theme.spacing["md"]
            }
        }
        
        return styles.get(element, {})


# Global theme manager instance
theme_manager = ThemeManager("light")