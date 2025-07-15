#!/usr/bin/env python3
"""
Test Script for UI Improvements

This script tests the UI improvements made to Tektra AI Assistant.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import toga
from tektra.gui.themes import theme_manager, LIGHT_THEME
from tektra.gui.progress_dialog import ProgressDialog


class TestUIApp(toga.App):
    """Test app to demonstrate UI improvements."""
    
    def startup(self):
        """Initialize test app."""
        self.main_window = toga.MainWindow(title="Tektra UI Test")
        
        # Test theme
        theme = theme_manager.get_theme()
        print(f"Current theme: {theme.name}")
        print(f"Primary color: {theme.colors.primary}")
        print(f"Background: {theme.colors.background}")
        
        # Build test interface
        main_box = toga.Box(style=toga.Pack(
            direction="column",
            padding=20,
            background_color=theme.colors.background
        ))
        
        # Test header
        header = toga.Label(
            "Tektra UI Improvements Test",
            style=toga.Pack(
                font_size=theme.typography["heading1"]["size"],
                font_weight=theme.typography["heading1"]["weight"],
                color=theme.colors.primary,
                margin_bottom=20
            )
        )
        main_box.add(header)
        
        # Test buttons
        button_box = toga.Box(style=toga.Pack(direction="row", padding=10))
        
        primary_button = toga.Button(
            "Primary Button",
            style=toga.Pack(
                padding=10,
                background_color=theme.colors.primary,
                color="#ffffff",
                font_weight="bold"
            )
        )
        button_box.add(primary_button)
        
        secondary_button = toga.Button(
            "Secondary Button",
            style=toga.Pack(
                padding=10,
                margin_left=10,
                background_color=theme.colors.surface,
                color=theme.colors.primary,
                border=f"2px solid {theme.colors.primary}"
            )
        )
        button_box.add(secondary_button)
        
        main_box.add(button_box)
        
        # Test progress dialog
        progress_button = toga.Button(
            "Test Progress Dialog",
            on_press=self.test_progress_dialog,
            style=toga.Pack(
                margin_top=20,
                padding=10,
                background_color=theme.colors.accent,
                color="#ffffff"
            )
        )
        main_box.add(progress_button)
        
        # Status indicators
        status_box = toga.Box(style=toga.Pack(
            direction="row",
            margin_top=20
        ))
        
        for status, color in [("Ready", theme.colors.success), 
                             ("Warning", theme.colors.warning),
                             ("Error", theme.colors.error)]:
            indicator = toga.Label(
                f"● {status}",
                style=toga.Pack(
                    margin_right=15,
                    padding=5,
                    color=color,
                    font_weight="normal"
                )
            )
            status_box.add(indicator)
        
        main_box.add(status_box)
        
        self.main_window.content = main_box
        self.main_window.show()
        
        print("\nUI Test App Started!")
        print("- Modern light theme applied")
        print("- Progress dialog available")
        print("- Clean, professional styling")
        
    async def test_progress_dialog(self, widget):
        """Test the progress dialog."""
        dialog = ProgressDialog(self, "Testing Progress Dialog")
        dialog.show()
        
        # Simulate progress
        for i in range(101):
            dialog.update(
                f"Processing... Step {i}",
                i,
                f"Simulating download: {i}MB / 100MB",
                bytes_downloaded=i * 1024 * 1024,
                total_bytes=100 * 1024 * 1024
            )
            await asyncio.sleep(0.05)
            
            if dialog.is_cancelled:
                break
        
        dialog.hide()


def main():
    """Run the test app."""
    app = TestUIApp("Tektra UI Test", "org.tektra.uitest")
    return app.main_loop()


if __name__ == "__main__":
    print("Starting Tektra UI Improvements Test...")
    print("This will demonstrate:")
    print("1. Modern light theme")
    print("2. Progress dialog")
    print("3. Professional styling")
    print()
    
    sys.exit(main())