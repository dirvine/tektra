"""
Voice-First Interaction Patterns

This module implements natural voice interaction patterns that make Tektra
feel more conversational and intuitive to use.
"""

import re
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

from loguru import logger


class VoiceMode(Enum):
    """Voice interaction modes."""
    INACTIVE = "inactive"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    PUSH_TO_TALK = "push_to_talk"
    ALWAYS_ON = "always_on"
    WAKE_WORD = "wake_word"


class VoicePattern:
    """Represents a voice interaction pattern."""
    
    def __init__(self, name: str, trigger: str, response_template: str, priority: int = 1):
        self.name = name
        self.trigger = trigger
        self.response_template = response_template
        self.priority = priority


class VoiceInteractionPatterns:
    """
    Manages voice-first interaction patterns for natural conversation.
    
    Features:
    - Wake word detection patterns
    - Natural conversation flow
    - Voice command recognition
    - Interruption handling
    - Context-aware responses
    """

    def __init__(self):
        """Initialize voice interaction patterns."""
        self.current_mode = VoiceMode.INACTIVE
        self.is_listening = False
        self.last_interaction = None
        self.conversation_state = {}
        
        # Voice patterns
        self.patterns = self._initialize_patterns()
        self.wake_words = ["hey tektra", "tektra", "hey assistant"]
        
        # Timing settings
        self.silence_timeout = 3.0  # Seconds of silence before stopping
        self.wake_word_timeout = 5.0  # Seconds to wait after wake word
        self.response_delay = 0.5  # Natural pause before responding
        
        logger.info("Voice interaction patterns initialized")

    def _initialize_patterns(self) -> List[VoicePattern]:
        """Initialize common voice interaction patterns."""
        return [
            # Greetings and conversation starters
            VoicePattern(
                "greeting",
                r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
                "Hello! How can I help you today?",
                priority=5
            ),
            
            # Questions about capabilities
            VoicePattern(
                "what_can_you_do",
                r"what can you (do|help with)|what are you capable of",
                "I can help you with conversations, run Python code, analyze files, and create AI agents. What would you like to explore?",
                priority=4
            ),
            
            # Voice-specific commands
            VoicePattern(
                "stop_listening",
                r"\b(stop listening|stop|cancel|never mind)\b",
                "Okay, I'll stop listening now.",
                priority=10
            ),
            
            VoicePattern(
                "repeat_last",
                r"\b(repeat|say that again|what did you say)\b",
                "I said: {last_response}",
                priority=3
            ),
            
            # Natural conversation transitions
            VoicePattern(
                "change_topic",
                r"\b(let's talk about|can we discuss|tell me about)\b",
                "Sure! What would you like to know about {topic}?",
                priority=2
            ),
            
            # Agent-related voice commands
            VoicePattern(
                "create_agent_voice",
                r"\b(create|make|build) (an agent|agent) (that|to|for)\b",
                "I'll help you create an agent. What should it do?",
                priority=6
            ),
            
            # Code execution requests
            VoicePattern(
                "run_code_voice",
                r"\b(run|execute|calculate) (some code|code|this)\b",
                "I'll run that code for you. Let me process it...",
                priority=4
            ),
            
            # File analysis requests
            VoicePattern(
                "analyze_file_voice",
                r"\b(analyze|look at|examine) (this|the) (file|image|document)\b",
                "I'll analyze that file for you. What specifically would you like to know?",
                priority=4
            ),
        ]

    def detect_wake_word(self, text: str) -> bool:
        """
        Detect wake words in transcribed text.
        
        Args:
            text: Transcribed text to check
            
        Returns:
            bool: True if wake word detected
        """
        text_lower = text.lower().strip()
        
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                logger.info(f"Wake word detected: '{wake_word}' in '{text}'")
                return True
        
        return False

    def should_activate_voice(self, text: str) -> bool:
        """
        Determine if voice mode should be activated.
        
        Args:
            text: Transcribed text
            
        Returns:
            bool: True if voice should be activated
        """
        # Wake word detection
        if self.detect_wake_word(text):
            return True
        
        # Direct questions or commands
        question_patterns = [
            r"^(what|how|when|where|why|who|can you|will you|could you)",
            r"\?$",  # Ends with question mark
            r"\b(please|help|assist)\b"
        ]
        
        text_lower = text.lower()
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Question pattern detected: {pattern}")
                return True
        
        return False

    def process_voice_input(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        Process voice input and determine appropriate response pattern.
        
        Args:
            text: Transcribed voice input
            context: Optional context information
            
        Returns:
            dict: Response information with pattern, text, and metadata
        """
        text = text.strip()
        context = context or {}
        
        # Check for pattern matches
        for pattern in sorted(self.patterns, key=lambda p: p.priority, reverse=True):
            if re.search(pattern.trigger, text, re.IGNORECASE):
                response = self._format_response(pattern, text, context)
                
                return {
                    "pattern": pattern.name,
                    "response": response,
                    "should_speak": True,
                    "confidence": 0.8,
                    "metadata": {
                        "trigger": pattern.trigger,
                        "original_text": text,
                        "priority": pattern.priority
                    }
                }
        
        # No specific pattern matched, use general conversational response
        return {
            "pattern": "general_conversation",
            "response": None,  # Will be processed by main LLM
            "should_speak": True,
            "confidence": 0.5,
            "metadata": {
                "original_text": text,
                "needs_llm_processing": True
            }
        }

    def _format_response(self, pattern: VoicePattern, text: str, context: Dict) -> str:
        """
        Format a response based on the pattern and context.
        
        Args:
            pattern: Matched voice pattern
            text: Original text
            context: Context information
            
        Returns:
            str: Formatted response
        """
        response = pattern.response_template
        
        # Handle template variables
        if "{last_response}" in response:
            last_response = context.get("last_response", "I don't remember what I said last.")
            response = response.replace("{last_response}", last_response)
        
        if "{topic}" in response:
            # Extract topic from text
            topic_match = re.search(r"(about|discuss) (.+)", text, re.IGNORECASE)
            topic = topic_match.group(2) if topic_match else "that topic"
            response = response.replace("{topic}", topic)
        
        return response

    def get_voice_status_message(self, mode: VoiceMode) -> str:
        """
        Get a natural status message for voice mode changes.
        
        Args:
            mode: Current voice mode
            
        Returns:
            str: Natural status message
        """
        messages = {
            VoiceMode.LISTENING: "I'm listening...",
            VoiceMode.THINKING: "Let me think about that...",
            VoiceMode.SPEAKING: "Here's what I think...",
            VoiceMode.INACTIVE: "Voice mode is off.",
            VoiceMode.PUSH_TO_TALK: "Press and hold to talk to me.",
            VoiceMode.ALWAYS_ON: "I'm always listening for your voice.",
            VoiceMode.WAKE_WORD: "Say 'Hey Tektra' to wake me up."
        }
        
        return messages.get(mode, "Voice status updated.")

    def should_interrupt_speaking(self, new_input: str) -> bool:
        """
        Determine if new voice input should interrupt current speaking.
        
        Args:
            new_input: New voice input
            
        Returns:
            bool: True if should interrupt
        """
        interrupt_patterns = [
            r"\b(stop|wait|hold on|pause)\b",
            r"\b(excuse me|sorry|interrupt)\b",
            r"\b(hey tektra|tektra)\b"  # Wake word should interrupt
        ]
        
        text_lower = new_input.lower()
        for pattern in interrupt_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"Interrupt detected: {pattern}")
                return True
        
        return False

    def get_natural_transitions(self) -> Dict[str, str]:
        """
        Get natural transition phrases for voice interactions.
        
        Returns:
            dict: Transition phrases for different contexts
        """
        return {
            "thinking": [
                "Let me think about that...",
                "Hmm, interesting question...",
                "Give me a moment to process that...",
                "That's a good point, let me consider..."
            ],
            "processing": [
                "I'm working on that...",
                "Processing your request...",
                "Let me take care of that...",
                "Working on it..."
            ],
            "clarification": [
                "Could you clarify what you mean?",
                "I'm not sure I understand. Could you rephrase that?",
                "Can you be more specific about that?",
                "What exactly would you like me to help with?"
            ],
            "completion": [
                "Done! Is there anything else you'd like to know?",
                "All finished. What else can I help with?",
                "There you go! Any other questions?",
                "Task completed. What's next?"
            ],
            "error": [
                "I ran into a problem with that. Let me try a different approach.",
                "Something went wrong. Could you try rephrasing your request?",
                "I'm having trouble with that. Can you help me understand what you need?",
                "Oops, that didn't work as expected. Let's try again."
            ]
        }

    def adjust_response_for_voice(self, text_response: str) -> str:
        """
        Adjust a text response to be more natural for voice.
        
        Args:
            text_response: Original text response
            
        Returns:
            str: Voice-optimized response
        """
        # Remove markdown formatting for voice
        voice_response = text_response
        
        # Remove code block markers
        voice_response = re.sub(r'```[\w]*\n?', '', voice_response)
        voice_response = re.sub(r'```', '', voice_response)
        
        # Convert markdown formatting to natural speech
        voice_response = re.sub(r'\*\*(.*?)\*\*', r'\1', voice_response)  # Bold
        voice_response = re.sub(r'\*(.*?)\*', r'\1', voice_response)      # Italic
        voice_response = re.sub(r'`(.*?)`', r'\1', voice_response)        # Inline code
        
        # Replace bullet points with natural speech
        voice_response = re.sub(r'^\s*[-*+]\s+', 'First, ', voice_response, count=1, flags=re.MULTILINE)
        voice_response = re.sub(r'^\s*[-*+]\s+', 'Next, ', voice_response, flags=re.MULTILINE)
        
        # Remove excessive newlines
        voice_response = re.sub(r'\n{3,}', '\n\n', voice_response)
        
        # Add natural pauses
        voice_response = voice_response.replace('. ', '. ... ')
        voice_response = voice_response.replace('! ', '! ... ')
        voice_response = voice_response.replace('? ', '? ... ')
        
        return voice_response.strip()

    def get_conversation_starters(self) -> List[str]:
        """
        Get natural conversation starters for voice interaction.
        
        Returns:
            list: Conversation starter phrases
        """
        return [
            "Hi there! What can I help you with today?",
            "Hello! I'm ready to assist you. What's on your mind?",
            "Hey! What would you like to explore together?",
            "Hi! I'm here to help. What questions do you have?",
            "Hello! Ready for a conversation? What interests you?",
            "Hey there! What can we work on together today?",
        ]


# Global voice patterns instance
_voice_patterns = None

def get_voice_patterns() -> VoiceInteractionPatterns:
    """Get the global voice patterns instance."""
    global _voice_patterns
    if _voice_patterns is None:
        _voice_patterns = VoiceInteractionPatterns()
    return _voice_patterns