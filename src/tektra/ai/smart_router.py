"""
Smart Query Router

This module provides intelligent routing between different AI systems:
- Unmute for natural conversational queries
- Qwen for complex analytical and vision tasks
- Hybrid processing for mixed queries
- Context-aware routing decisions
"""

import asyncio
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from loguru import logger
import time

from .qwen_backend import QwenBackend
from ..voice import VoiceConversationPipeline
from .multimodal import MultimodalProcessor


class QueryRoute(Enum):
    """Query routing destinations."""
    UNMUTE_VOICE = "unmute_voice"        # Natural conversation via Unmute
    QWEN_ANALYTICAL = "qwen_analytical"  # Complex reasoning via Qwen
    QWEN_VISION = "qwen_vision"          # Vision tasks via Qwen-VL  
    MIXED = "mixed"                      # Use both systems strategically
    ERROR = "error"                      # Routing error


@dataclass
class QueryContext:
    """Context information for query routing."""
    query_text: str
    has_image: bool = False
    image_data: Any = None
    file_attachments: List[Dict[str, Any]] = None
    is_voice_input: bool = False
    conversation_history: List[Dict[str, str]] = None
    user_preference: Optional[str] = None
    session_context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.file_attachments is None:
            self.file_attachments = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.session_context is None:
            self.session_context = {}


@dataclass 
class RoutingResult:
    """Result of query routing decision."""
    route: QueryRoute
    confidence: float
    reasoning: str
    processing_type: str  # 'sync', 'async', 'hybrid'
    estimated_response_time: float
    recommended_model: Optional[str] = None
    context_used: List[str] = None
    
    def __post_init__(self):
        if self.context_used is None:
            self.context_used = []


class QueryClassifier:
    """Classifies queries into different types for routing."""
    
    def __init__(self):
        """Initialize query classifier with keyword patterns."""
        
        # Conversational patterns - route to Unmute
        self.conversational_patterns = {
            'greetings': [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\bhow are you\b', r'\bnice to meet you\b'
            ],
            'social': [
                r'\b(thanks|thank you|please|sorry)\b',
                r'\b(goodbye|bye|see you|talk later)\b'
            ],
            'casual_questions': [
                r'\bwhat\'?s up\b', r'\bhow\'?s it going\b',
                r'\btell me (a joke|about yourself)\b'
            ],
            'simple_requests': [
                r'\b(weather|time|date)\b',
                r'\bcan you (help|assist)\b'
            ]
        }
        
        # Analytical patterns - route to Qwen
        self.analytical_patterns = {
            'complex_reasoning': [
                r'\b(analyze|analysis|explain|explanation)\b',
                r'\b(calculate|computation|solve|solution)\b',
                r'\b(algorithm|logic|reasoning|theory)\b',
                r'\bwhy does\b', r'\bhow does\b'
            ],
            'technical': [
                r'\b(code|programming|function|debug)\b',
                r'\b(technical|scientific|mathematical)\b',
                r'\b(implementation|architecture|design)\b'
            ],
            'research': [
                r'\b(research|study|investigation|detailed)\b',
                r'\b(comprehensive|thorough|in-depth)\b',
                r'\bcompare and contrast\b'
            ],
            'problem_solving': [
                r'\bfind a solution\b', r'\bstep by step\b',
                r'\bmethodology|approach|strategy\b'
            ]
        }
        
        # Vision patterns - route to Qwen-VL
        self.vision_patterns = {
            'image_analysis': [
                r'\b(image|picture|photo|visual)\b',
                r'\bwhat do you see\b', r'\bdescribe (this|the)\b',
                r'\banalyze (this|the) (image|picture|photo)\b'
            ],
            'visual_questions': [
                r'\bin this (image|picture|photo)\b',
                r'\bwhat\'?s in\b', r'\bidentify\b',
                r'\brecognize|detect|find\b'
            ]
        }
        
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_patterns = {}
        
        for category, patterns_dict in [
            ('conversational', self.conversational_patterns),
            ('analytical', self.analytical_patterns), 
            ('vision', self.vision_patterns)
        ]:
            self.compiled_patterns[category] = {}
            for subcategory, patterns in patterns_dict.items():
                self.compiled_patterns[category][subcategory] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]
    
    def classify_query(self, context: QueryContext) -> Dict[str, float]:
        """
        Classify query and return confidence scores for each category.
        
        Args:
            context: Query context information
            
        Returns:
            Dict mapping categories to confidence scores (0-1)
        """
        query_text = context.query_text.lower()
        scores = {
            'conversational': 0.0,
            'analytical': 0.0,
            'vision': 0.0
        }
        
        # Check for explicit vision context
        if context.has_image or context.image_data:
            scores['vision'] += 0.7
        
        # Check for file attachments
        if context.file_attachments:
            scores['analytical'] += 0.3  # File analysis is usually analytical
        
        # Pattern matching
        for category, subcategories in self.compiled_patterns.items():
            category_score = 0.0
            matches = 0
            
            for subcategory, patterns in subcategories.items():
                for pattern in patterns:
                    if pattern.search(query_text):
                        matches += 1
                        # Weight certain patterns higher
                        if subcategory in ['complex_reasoning', 'image_analysis']:
                            category_score += 0.3
                        else:
                            category_score += 0.2
            
            # Normalize and cap at 1.0
            if matches > 0:
                scores[category] = min(1.0, category_score)
        
        # Length-based heuristics
        word_count = len(query_text.split())
        if word_count > 15:
            scores['analytical'] += 0.2  # Long queries tend to be analytical
        elif word_count < 5:
            scores['conversational'] += 0.2  # Short queries tend to be conversational
        
        # Voice input bias
        if context.is_voice_input:
            scores['conversational'] += 0.1  # Voice tends to be more conversational
        
        # Conversation history context
        if context.conversation_history:
            recent_messages = context.conversation_history[-3:]  # Last 3 messages
            for msg in recent_messages:
                if msg.get('role') == 'assistant':
                    # If previous responses were analytical, continue the pattern
                    if len(msg.get('content', '')) > 200:
                        scores['analytical'] += 0.1
        
        # Normalize scores to ensure they sum reasonably
        total = sum(scores.values())
        if total > 1.5:  # Prevent over-inflation
            for key in scores:
                scores[key] = scores[key] / total * 1.5
        
        return scores


class SmartRouter:
    """
    Intelligent router that directs queries to the most appropriate AI system.
    
    Routes between:
    - Unmute: Natural conversation, quick responses
    - Qwen: Complex analysis, vision tasks, detailed reasoning
    """
    
    def __init__(
        self,
        qwen_backend: QwenBackend,
        voice_pipeline: VoiceConversationPipeline,
        multimodal_processor: Optional[MultimodalProcessor] = None
    ):
        """
        Initialize smart router.
        
        Args:
            qwen_backend: Qwen backend for analytical tasks
            voice_pipeline: Voice pipeline for conversational tasks
            multimodal_processor: Optional multimodal processor
        """
        self.qwen_backend = qwen_backend
        self.voice_pipeline = voice_pipeline
        self.multimodal_processor = multimodal_processor or MultimodalProcessor()
        
        # Query classifier
        self.classifier = QueryClassifier()
        
        # Routing statistics
        self.routing_stats = {
            'total_queries': 0,
            'unmute_routed': 0,
            'qwen_routed': 0,
            'mixed_routed': 0,
            'routing_errors': 0,
            'avg_decision_time': 0.0
        }
        
        # Configuration
        self.config = {
            'confidence_threshold': 0.6,  # Minimum confidence for routing decision
            'mixed_query_threshold': 0.4,  # Threshold for mixed routing
            'voice_bias': 0.1,  # Bias towards conversational for voice input
            'enable_hybrid_routing': True
        }
        
        logger.info("Smart router initialized")
    
    async def route_query(self, context: QueryContext) -> Dict[str, Any]:
        """
        Route query to appropriate AI system.
        
        Args:
            context: Query context with text and metadata
            
        Returns:
            Dict containing routing decision and response
        """
        start_time = time.time()
        self.routing_stats['total_queries'] += 1
        
        try:
            # Classify the query
            classification_scores = self.classifier.classify_query(context)
            
            # Make routing decision
            routing_result = await self._make_routing_decision(context, classification_scores)
            
            logger.info(f"Query routed to: {routing_result.route.value} "
                       f"(confidence: {routing_result.confidence:.2f})")
            
            # Execute the routing
            response = await self._execute_routing(context, routing_result)
            
            # Update statistics
            decision_time = time.time() - start_time
            self._update_stats(routing_result.route, decision_time)
            
            return {
                'routing_decision': routing_result,
                'response': response,
                'processing_time': decision_time,
                'classification_scores': classification_scores
            }
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            self.routing_stats['routing_errors'] += 1
            
            return {
                'routing_decision': RoutingResult(
                    route=QueryRoute.ERROR,
                    confidence=0.0,
                    reasoning=f"Routing error: {e}",
                    processing_type='error',
                    estimated_response_time=0.0
                ),
                'response': f"Sorry, I encountered an error processing your request: {e}",
                'error': str(e)
            }
    
    async def _make_routing_decision(
        self, 
        context: QueryContext, 
        scores: Dict[str, float]
    ) -> RoutingResult:
        """Make intelligent routing decision based on classification scores."""
        
        # Special cases first
        if context.has_image or context.image_data or scores['vision'] > 0.5:
            return RoutingResult(
                route=QueryRoute.QWEN_VISION,
                confidence=max(0.7, scores['vision']),
                reasoning="Query involves image analysis - routed to Qwen-VL",
                processing_type='sync',
                estimated_response_time=5.0,
                recommended_model='qwen-vl',
                context_used=['image_data', 'vision_patterns']
            )
        
        # Determine primary route
        max_score = max(scores.values())
        primary_category = max(scores, key=scores.get)
        
        # Check for mixed queries
        secondary_scores = [s for s in scores.values() if s != max_score and s > 0.3]
        is_mixed = len(secondary_scores) > 0 and max_score < 0.7
        
        if is_mixed and self.config['enable_hybrid_routing']:
            return RoutingResult(
                route=QueryRoute.MIXED,
                confidence=0.6,
                reasoning=f"Mixed query detected: {primary_category} ({max_score:.2f}) with secondary patterns",
                processing_type='hybrid',
                estimated_response_time=3.0,
                context_used=['multiple_patterns', 'hybrid_logic']
            )
        
        # Route based on highest scoring category
        if primary_category == 'conversational' and max_score > self.config['confidence_threshold']:
            # Add voice input bias
            confidence = max_score
            if context.is_voice_input:
                confidence = min(1.0, confidence + self.config['voice_bias'])
            
            return RoutingResult(
                route=QueryRoute.UNMUTE_VOICE,
                confidence=confidence,
                reasoning=f"Conversational query (score: {max_score:.2f}) - routed to Unmute",
                processing_type='async',
                estimated_response_time=1.0,
                recommended_model='unmute-llm',
                context_used=['conversational_patterns', 'voice_input' if context.is_voice_input else None]
            )
            
        elif primary_category == 'analytical' and max_score > self.config['confidence_threshold']:
            return RoutingResult(
                route=QueryRoute.QWEN_ANALYTICAL,
                confidence=max_score,
                reasoning=f"Analytical query (score: {max_score:.2f}) - routed to Qwen",
                processing_type='sync',
                estimated_response_time=3.0,
                recommended_model='qwen-text',
                context_used=['analytical_patterns']
            )
        
        # Default routing based on input type and context
        if context.is_voice_input or max_score < 0.4:
            # Default to conversational for voice or low-confidence queries
            return RoutingResult(
                route=QueryRoute.UNMUTE_VOICE,
                confidence=0.5,
                reasoning="Low confidence or voice input - defaulting to conversational",
                processing_type='async',
                estimated_response_time=1.0,
                recommended_model='unmute-llm',
                context_used=['default_logic', 'voice_input' if context.is_voice_input else 'low_confidence']
            )
        else:
            # Default to analytical for text input
            return RoutingResult(
                route=QueryRoute.QWEN_ANALYTICAL,
                confidence=0.5,
                reasoning="Text input with unclear intent - defaulting to analytical",
                processing_type='sync',
                estimated_response_time=3.0,
                recommended_model='qwen-text',
                context_used=['default_logic', 'text_input']
            )
    
    async def _execute_routing(self, context: QueryContext, routing: RoutingResult) -> str:
        """Execute the routing decision and get response."""
        
        try:
            if routing.route == QueryRoute.UNMUTE_VOICE:
                return await self._handle_unmute_query(context)
            
            elif routing.route == QueryRoute.QWEN_ANALYTICAL:
                return await self._handle_qwen_analytical(context)
            
            elif routing.route == QueryRoute.QWEN_VISION:
                return await self._handle_qwen_vision(context)
            
            elif routing.route == QueryRoute.MIXED:
                return await self._handle_mixed_query(context)
            
            else:
                raise ValueError(f"Unknown routing destination: {routing.route}")
                
        except Exception as e:
            logger.error(f"Error executing routing for {routing.route}: {e}")
            # Fallback to mock response when backends are unavailable
            return await self._generate_fallback_response(context, routing.route, e)
    
    async def _generate_fallback_response(self, context: QueryContext, attempted_route: QueryRoute, error: Exception) -> str:
        """Generate a fallback response when backends are unavailable."""
        query = context.query_text.lower()
        
        # Simple keyword-based responses
        if any(greeting in query for greeting in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm Tektra AI Assistant. I'm currently in limited mode while my AI models are loading. How can I help you today?"
        
        elif any(word in query for word in ['how', 'what', 'why', 'when', 'where', 'help']):
            return f"I understand you're asking about '{context.query_text}'. I'm currently initializing my AI models and can't provide detailed analysis yet. Please try again in a few moments, or ask me something simple!"
        
        elif any(word in query for word in ['analyze', 'explain', 'complex', 'detailed']):
            return f"I would love to provide a detailed analysis of '{context.query_text}', but my analytical AI system (Qwen) is currently loading. This usually takes a few minutes. Please try again shortly!"
        
        elif any(word in query for word in ['image', 'picture', 'photo', 'visual', 'see']):
            return f"I can help with image analysis once my vision system is ready. The AI models are currently downloading and should be available soon. Please try again in a few minutes!"
        
        elif any(word in query for word in ['voice', 'speak', 'talk', 'listen']):
            return f"Voice conversation features are available, but I'm currently in text-only mode while the AI models finish loading. You can still chat with me here!"
        
        else:
            return f"I received your message: '{context.query_text}'. I'm currently in limited mode while my AI systems initialize. This includes downloading large language models which may take a few minutes. Please try again soon for full AI capabilities!"
    
    async def _handle_unmute_query(self, context: QueryContext) -> str:
        """Handle query via Unmute voice system."""
        try:
            # Send to Unmute for processing
            success = await self.voice_pipeline.send_text_message(context.query_text)
            
            if success:
                return "Processing via voice conversation system..."  # Actual response comes via callbacks
            else:
                raise RuntimeError("Failed to send message to Unmute")
                
        except Exception as e:
            logger.error(f"Error in Unmute processing: {e}")
            raise
    
    async def _handle_qwen_analytical(self, context: QueryContext) -> str:
        """Handle analytical query via Qwen."""
        try:
            # Check if Qwen backend is initialized
            if not self.qwen_backend or not self.qwen_backend.is_initialized:
                raise RuntimeError("Qwen backend not initialized")
            
            # Prepare context for Qwen
            qwen_context = {
                'conversation_history': context.conversation_history,
                'session_context': context.session_context
            }
            
            # Process file attachments if any
            if context.file_attachments:
                qwen_context['attachments'] = await self._process_attachments(context.file_attachments)
            
            # Generate response
            response = await self.qwen_backend.generate_response(context.query_text, qwen_context)
            return response
            
        except Exception as e:
            logger.error(f"Error in Qwen analytical processing: {e}")
            raise
    
    async def _handle_qwen_vision(self, context: QueryContext) -> str:
        """Handle vision query via Qwen-VL."""
        try:
            if not (context.has_image or context.image_data):
                raise ValueError("Vision query but no image data provided")
            
            # Process the image
            if context.image_data:
                response = await self.qwen_backend.process_vision_query(
                    context.query_text, 
                    context.image_data
                )
            else:
                # Handle case where image is in file attachments
                image_attachments = [
                    att for att in context.file_attachments 
                    if att.get('content_type') == 'image'
                ]
                
                if not image_attachments:
                    raise ValueError("No image found in attachments")
                
                # Use the first image
                image_data = image_attachments[0]['image']
                response = await self.qwen_backend.process_vision_query(
                    context.query_text,
                    image_data
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Qwen vision processing: {e}")
            raise
    
    async def _handle_mixed_query(self, context: QueryContext) -> str:
        """Handle mixed query using both systems strategically."""
        try:
            # Start with conversational acknowledgment via Unmute
            await self.voice_pipeline.send_text_message(
                f"Let me analyze that for you: {context.query_text}"
            )
            
            # Then process with Qwen for detailed analysis
            analytical_response = await self._handle_qwen_analytical(context)
            
            # Return the analytical response (Unmute response comes via callbacks)
            return analytical_response
            
        except Exception as e:
            logger.error(f"Error in mixed query processing: {e}")
            # Fallback to analytical only
            return await self._handle_qwen_analytical(context)
    
    async def _process_attachments(self, attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process file attachments for context."""
        processed = []
        
        for attachment in attachments:
            try:
                # Process based on content type
                if attachment.get('content_type') == 'document':
                    processed.append({
                        'type': 'document',
                        'content': attachment.get('content', ''),
                        'metadata': attachment.get('metadata', {})
                    })
                elif attachment.get('content_type') == 'image':
                    processed.append({
                        'type': 'image',
                        'analysis': attachment.get('analysis', {}),
                        'metadata': attachment.get('metadata', {})
                    })
                
            except Exception as e:
                logger.warning(f"Error processing attachment: {e}")
                continue
        
        return processed
    
    def _update_stats(self, route: QueryRoute, decision_time: float):
        """Update routing statistics."""
        if route == QueryRoute.UNMUTE_VOICE:
            self.routing_stats['unmute_routed'] += 1
        elif route in [QueryRoute.QWEN_ANALYTICAL, QueryRoute.QWEN_VISION]:
            self.routing_stats['qwen_routed'] += 1
        elif route == QueryRoute.MIXED:
            self.routing_stats['mixed_routed'] += 1
        
        # Update average decision time
        total = self.routing_stats['total_queries']
        current_avg = self.routing_stats['avg_decision_time']
        self.routing_stats['avg_decision_time'] = (
            (current_avg * (total - 1)) + decision_time
        ) / total
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        total = max(1, self.routing_stats['total_queries'])
        
        return {
            **self.routing_stats,
            'unmute_percentage': self.routing_stats['unmute_routed'] / total * 100,
            'qwen_percentage': self.routing_stats['qwen_routed'] / total * 100,
            'mixed_percentage': self.routing_stats['mixed_routed'] / total * 100,
            'error_rate': self.routing_stats['routing_errors'] / total * 100,
            'config': self.config.copy()
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update router configuration."""
        self.config.update(new_config)
        logger.info(f"Router configuration updated: {new_config}")
    
    async def cleanup(self):
        """Cleanup router resources."""
        logger.info("Smart router cleanup complete")