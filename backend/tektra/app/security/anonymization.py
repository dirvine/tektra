"""
Anonymization Service.

Anonymizes queries before sending to external APIs to protect user privacy.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class AnonymizationService:
    """Service for anonymizing queries sent to external APIs."""

    def __init__(self):
        # PII patterns for detection and removal
        self.pii_patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "url": re.compile(
                r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?"
            ),
            "file_path": re.compile(
                r'[A-Za-z]:\\(?:[^\\/:*?"<>|]+\\)*[^\\/:*?"<>|]*|/(?:[^/\s]*/)*/[^/\s]*'
            ),
            "api_key": re.compile(r"\b[A-Za-z0-9]{32,}\b"),
        }

        # Context-specific patterns
        self.context_patterns = {
            "lab_equipment": re.compile(
                r"\b(?:microscope|centrifuge|spectrometer|incubator|robot|arm)\s+(?:ID|#|serial)?\s*[A-Z0-9-]+\b",
                re.IGNORECASE,
            ),
            "project_names": re.compile(r"\bproject\s+[A-Za-z0-9-]+\b", re.IGNORECASE),
            "internal_domains": re.compile(
                r"\b\w+\.(?:local|internal|lab|private)\b", re.IGNORECASE
            ),
        }

        # Replacement mappings for consistency
        self.replacement_cache: Dict[str, str] = {}
        self.anonymization_log: List[Dict] = []

    def _generate_replacement(self, original: str, category: str) -> str:
        """Generate consistent replacement for PII."""
        # Create hash-based replacement for consistency
        hash_value = hashlib.md5(original.encode()).hexdigest()[:8]

        replacements = {
            "email": f"user{hash_value}@example.com",
            "phone": f"+1-555-{hash_value[:3]}-{hash_value[3:7]}",
            "ssn": f"XXX-XX-{hash_value[:4]}",
            "credit_card": f"****-****-****-{hash_value[:4]}",
            "ip_address": f"192.168.1.{int(hash_value[:2], 16) % 255}",
            "url": f"https://example.com/{hash_value}",
            "file_path": f"/path/to/{hash_value}",
            "api_key": f"anonymous_key_{hash_value}",
            "lab_equipment": f"DEVICE_{hash_value.upper()}",
            "project_names": f"PROJECT_{hash_value.upper()}",
            "internal_domains": f"internal{hash_value[:4]}.example.com",
        }

        return replacements.get(category, f"ANONYMOUS_{hash_value.upper()}")

    def _anonymize_text(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Anonymize text by removing/replacing PII.

        Args:
            text: Original text

        Returns:
            Tuple of (anonymized_text, detected_pii_list)
        """
        anonymized = text
        detected_pii = []

        # Process all PII patterns
        all_patterns = {**self.pii_patterns, **self.context_patterns}

        for category, pattern in all_patterns.items():
            matches = pattern.findall(anonymized)

            for match in matches:
                original = match if isinstance(match, str) else match[0]

                # Get or generate replacement
                if original in self.replacement_cache:
                    replacement = self.replacement_cache[original]
                else:
                    replacement = self._generate_replacement(original, category)
                    self.replacement_cache[original] = replacement

                # Replace in text
                anonymized = anonymized.replace(original, replacement)

                # Log detection
                detected_pii.append(
                    {
                        "category": category,
                        "original_length": len(original),
                        "replacement": replacement,
                        "context": "query_anonymization",
                    }
                )

        return anonymized, detected_pii

    def _generalize_context(self, text: str) -> str:
        """
        Generalize specific contexts while preserving meaning.

        Args:
            text: Text to generalize

        Returns:
            Generalized text
        """
        generalizations = [
            # Specific to general locations
            (
                re.compile(
                    r"\b(?:in|at|from)\s+(?:our|my|the)\s+lab(?:oratory)?\b",
                    re.IGNORECASE,
                ),
                "in the laboratory",
            ),
            (
                re.compile(
                    r"\b(?:our|my|the)\s+(?:company|organization|team)\b", re.IGNORECASE
                ),
                "the organization",
            ),
            # Specific to general timeframes
            (
                re.compile(
                    r"\b(?:yesterday|today|tomorrow|this week|last week|next week)\b",
                    re.IGNORECASE,
                ),
                "recently",
            ),
            (
                re.compile(
                    r"\b(?:this morning|this afternoon|tonight)\b", re.IGNORECASE
                ),
                "earlier",
            ),
            # Specific to general quantities
            (
                re.compile(
                    r"\b\d+\s*(?:ml|μl|mg|kg|g|pounds?|ounces?)\b", re.IGNORECASE
                ),
                "a measured amount",
            ),
            (
                re.compile(r"\b\d+\s*(?:degrees?|°[CF])\b", re.IGNORECASE),
                "at temperature",
            ),
            # Personal pronouns to general
            (re.compile(r"\b(?:I|we|my|our)\b", re.IGNORECASE), "the user"),
        ]

        result = text
        for pattern, replacement in generalizations:
            result = pattern.sub(replacement, result)

        return result

    def anonymize_query(
        self, query: str, preserve_technical: bool = True
    ) -> Dict[str, Any]:
        """
        Anonymize a query for external API usage.

        Args:
            query: Original user query
            preserve_technical: Whether to preserve technical terminology

        Returns:
            Anonymization result with anonymized query and metadata
        """
        try:
            # Step 1: Remove/replace PII
            anonymized_text, detected_pii = self._anonymize_text(query)

            # Step 2: Generalize context
            generalized_text = self._generalize_context(anonymized_text)

            # Step 3: Technical term preservation
            if preserve_technical:
                # Keep scientific/technical terms that don't reveal sensitive info
                technical_preservations = [
                    "DNA",
                    "RNA",
                    "protein",
                    "enzyme",
                    "molecule",
                    "cell",
                    "tissue",
                    "microscopy",
                    "spectroscopy",
                    "chromatography",
                    "electrophoresis",
                    "PCR",
                    "sequencing",
                    "analysis",
                    "synthesis",
                    "purification",
                    "temperature",
                    "pressure",
                    "concentration",
                    "pH",
                    "buffer",
                    "algorithm",
                    "model",
                    "neural network",
                    "machine learning",
                ]
                # Technical terms are naturally preserved unless they're PII

            # Create anonymization result
            result = {
                "original_query": query,
                "anonymized_query": generalized_text,
                "detected_pii": detected_pii,
                "pii_count": len(detected_pii),
                "anonymization_level": "high" if detected_pii else "low",
                "preserve_technical": preserve_technical,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Log anonymization
            self.anonymization_log.append(
                {
                    "timestamp": result["timestamp"],
                    "pii_count": result["pii_count"],
                    "query_length": len(query),
                    "anonymized_length": len(generalized_text),
                    "categories": list(set(item["category"] for item in detected_pii)),
                }
            )

            logger.info(f"Anonymized query: {len(detected_pii)} PII items removed")
            return result

        except Exception as e:
            logger.error(f"Query anonymization failed: {e}")
            # Fallback: heavily redacted query
            return {
                "original_query": query,
                "anonymized_query": "General research question about laboratory procedures",
                "detected_pii": [],
                "pii_count": 0,
                "anonymization_level": "fallback",
                "preserve_technical": preserve_technical,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            }

    def anonymize_response(self, response: str, query_context: Dict) -> str:
        """
        Clean external API response before returning to user.

        Args:
            response: External API response
            query_context: Context from original anonymization

        Returns:
            Cleaned response
        """
        try:
            # Remove any accidental PII that might be in response
            cleaned_response, _ = self._anonymize_text(response)

            # Remove references to example domains/data
            cleaned_response = re.sub(
                r"example\.com|anonymous_key_\w+|DEVICE_\w+",
                "[REDACTED]",
                cleaned_response,
            )

            return cleaned_response

        except Exception as e:
            logger.error(f"Response cleaning failed: {e}")
            return response

    def get_anonymization_stats(self) -> Dict[str, Any]:
        """Get anonymization statistics."""
        if not self.anonymization_log:
            return {
                "total_queries": 0,
                "total_pii_removed": 0,
                "common_categories": [],
                "average_pii_per_query": 0,
            }

        total_queries = len(self.anonymization_log)
        total_pii = sum(entry["pii_count"] for entry in self.anonymization_log)

        # Count categories
        category_counts = {}
        for entry in self.anonymization_log:
            for category in entry["categories"]:
                category_counts[category] = category_counts.get(category, 0) + 1

        common_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_queries": total_queries,
            "total_pii_removed": total_pii,
            "common_categories": common_categories,
            "average_pii_per_query": (
                total_pii / total_queries if total_queries > 0 else 0
            ),
            "replacement_cache_size": len(self.replacement_cache),
        }

    def clear_cache(self):
        """Clear anonymization cache and logs."""
        self.replacement_cache.clear()
        self.anonymization_log.clear()
        logger.info("Anonymization cache cleared")

    def validate_anonymization(self, original: str, anonymized: str) -> Dict[str, Any]:
        """
        Validate that anonymization was effective.

        Args:
            original: Original text
            anonymized: Anonymized text

        Returns:
            Validation results
        """
        # Check for potential leaks
        leaks = []

        # Check for email patterns
        if self.pii_patterns["email"].search(anonymized):
            leaks.append("email_leak")

        # Check for phone patterns
        if self.pii_patterns["phone"].search(anonymized):
            leaks.append("phone_leak")

        # Check for file paths
        if self.pii_patterns["file_path"].search(anonymized):
            leaks.append("path_leak")

        # Check for API keys
        if self.pii_patterns["api_key"].search(anonymized):
            leaks.append("key_leak")

        return {
            "is_safe": len(leaks) == 0,
            "detected_leaks": leaks,
            "reduction_ratio": len(anonymized) / len(original) if original else 1.0,
            "anonymization_effective": len(leaks) == 0
            and len(anonymized) < len(original),
        }


# Global service instance
anonymization_service = AnonymizationService()
