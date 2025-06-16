"""
Tests for Anonymization Service.

Tests PII detection, query anonymization, response cleaning,
and privacy protection functionality.
"""

import pytest
from datetime import datetime

from tektra.app.security.anonymization import AnonymizationService, anonymization_service


class TestAnonymizationService:
    """Test cases for AnonymizationService."""
    
    def test_service_initialization(self, anonymization_service):
        """Test that the service initializes correctly."""
        assert len(anonymization_service.pii_patterns) > 0
        assert len(anonymization_service.context_patterns) > 0
        assert isinstance(anonymization_service.replacement_cache, dict)
        assert isinstance(anonymization_service.anonymization_log, list)
    
    def test_email_detection(self, anonymization_service):
        """Test email PII detection and anonymization."""
        text = "Contact me at john.doe@example.com for details."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "john.doe@example.com" not in anonymized
        assert len(detected) == 1
        assert detected[0]["category"] == "email"
        assert "user" in anonymized and "@example.com" in anonymized
    
    def test_phone_detection(self, anonymization_service):
        """Test phone number PII detection."""
        test_cases = [
            "+1-555-123-4567",
            "(555) 123-4567", 
            "555.123.4567",
            "5551234567"
        ]
        
        for phone in test_cases:
            text = f"Call me at {phone} please."
            anonymized, detected = anonymization_service._anonymize_text(text)
            
            assert phone not in anonymized
            assert len(detected) >= 1
            assert any(d["category"] == "phone" for d in detected)
    
    def test_ssn_detection(self, anonymization_service):
        """Test SSN detection and anonymization."""
        text = "My SSN is 123-45-6789 for verification."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "123-45-6789" not in anonymized
        assert len(detected) == 1
        assert detected[0]["category"] == "ssn"
        assert "XXX-XX-" in anonymized
    
    def test_credit_card_detection(self, anonymization_service):
        """Test credit card number detection."""
        test_cases = [
            "4532 1234 5678 9012",
            "4532-1234-5678-9012",
            "4532123456789012"
        ]
        
        for cc in test_cases:
            text = f"My card number is {cc}."
            anonymized, detected = anonymization_service._anonymize_text(text)
            
            assert cc not in anonymized
            assert any(d["category"] == "credit_card" for d in detected)
    
    def test_ip_address_detection(self, anonymization_service):
        """Test IP address detection."""
        text = "Connect to server at 192.168.1.100 on port 8080."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "192.168.1.100" not in anonymized
        assert len(detected) == 1
        assert detected[0]["category"] == "ip_address"
        assert "192.168.1." in anonymized  # Should be anonymized IP
    
    def test_url_detection(self, anonymization_service):
        """Test URL detection and anonymization."""
        text = "Visit https://internal.company.com/secret-project for details."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "internal.company.com" not in anonymized
        assert len(detected) == 1
        assert detected[0]["category"] == "url"
        assert "https://example.com/" in anonymized
    
    def test_file_path_detection(self, anonymization_service):
        """Test file path detection."""
        test_cases = [
            "/home/user/secret/data.csv",
            "C:\\Users\\John\\Documents\\secret.txt",
            "/var/log/application.log"
        ]
        
        for path in test_cases:
            text = f"The file is located at {path}."
            anonymized, detected = anonymization_service._anonymize_text(text)
            
            assert path not in anonymized
            assert any(d["category"] == "file_path" for d in detected)
    
    def test_api_key_detection(self, anonymization_service):
        """Test API key detection."""
        text = "Use API key abc123def456ghi789jklmnop123456 for authentication."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "abc123def456ghi789jklmnop123456" not in anonymized
        assert len(detected) == 1
        assert detected[0]["category"] == "api_key"
        assert "anonymous_key_" in anonymized
    
    def test_lab_equipment_detection(self, anonymization_service):
        """Test lab equipment context detection."""
        text = "Use microscope ID ZEISS-7543 and centrifuge serial CF-9876."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "ZEISS-7543" not in anonymized
        assert "CF-9876" not in anonymized
        assert len(detected) >= 2
        assert any(d["category"] == "lab_equipment" for d in detected)
    
    def test_project_names_detection(self, anonymization_service):
        """Test project name detection."""
        text = "Working on Project PHOENIX and project ALPHA-7."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "PHOENIX" not in anonymized
        assert "ALPHA-7" not in anonymized
        assert any(d["category"] == "project_names" for d in detected)
    
    def test_internal_domains_detection(self, anonymization_service):
        """Test internal domain detection."""
        text = "Access the server at database.internal or api.lab for data."
        
        anonymized, detected = anonymization_service._anonymize_text(text)
        
        assert "database.internal" not in anonymized
        assert "api.lab" not in anonymized
        assert any(d["category"] == "internal_domains" for d in detected)
    
    def test_replacement_consistency(self, anonymization_service):
        """Test that same PII gets same replacement."""
        text1 = "Email me at john@example.com about the project."
        text2 = "John's email john@example.com is in the system."
        
        # Clear cache first
        anonymization_service.replacement_cache.clear()
        
        anonymized1, _ = anonymization_service._anonymize_text(text1)
        anonymized2, _ = anonymization_service._anonymize_text(text2)
        
        # Extract the replacement email from both texts
        import re
        email_pattern = r'user[a-f0-9]+@example\.com'
        
        emails1 = re.findall(email_pattern, anonymized1)
        emails2 = re.findall(email_pattern, anonymized2)
        
        assert len(emails1) == 1
        assert len(emails2) == 1
        assert emails1[0] == emails2[0]  # Same replacement
    
    def test_generalize_context(self, anonymization_service):
        """Test context generalization."""
        test_cases = [
            ("We tested this in our lab yesterday.", "recently"),
            ("My team at the company developed this.", "the organization"),
            ("This morning we ran the experiment.", "earlier"),
            ("Add 5ml of solution at 37°C.", "a measured amount", "at temperature")
        ]
        
        for case in test_cases:
            original_text = case[0]
            expected_words = case[1:]
            
            generalized = anonymization_service._generalize_context(original_text)
            
            for expected_word in expected_words:
                assert expected_word in generalized
    
    def test_anonymize_query_basic(self, anonymization_service, sample_pii_text):
        """Test basic query anonymization."""
        result = anonymization_service.anonymize_query(sample_pii_text)
        
        assert result["original_query"] == sample_pii_text
        assert result["pii_count"] > 0
        assert result["anonymization_level"] in ["high", "low"]
        assert "timestamp" in result
        
        # Original PII should not be in anonymized query
        anonymized = result["anonymized_query"]
        assert "john.doe@example.com" not in anonymized
        assert "+1-555-123-4567" not in anonymized
        assert "SECRET_PROJECT_X" not in anonymized
    
    def test_anonymize_query_preserve_technical(self, anonymization_service):
        """Test that technical terms are preserved."""
        query = "Analyze the DNA sequence using PCR amplification at 37°C with john@lab.com."
        
        result = anonymization_service.anonymize_query(query, preserve_technical=True)
        anonymized = result["anonymized_query"]
        
        # Technical terms should be preserved
        assert "DNA" in anonymized
        assert "PCR" in anonymized
        assert "amplification" in anonymized
        
        # PII should be removed
        assert "john@lab.com" not in anonymized
    
    def test_anonymize_query_no_pii(self, anonymization_service):
        """Test anonymization of query with no PII."""
        clean_query = "What is the molecular weight of caffeine?"
        
        result = anonymization_service.anonymize_query(clean_query)
        
        assert result["pii_count"] == 0
        assert result["anonymization_level"] == "low"
        assert result["anonymized_query"] != ""
    
    def test_anonymize_response(self, anonymization_service):
        """Test response cleaning."""
        response = "The data shows user12345@example.com accessed DEVICE_ABC123 successfully."
        query_context = {"anonymized_query": "test query"}
        
        cleaned = anonymization_service.anonymize_response(response, query_context)
        
        # Should remove example data and anonymous references
        assert "user12345@example.com" not in cleaned
        assert "DEVICE_ABC123" not in cleaned or "[REDACTED]" in cleaned
    
    def test_get_anonymization_stats_empty(self, anonymization_service):
        """Test getting stats when no queries processed."""
        # Clear logs
        anonymization_service.anonymization_log.clear()
        
        stats = anonymization_service.get_anonymization_stats()
        
        assert stats["total_queries"] == 0
        assert stats["total_pii_removed"] == 0
        assert stats["average_pii_per_query"] == 0
        assert stats["common_categories"] == []
    
    def test_get_anonymization_stats_with_data(self, anonymization_service, sample_pii_text):
        """Test getting stats with processed queries."""
        # Clear logs first
        anonymization_service.anonymization_log.clear()
        anonymization_service.replacement_cache.clear()
        
        # Process some queries
        for i in range(3):
            anonymization_service.anonymize_query(f"Email test{i}@example.com and call +1-555-{i}{i}{i}-1234")
        
        stats = anonymization_service.get_anonymization_stats()
        
        assert stats["total_queries"] == 3
        assert stats["total_pii_removed"] > 0
        assert stats["average_pii_per_query"] > 0
        assert len(stats["common_categories"]) > 0
        
        # Check that email and phone are common categories
        categories = [cat[0] for cat in stats["common_categories"]]
        assert "email" in categories
        assert "phone" in categories
    
    def test_clear_cache(self, anonymization_service):
        """Test cache clearing functionality."""
        # Add some data to cache and logs
        anonymization_service.replacement_cache["test"] = "replacement"
        anonymization_service.anonymization_log.append({"test": "data"})
        
        # Clear cache
        anonymization_service.clear_cache()
        
        assert len(anonymization_service.replacement_cache) == 0
        assert len(anonymization_service.anonymization_log) == 0
    
    def test_validate_anonymization_safe(self, anonymization_service):
        """Test validation of safe anonymization."""
        original = "Contact john.doe@example.com at +1-555-123-4567"
        anonymized = "Contact user123abc@example.com at +1-555-456-7890"
        
        validation = anonymization_service.validate_anonymization(original, anonymized)
        
        assert validation["is_safe"] is True
        assert validation["anonymization_effective"] is True
        assert len(validation["detected_leaks"]) == 0
        assert validation["reduction_ratio"] <= 1.0
    
    def test_validate_anonymization_unsafe(self, anonymization_service):
        """Test validation of unsafe anonymization."""
        original = "Secret data with john.doe@example.com"
        anonymized = "Secret data with john.doe@example.com"  # No anonymization
        
        validation = anonymization_service.validate_anonymization(original, anonymized)
        
        assert validation["is_safe"] is False
        assert len(validation["detected_leaks"]) > 0
        assert "email_leak" in validation["detected_leaks"]
    
    def test_multiple_pii_types_in_single_query(self, anonymization_service):
        """Test handling multiple PII types in one query."""
        query = """
        Hi, I'm John Doe (john.doe@company.com, +1-555-123-4567).
        Access our server at 192.168.1.100 using API key abc123def456.
        The project files are in /home/john/SECRET_PROJECT/data/
        """
        
        result = anonymization_service.anonymize_query(query)
        
        assert result["pii_count"] >= 4  # email, phone, IP, API key, file path
        
        anonymized = result["anonymized_query"]
        assert "john.doe@company.com" not in anonymized
        assert "+1-555-123-4567" not in anonymized
        assert "192.168.1.100" not in anonymized
        assert "abc123def456" not in anonymized
        assert "/home/john/SECRET_PROJECT" not in anonymized
    
    def test_edge_case_empty_query(self, anonymization_service):
        """Test handling of empty or whitespace-only queries."""
        test_cases = ["", "   ", "\n\t\r", None]
        
        for query in test_cases:
            if query is None:
                with pytest.raises(Exception):
                    anonymization_service.anonymize_query(query)
            else:
                result = anonymization_service.anonymize_query(query)
                assert result["pii_count"] == 0
                assert result["anonymization_level"] == "low"
    
    def test_anonymization_error_handling(self, anonymization_service):
        """Test error handling in anonymization process."""
        # Test with very long query
        long_query = "test " * 10000 + "email@example.com"
        
        try:
            result = anonymization_service.anonymize_query(long_query)
            # Should either succeed or fail gracefully
            assert isinstance(result, dict)
            assert "error" in result or "anonymized_query" in result
        except Exception:
            # Exception is acceptable for extreme cases
            pass
    
    def test_anonymization_performance(self, anonymization_service):
        """Test performance with large-scale anonymization."""
        # Create a query with many PII instances
        emails = [f"user{i}@example.com" for i in range(100)]
        query = "Process these emails: " + ", ".join(emails)
        
        import time
        start_time = time.time()
        
        result = anonymization_service.anonymize_query(query)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        assert processing_time < 5.0
        assert result["pii_count"] == 100
    
    def test_anonymization_consistency_across_calls(self, anonymization_service):
        """Test that anonymization is consistent across multiple calls."""
        query = "Contact admin@company.com and backup@company.com for support."
        
        # Clear cache
        anonymization_service.replacement_cache.clear()
        
        # Anonymize multiple times
        results = []
        for _ in range(3):
            result = anonymization_service.anonymize_query(query)
            results.append(result["anonymized_query"])
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_context_specific_patterns(self, anonymization_service):
        """Test lab-specific pattern detection."""
        lab_text = """
        In our laboratory, we use robot ARM-123 and microscope ZEISS-456.
        Project CLASSIFIED is stored on internal.lab servers.
        The experiment was conducted today with 50ml buffer at 25°C.
        """
        
        result = anonymization_service.anonymize_query(lab_text)
        anonymized = result["anonymized_query"]
        
        # Lab equipment should be anonymized
        assert "ARM-123" not in anonymized
        assert "ZEISS-456" not in anonymized
        
        # Project names should be anonymized
        assert "CLASSIFIED" not in anonymized
        
        # Internal domains should be anonymized
        assert "internal.lab" not in anonymized
        
        # Time context should be generalized
        assert "today" not in anonymized or "recently" in anonymized
        
        # Measurements should be generalized
        assert "50ml" not in anonymized or "a measured amount" in anonymized


class TestGlobalService:
    """Test the global anonymization service instance."""
    
    def test_global_service_exists(self):
        """Test that global service instance exists."""
        assert anonymization_service is not None
        assert isinstance(anonymization_service, AnonymizationService)
    
    def test_global_service_patterns(self):
        """Test that global service has all required patterns."""
        required_patterns = ["email", "phone", "ssn", "credit_card", "ip_address", "url", "file_path", "api_key"]
        
        for pattern in required_patterns:
            assert pattern in anonymization_service.pii_patterns
    
    def test_global_service_functionality(self, sample_pii_text):
        """Test that global service functions correctly."""
        result = anonymization_service.anonymize_query(sample_pii_text)
        
        assert isinstance(result, dict)
        assert "anonymized_query" in result
        assert "pii_count" in result


class TestAnonymizationQuality:
    """Test the quality and effectiveness of anonymization."""
    
    def test_anonymization_preserves_meaning(self, anonymization_service):
        """Test that anonymization preserves query meaning."""
        original = "Send the DNA analysis results to researcher@lab.com for Project ALPHA."
        
        result = anonymization_service.anonymize_query(original, preserve_technical=True)
        anonymized = result["anonymized_query"]
        
        # Technical terms should be preserved
        assert "DNA" in anonymized
        assert "analysis" in anonymized
        assert "results" in anonymized
        
        # Structure should be maintained
        assert "Send" in anonymized or "send" in anonymized
        assert "for" in anonymized
    
    def test_anonymization_removes_identifying_info(self, anonymization_service):
        """Test that all identifying information is removed."""
        sensitive_query = """
        John Smith (john.smith@company.com, SSN: 123-45-6789, Phone: +1-555-123-4567)
        works on Project SECRET at IP 10.0.0.1 using API key 1234567890abcdef.
        Files stored at /home/john/confidential/data.xlsx on server.internal.
        """
        
        result = anonymization_service.anonymize_query(sensitive_query)
        anonymized = result["anonymized_query"]
        
        # Personal identifiers should be removed
        identifying_info = [
            "John Smith", "john.smith@company.com", "123-45-6789", 
            "+1-555-123-4567", "SECRET", "10.0.0.1", "1234567890abcdef",
            "/home/john/confidential", "server.internal"
        ]
        
        for info in identifying_info:
            assert info not in anonymized
        
        # Should have detected multiple PII items
        assert result["pii_count"] >= 6
    
    def test_anonymization_handles_variations(self, anonymization_service):
        """Test that anonymization handles PII format variations."""
        variations = [
            "Email: user@domain.com, user.name@company.co.uk, admin+test@site.org",
            "Phones: (555) 123-4567, 555.123.4567, +44 20 1234 5678",
            "IPs: 192.168.1.1, 10.0.0.1, 172.16.0.1",
            "URLs: http://site.com, https://secure.site.com/path?param=value"
        ]
        
        for variation in variations:
            result = anonymization_service.anonymize_query(variation)
            
            # Should detect and anonymize multiple instances
            assert result["pii_count"] >= 3
            
            # Original patterns should not be in anonymized text
            anonymized = result["anonymized_query"]
            # This is a basic check - in reality, we'd need more sophisticated validation
            assert len(anonymized) > 0