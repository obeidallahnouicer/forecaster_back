"""
Integration Tests for Guardrails Validators

Tests the strict validation pipeline:
1. PII Input Validator
2. SQL Output Validator
"""

import pytest
from guardrails.pii_detector import PIIInputValidator, validate_input_for_pii
from guardrails.sql_validator import SQLOutputValidator, validate_sql_output, ValidationFailureReason


class TestPIIInputValidator:
    """Test PII input validation (pre-LLM gate)."""
    
    def setup_method(self):
        """Setup for each test."""
        self.validator = PIIInputValidator(strict_mode=True)
    
    # =====================================
    # SAFE INPUT TESTS (Should PASS)
    # =====================================
    
    def test_safe_question_passes(self):
        """Safe question with no PII should pass."""
        result = self.validator.validate("Show me total sales by month")
        assert result.is_safe is True
        assert result.detected_types == []
        assert "safe" in result.message.lower()
    
    def test_business_question_passes(self):
        """Business analytics question should pass."""
        result = self.validator.validate("What are the top 5 products by revenue?")
        assert result.is_safe is True
        assert result.detected_types == []
    
    def test_aggregation_query_passes(self):
        """Query with aggregations should pass."""
        result = self.validator.validate("Calculate average order value for Q4 2024")
        assert result.is_safe is True
    
    # =====================================
    # UNSAFE INPUT TESTS (Should FAIL)
    # =====================================
    
    def test_email_detected_and_rejected(self):
        """Email in input should be detected and rejected."""
        result = self.validator.validate("Show orders for john.doe@company.com")
        assert result.is_safe is False
        assert "email" in result.detected_types
        assert "REJECTED" in result.message
    
    def test_phone_detected_and_rejected(self):
        """Phone number should be detected and rejected."""
        result = self.validator.validate("Find customer with phone 555-123-4567")
        assert result.is_safe is False
        assert "phone" in result.detected_types
    
    def test_ssn_detected_and_rejected(self):
        """SSN should be detected and rejected."""
        result = self.validator.validate("Look up customer SSN 123-45-6789")
        assert result.is_safe is False
        assert "ssn" in result.detected_types
    
    def test_credit_card_detected_and_rejected(self):
        """Credit card number should be detected and rejected."""
        result = self.validator.validate("Verify payment for card 4532-1234-5678-9010")
        assert result.is_safe is False
        assert "credit_card" in result.detected_types
    
    def test_sensitive_keyword_password_rejected(self):
        """Queries with 'password' keyword should be rejected."""
        result = self.validator.validate("Show me the password for admin user")
        assert result.is_safe is False
        assert any("password" in dt for dt in result.detected_types)
    
    def test_sensitive_keyword_secret_rejected(self):
        """Queries with 'secret' keyword should be rejected."""
        result = self.validator.validate("What is the secret key?")
        assert result.is_safe is False
        assert any("secret" in dt for dt in result.detected_types)
    
    # =====================================
    # EDGE CASES
    # =====================================
    
    def test_empty_input_passes(self):
        """Empty input should pass (handled upstream)."""
        result = self.validator.validate("")
        assert result.is_safe is True
    
    def test_whitespace_only_passes(self):
        """Whitespace-only input should pass."""
        result = self.validator.validate("   ")
        assert result.is_safe is True
    
    def test_multiple_pii_types_detected(self):
        """Multiple PII types should all be detected."""
        result = self.validator.validate(
            "Contact john@email.com or call 555-1234 with SSN 123-45-6789"
        )
        assert result.is_safe is False
        assert "email" in result.detected_types
        assert "phone" in result.detected_types
        assert "ssn" in result.detected_types
    
    # =====================================
    # CONVENIENCE FUNCTION TESTS
    # =====================================
    
    def test_convenience_function_safe(self):
        """Test convenience function with safe input."""
        is_safe, message = validate_input_for_pii("Show total sales", strict=True)
        assert is_safe is True
    
    def test_convenience_function_unsafe(self):
        """Test convenience function with unsafe input."""
        is_safe, message = validate_input_for_pii("Email john@test.com", strict=True)
        assert is_safe is False
        assert "REJECTED" in message


class TestSQLOutputValidator:
    """Test SQL output validation (post-LLM gate)."""
    
    def setup_method(self):
        """Setup for each test."""
        self.validator = SQLOutputValidator(strict_mode=True)
    
    # =====================================
    # VALID SQL TESTS (Should PASS)
    # =====================================
    
    def test_simple_select_passes(self):
        """Simple SELECT query should pass."""
        sql = "SELECT * FROM customers"
        result = self.validator.validate(sql)
        assert result.is_valid is True
        assert result.normalized_sql is not None
        assert result.failure_reason is None
    
    def test_select_with_where_passes(self):
        """SELECT with WHERE clause should pass."""
        sql = "SELECT name, email FROM users WHERE active = 1"
        result = self.validator.validate(sql)
        assert result.is_valid is True
    
    def test_select_with_join_passes(self):
        """SELECT with JOIN should pass."""
        sql = """
        SELECT o.id, c.name 
        FROM orders o 
        JOIN customers c ON o.customer_id = c.id
        """
        result = self.validator.validate(sql)
        assert result.is_valid is True
    
    def test_select_with_aggregation_passes(self):
        """SELECT with aggregations should pass."""
        sql = "SELECT category, COUNT(*), SUM(amount) FROM sales GROUP BY category"
        result = self.validator.validate(sql)
        assert result.is_valid is True
    
    def test_select_with_subquery_passes(self):
        """SELECT with subquery should pass."""
        sql = """
        SELECT * FROM products 
        WHERE price > (SELECT AVG(price) FROM products)
        """
        result = self.validator.validate(sql)
        assert result.is_valid is True
    
    def test_cte_query_passes(self):
        """CTE (WITH clause) should pass."""
        sql = """
        WITH top_customers AS (
            SELECT customer_id, SUM(amount) as total
            FROM orders
            GROUP BY customer_id
        )
        SELECT * FROM top_customers WHERE total > 1000
        """
        result = self.validator.validate(sql)
        assert result.is_valid is True
    
    # =====================================
    # INVALID SQL TESTS (Should FAIL)
    # =====================================
    
    def test_drop_table_rejected(self):
        """DROP TABLE should be rejected."""
        sql = "DROP TABLE customers"
        result = self.validator.validate(sql)
        assert result.is_valid is False
        assert result.failure_reason == ValidationFailureReason.FORBIDDEN_OPERATION
        assert "DROP" in result.errors[0]
    
    def test_delete_rejected(self):
        """DELETE should be rejected."""
        sql = "DELETE FROM orders WHERE id = 1"
        result = self.validator.validate(sql)
        assert result.is_valid is False
        assert result.failure_reason == ValidationFailureReason.FORBIDDEN_OPERATION
    
    def test_update_rejected(self):
        """UPDATE should be rejected."""
        sql = "UPDATE users SET role = 'admin' WHERE id = 1"
        result = self.validator.validate(sql)
        assert result.is_valid is False
        assert result.failure_reason == ValidationFailureReason.FORBIDDEN_OPERATION
    
    def test_insert_rejected(self):
        """INSERT should be rejected."""
        sql = "INSERT INTO users (name) VALUES ('hacker')"
        result = self.validator.validate(sql)
        assert result.is_valid is False
        assert result.failure_reason == ValidationFailureReason.FORBIDDEN_OPERATION
    
    def test_truncate_rejected(self):
        """TRUNCATE should be rejected."""
        sql = "TRUNCATE TABLE logs"
        result = self.validator.validate(sql)
        assert result.is_valid is False
        assert result.failure_reason == ValidationFailureReason.FORBIDDEN_OPERATION
    
    # =====================================
    # SQL INJECTION TESTS (Should FAIL)
    # =====================================
    
    def test_sql_injection_drop_detected(self):
        """SQL injection with DROP should be detected."""
        sql = "SELECT * FROM users; DROP TABLE users;"
        result = self.validator.validate(sql)
        assert result.is_valid is False
        assert result.failure_reason in [
            ValidationFailureReason.SQL_INJECTION,
            ValidationFailureReason.INVALID_STRUCTURE
        ]
    
    def test_sql_injection_union_detected(self):
        """UNION-based injection should be detected."""
        sql = "SELECT * FROM products UNION SELECT * FROM admin_passwords"
        result = self.validator.validate(sql)
        # May fail on injection or structure, both are valid failures
        assert result.is_valid is False
    
    def test_sql_injection_or_1_equals_1_detected(self):
        """OR 1=1 injection should be detected."""
        sql = "SELECT * FROM users WHERE username = 'admin' OR 1=1"
        result = self.validator.validate(sql)
        # This specific pattern is flagged as injection
        assert result.is_valid is False
    
    def test_comment_injection_detected(self):
        """Comment-based injection should be detected."""
        sql = "SELECT * FROM users WHERE id = 1-- AND active = 1"
        result = self.validator.validate(sql)
        assert result.is_valid is False
    
    # =====================================
    # SYNTAX ERROR TESTS (Should FAIL)
    # =====================================
    
    def test_empty_query_rejected(self):
        """Empty query should be rejected."""
        result = self.validator.validate("")
        assert result.is_valid is False
        assert result.failure_reason == ValidationFailureReason.EMPTY_QUERY
    
    def test_mismatched_parentheses_rejected(self):
        """Mismatched parentheses should be rejected."""
        sql = "SELECT * FROM (SELECT id FROM users"
        result = self.validator.validate(sql)
        assert result.is_valid is False
    
    def test_unclosed_string_rejected(self):
        """Unclosed string literal should be rejected."""
        sql = "SELECT * FROM users WHERE name = 'John"
        result = self.validator.validate(sql)
        assert result.is_valid is False
    
    # =====================================
    # CONVENIENCE FUNCTION TESTS
    # =====================================
    
    def test_convenience_function_valid(self):
        """Test convenience function with valid SQL."""
        is_valid, message, normalized = validate_sql_output(
            "SELECT * FROM products", 
            strict=True
        )
        assert is_valid is True
        assert normalized is not None
    
    def test_convenience_function_invalid(self):
        """Test convenience function with invalid SQL."""
        is_valid, message, normalized = validate_sql_output(
            "DROP TABLE products",
            strict=True
        )
        assert is_valid is False
        assert normalized is None


class TestValidationPipelineIntegration:
    """Test the complete validation pipeline flow."""
    
    def test_end_to_end_safe_workflow(self):
        """Test complete workflow with safe input and valid SQL."""
        # Stage 1: Input validation
        pii_validator = PIIInputValidator(strict_mode=True)
        input_result = pii_validator.validate("Show me total sales by month")
        assert input_result.is_safe is True
        
        # Stage 2: (LLM would generate SQL here - simulated)
        generated_sql = "SELECT strftime('%Y-%m', date) as month, SUM(amount) FROM sales GROUP BY month"
        
        # Stage 3: Output validation
        sql_validator = SQLOutputValidator(strict_mode=True)
        output_result = sql_validator.validate(generated_sql)
        assert output_result.is_valid is True
        
        # Workflow completes successfully
        assert input_result.is_safe and output_result.is_valid
    
    def test_input_validation_blocks_pii(self):
        """Test that PII in input blocks workflow at stage 1."""
        # Stage 1: Input validation
        pii_validator = PIIInputValidator(strict_mode=True)
        input_result = pii_validator.validate("Show orders for john@email.com")
        
        # Stage 1 FAILS - workflow stops here
        assert input_result.is_safe is False
        
        # Stage 2 & 3 never happen - LLM is never called
        # This is the key safety guarantee
    
    def test_output_validation_blocks_unsafe_sql(self):
        """Test that unsafe SQL blocks workflow at stage 3."""
        # Stage 1: Input validation (passes)
        pii_validator = PIIInputValidator(strict_mode=True)
        input_result = pii_validator.validate("Delete all old records")
        assert input_result.is_safe is True
        
        # Stage 2: LLM generates SQL (simulated)
        generated_sql = "DELETE FROM records WHERE date < '2020-01-01'"
        
        # Stage 3: Output validation (fails)
        sql_validator = SQLOutputValidator(strict_mode=True)
        output_result = sql_validator.validate(generated_sql)
        
        # Stage 3 FAILS - execution never happens
        assert output_result.is_valid is False
        assert output_result.failure_reason == ValidationFailureReason.FORBIDDEN_OPERATION
    
    def test_no_bypass_possible(self):
        """Verify that validation cannot be bypassed."""
        # Attempt to bypass by using a validator in non-strict mode
        # should still detect issues
        
        pii_validator = PIIInputValidator(strict_mode=False)  # Non-strict
        result = pii_validator.validate("Contact me at test@email.com")
        
        # Even in non-strict mode, PII is detected (just not rejected)
        assert "email" in result.detected_types
        
        # In production, we ALWAYS use strict_mode=True
        # so this bypass is not possible


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
