"""
Quick validation test for the fixed prompt template
"""

def test_prompt_template_format():
    """Ensure prompt template can be formatted without errors"""
    from prompts.sql_generation import SQL_GENERATION_TEMPLATE
    from core.table_schemas import format_schema_for_llm
    
    # Get schema
    schema = format_schema_for_llm()
    
    # Test formatting the template
    question = "Quels clients ont chuté de plus de 20 % cette année ?"
    
    # This should not raise KeyError
    try:
        formatted = SQL_GENERATION_TEMPLATE.format(
            schema=schema,
            question=question
        )
        print("✓ Template formatted successfully")
        print(f"✓ Template length: {len(formatted)} characters")
        
        # Verify no stray curly braces
        import re
        # Find any unmatched curly braces (not part of format variables)
        unmatched = re.findall(r'\{[^}]*\}', formatted)
        if unmatched:
            print(f"⚠ Warning: Found {len(unmatched)} potential format placeholders in output")
        else:
            print("✓ No unmatched curly braces in formatted output")
        
        return True
    except KeyError as e:
        print(f"✗ KeyError: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_prompt_template_format()
    exit(0 if success else 1)
