from text2sql.validator import validate_and_autocorrect


def test_validate_simple_select():
    schema = {"singer": ["Singer_ID", "Name"], "stadium": ["Stadium_ID", "Name"]}
    sql = "SELECT COUNT(*) FROM singer;"
    corrected, issues = validate_and_autocorrect(sql, schema)
    assert "Unknown identifier" not in " ".join(issues)


def test_autocorrect_column():
    schema = {"singer": ["Singer_ID", "Name"]}
    # misspelled column 'Snger_ID' should be autocorrected if fuzzy libs are available
    sql = "SELECT Snger_ID FROM singer;"
    corrected, issues = validate_and_autocorrect(sql, schema, fuzzy_threshold=0)
    # With threshold 0 the function will autocorrect via naive substring logic
    assert "Snger_ID" not in corrected
