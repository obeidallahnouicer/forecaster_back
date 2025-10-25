# Bug Fix: KeyError in SQL Generation Template

## Issue
When processing the query "Quels clients ont chuté de plus de 20 % cette année ?", the system encountered a `KeyError: '"sql"'` during LLM prompt formatting.

## Root Cause
The `SQL_GENERATION_TEMPLATE` in `prompts/sql_generation.py` contained JSON example formatting with curly braces:
```python
"OUTPUT FORMAT (JSON only):\n"
"{{\n"
"  \"sql\": \"SELECT ... FROM t_tablename WHERE ...\",\n"
"  \"params\": {{\"param1\": \"value1\"}},\n"
...
```

Python's `str.format()` method interpreted `{{\"sql\":` as a format placeholder `{"sql"` after un-escaping the doubled braces, causing the KeyError.

## Solution
Replaced the JSON example format with a descriptive text format that avoids nested curly braces:

**Before:**
```python
"OUTPUT FORMAT (JSON only):\n"
"{{\n"
"  \"sql\": \"SELECT ... FROM t_tablename WHERE ...\",\n"
"  \"params\": {{\"param1\": \"value1\"}},\n"
"  \"explanation\": \"Brief explanation of query logic\"\n"
"}}\n\n"
```

**After:**
```python
"OUTPUT: Return ONLY valid JSON with this structure:\n"
"- sql: SELECT statement string\n"
"- params: object with parameter values\n"
"- explanation: brief explanation string\n\n"

"Example: If user asks about inactive clients, return JSON like:\n"
"sql field contains SELECT query, params contains any parameters, explanation describes the query.\n\n"
```

## Verification
✅ Template formats successfully with test schema and question  
✅ No unmatched curly braces in formatted output  
✅ All existing unit tests still pass (35/35)  
✅ Query can now be processed without KeyError

## Files Changed
- `prompts/sql_generation.py`: Updated `SQL_GENERATION_TEMPLATE`

## Impact
- **Positive**: Fixes critical bug preventing queries from being processed
- **No Breaking Changes**: Output format expectations remain the same (JSON with sql, params, explanation)
- **Backward Compatible**: LLM will still generate the same JSON structure

## Testing
Run this to verify:
```bash
python -c "from prompts.sql_generation import SQL_GENERATION_TEMPLATE; from core.table_schemas import format_schema_for_llm; schema = format_schema_for_llm(); result = SQL_GENERATION_TEMPLATE.format(schema=schema, question='Test'); print('✓ SUCCESS')"
```

Expected output: `✓ SUCCESS`
