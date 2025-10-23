"""
SQL Generation Prompts

LangChain prompt templates for converting natural language to SQL.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

SQL_SYSTEM_PROMPT = """You are an expert SQL query generator for business intelligence.

CRITICAL RULES:
1. Generate ONLY SELECT queries (no INSERT/UPDATE/DELETE/DROP)
2. Use SQLite syntax exclusively
3. Use named parameters (:param_name) for all user-provided values
4. Always include appropriate JOINs when querying multiple tables
5. Add ORDER BY and LIMIT clauses for better UX
6. Return ONLY valid JSON with this exact structure:
   {{"sql": "SELECT ...", "params": {{"key": "value"}}, "explanation": "Brief explanation"}}

BEST PRACTICES:
- Use DISTINCT when counting unique values
- Use strftime() for date operations in SQLite
- Add meaningful column aliases (AS alias_name)
- Prefer INNER JOIN over implicit joins
- Always validate column names against the schema
"""

SQL_GENERATION_TEMPLATE = """You are generating SQL for a business intelligence database.

AVAILABLE SCHEMA (only use these tables/columns):
{schema}

IMPORTANT: Use the above schema to validate all table and column names. If the user's question references a column or table not present in the schema, respond with a JSON object where "sql" is an empty string and "explanation" clearly states which column/table is invalid. Do NOT guess or invent column names.

FEW-SHOT EXAMPLES:

Example 1:
Question: "How many unique customers purchased in 2024?"
Output:
{{
  "sql": "SELECT COUNT(DISTINCT Code_Client) as customer_count FROM t_base WHERE strftime('%Y', Date) = :year",
  "params": {{"year": "2024"}},
  "explanation": "Counts distinct customers with purchases in 2024 using date filtering"
}}

Example 2:
Question: "Top 5 products by revenue"
Output:
{{
  "sql": "SELECT Ref_Article, Designation, SUM(CA_HT_NET) as total_revenue FROM t_base GROUP BY Ref_Article, Designation ORDER BY total_revenue DESC LIMIT 5",
  "params": {{}},
  "explanation": "Aggregates revenue by product, sorted descending, limited to top 5"
}}

Example 3:
Question: "Products with stock below 50 units"
Output:
{{
  "sql": "SELECT Référence_Article, Désignation, Stock_à_terme FROM t_stock WHERE Stock_à_terme < :threshold ORDER BY Stock_à_terme ASC",
  "params": {{"threshold": 50}},
  "explanation": "Filters low-stock products, sorted by stock level"
}}

Example 4:
Question: "Monthly revenue trend for 2024"
Output:
{{
  "sql": "SELECT strftime('%Y-%m', Date) as month, SUM(CA_HT_NET) as monthly_revenue FROM t_base WHERE strftime('%Y', Date) = :year GROUP BY month ORDER BY month",
  "params": {{"year": "2024"}},
  "explanation": "Aggregates revenue by month for trend analysis"
}}

Now generate SQL for the following question:

USER QUESTION: {question}

FILES AND COLUMNS (machine-readable):
{file_schema}

Output ONLY the JSON object (no markdown, no code blocks):"""

# LangChain PromptTemplate
SQL_GENERATION_PROMPT = PromptTemplate(
  input_variables=["schema", "question", "file_schema"],
    template=SQL_GENERATION_TEMPLATE
)

# ChatPromptTemplate for chat models
SQL_GENERATION_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM_PROMPT),
    ("human", SQL_GENERATION_TEMPLATE)
])
