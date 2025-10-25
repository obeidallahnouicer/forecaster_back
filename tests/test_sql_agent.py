from llm import sql_agent
from prompts import sql_generation as sg


def sample_rows():
    return [
        {
            "Client_Principal": "ABC Group",
            "Code_Client": "C123",
            "Intitule_client": "ABC Distribution",
            "Categorie_Client": "Entreprise",
            "Pays": "France",
            "Zone": "Nord",
            "Gouvernorat": "Lille",
            "Marque": "Samsung",
            "Famille": "Électroménager",
            "Sous_Famille": "TV",
            "Ref_Article": "ART001",
            "Designation": 'Smart TV 50"',
            "Qte_Vendu": 25,
            "CA_HT_NET": 12500.00,
        }
    ]


def test_generate_sql_from_question_returns_parsable_structure_when_llm_fallback():
    q = "Top brands by sales"
    resp = sql_agent.generate_sql_from_question(q, sample_rows())
    # Since GROQ_API_KEY is likely not set in CI/dev, get_llm_response returns a fallback string
    # which is not valid JSON—adapter should return an explanation indicating parse failure
    assert "explanation" in resp
    assert resp["explanation"] == "LLM did not return valid JSON"
    assert "raw" in resp and isinstance(resp["raw"], str)
