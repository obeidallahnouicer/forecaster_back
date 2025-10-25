import json

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
        },
        {
            "Client_Principal": "XYZ Retail",
            "Code_Client": "C456",
            "Intitule_client": "XYZ Commerce",
            "Categorie_Client": "Particulier",
            "Pays": "France",
            "Zone": "Sud",
            "Gouvernorat": "Marseille",
            "Marque": "Apple",
            "Famille": "Informatique",
            "Sous_Famille": "Ordinateurs",
            "Ref_Article": "ART002",
            "Designation": "MacBook Air",
            "Qte_Vendu": 10,
            "CA_HT_NET": 15000.00,
        },
    ]


def test_build_prompt_happy_path_contains_schema_and_samples():
    q = "Top 5 brands by sales in 2024"
    prompt = sg.build_sql_generation_prompt(q, sample_rows())
    assert isinstance(prompt, str)
    # must include the schema table name
    assert "ventes_cleann" in prompt
    # must reference CA_HT_NET mapping guidance
    assert "CA_HT_NET" in prompt
    # must include sample data values
    assert "ABC Group" in prompt
    assert "MacBook Air" in prompt


def test_build_prompt_no_samples_shows_placeholder():
    q = "Total sales in 2023"
    prompt = sg.build_sql_generation_prompt(q, [])
    assert "(no sample rows provided)" in prompt
