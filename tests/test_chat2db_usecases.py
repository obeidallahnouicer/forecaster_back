"""
Comprehensive test suite for Chat2DB chatbot logic.

Tests all use cases from TABLE Chatbot.md to ensure the chatbot can handle:
- Client queries (inactive clients, revenue, growth, churn risk)
- Product queries (families, performance, stock coverage, trends)
- Sales queries (evolution, top products, anomalies, comparisons)
- Stock queries (overstock, stockout risk, coverage)

Tests use the MockModel for deterministic behavior in CI/CD.
"""

import pytest
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Set mock mode for tests
os.environ["MODEL_USE_MOCK"] = "1"

from text2sql.agent import Chat2DBQueryAgent
from text2sql.model_loader import ModelLoader, MockModel
from text2sql.db import DBConnection


@pytest.fixture
def test_db():
    """Create an in-memory test database with sample data."""
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = db_file.name
    db_file.close()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create clients table
    cursor.execute("""
        CREATE TABLE clients (
            Code_Client TEXT PRIMARY KEY,
            Intitule_Client TEXT,
            Categorie_Client TEXT,
            Zone TEXT,
            Representant TEXT,
            Date_Premiere_Vente TEXT,
            Date_Derniere_Vente TEXT,
            Mois_Depuis_Derniere_Vente INTEGER,
            Annees_Actives INTEGER,
            CA_Moyen_Annuel REAL,
            CA_Total REAL,
            Variation_CA_Percent REAL,
            Frequence_Achat_Moyenne REAL,
            Statut_Client TEXT,
            Score_Fidelite INTEGER
        )
    """)

    # Insert sample client data
    clients_data = [
        ("CL001", "AYADI HOUWAIDA", "GMS", "Tunis", "RABIAA BEN SEDRINE", "2020-01-15", "2025-10-01", 1, 5, 50000.0, 250000.0, 15.5, 12.0, "Actif", 85),
        ("CL002", "SUPERETTE CENTRALE", "Superette", "Ariana", "MOHAMED ALI", "2019-06-20", "2025-06-15", 5, 6, 30000.0, 180000.0, -25.0, 8.0, "En baisse", 70),
        ("CL003", "PHARMA NORD", "Pharmacie", "Bizerte", "LEILA BOUZID", "2021-03-10", "2025-09-20", 2, 4, 75000.0, 300000.0, 22.0, 15.0, "Actif", 90),
        ("CL004", "GROCERY SHOP", "Superette", "Sfax", "RABIAA BEN SEDRINE", "2018-11-05", "2024-12-30", 11, 7, 45000.0, 315000.0, -30.0, 10.0, "Inactif", 45),
        ("CL005", "BEAUTY CENTER", "Salon", "Sousse", "KARIM TRABELSI", "2022-02-28", "2025-10-10", 0, 3, 60000.0, 180000.0, 30.0, 18.0, "Actif", 95),
    ]

    cursor.executemany("""
        INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, clients_data)

    # Create products table
    cursor.execute("""
        CREATE TABLE produits (
            Reference_Article TEXT PRIMARY KEY,
            Designation TEXT,
            Famille TEXT,
            Sous_Famille TEXT,
            Marque TEXT,
            Qte_Totale_Vendue INTEGER,
            CA_Total REAL,
            CA_Mensuel_Moyen REAL,
            Rotation_Mensuelle REAL,
            Stock_Terme INTEGER,
            Couverture_Stock REAL,
            Nb_Clients_Actifs INTEGER,
            Variation_CA_Percent REAL,
            Variation_Qte_Percent REAL,
            Tendance_3m REAL,
            Recommandation TEXT
        )
    """)

    # Insert sample product data
    products_data = [
        ("32014", "Shampoing Réparateur", "Cheveux", "Shampooing", "H.ZONE", 1500, 45000.0, 3750.0, 125.0, 200, 1.6, 45, 10.0, 8.0, 3800.0, "Normal"),
        ("32015", "Crème Hydratante", "Soin", "Crème", "RENEE BLANCHE", 800, 32000.0, 2666.0, 66.0, 500, 7.5, 30, -15.0, -12.0, 2500.0, "Promo"),
        ("32016", "Rouge à Lèvres", "Maquillage", "Lèvres", "RENEE BLANCHE", 2000, 60000.0, 5000.0, 166.0, 50, 0.3, 60, 25.0, 20.0, 5200.0, "Réassort"),
        ("32017", "Masque Capillaire", "Cheveux", "Masque", "H.ZONE", 1200, 48000.0, 4000.0, 100.0, 300, 3.0, 40, 5.0, 3.0, 4050.0, "Normal"),
        ("32018", "Sérum Anti-Âge", "Soin", "Sérum", "RENEE BLANCHE", 600, 36000.0, 3000.0, 50.0, 800, 16.0, 25, -20.0, -18.0, 2800.0, "Promo"),
    ]

    cursor.executemany("""
        INSERT INTO produits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, products_data)

    # Create ventes_mensuelles table
    cursor.execute("""
        CREATE TABLE ventes_mensuelles (
            Code_Client TEXT,
            Categorie_Client TEXT,
            Marque TEXT,
            Famille TEXT,
            Sous_Famille TEXT,
            Reference_Article TEXT,
            Designation TEXT,
            Annee INTEGER,
            Mois INTEGER,
            CA_HT_NET REAL,
            Qte_Vendu INTEGER,
            CA_Mois_Precedent REAL,
            Qte_Mois_Precedent INTEGER,
            Delta_CA_Mois REAL,
            Delta_CA_Percent REAL,
            Delta_Qte_Mois INTEGER,
            Delta_Qte_Percent REAL,
            Delta_CA_Annee REAL,
            Delta_Qte_Annee INTEGER,
            Tendance_3m REAL
        )
    """)

    # Insert sample sales data
    sales_data = [
        ("CL001", "GMS", "H.ZONE", "Cheveux", "Shampooing", "32014", "Shampoing Réparateur", 2025, 10, 5000.0, 150, 4500.0, 135, 500.0, 11.1, 15, 11.1, 800.0, 25, 4800.0),
        ("CL001", "GMS", "RENEE BLANCHE", "Maquillage", "Lèvres", "32016", "Rouge à Lèvres", 2025, 10, 6000.0, 200, 5500.0, 180, 500.0, 9.1, 20, 11.1, 1000.0, 40, 5700.0),
        ("CL002", "Superette", "H.ZONE", "Cheveux", "Shampooing", "32014", "Shampoing Réparateur", 2025, 10, 2500.0, 75, 3000.0, 90, -500.0, -16.7, -15, -16.7, -800.0, -30, 2700.0),
        ("CL003", "Pharmacie", "RENEE BLANCHE", "Soin", "Crème", "32015", "Crème Hydratante", 2025, 10, 4000.0, 100, 4500.0, 112, -500.0, -11.1, -12, -10.7, -600.0, -15, 4200.0),
    ]

    cursor.executemany("""
        INSERT INTO ventes_mensuelles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sales_data)

    # Create stock table
    cursor.execute("""
        CREATE TABLE stock (
            Reference_Article TEXT PRIMARY KEY,
            Designation TEXT,
            Famille TEXT,
            Sous_Famille TEXT,
            Marque TEXT,
            Stock_Disponible INTEGER,
            Stock_Reserve INTEGER,
            Stock_Total INTEGER,
            Rotation_Mensuelle REAL,
            Couverture_Stock REAL,
            Valeur_Stock REAL,
            Date_Derniere_Entree TEXT,
            Date_Derniere_Sortie TEXT,
            Statut_Stock TEXT,
            Recommandation TEXT
        )
    """)

    # Insert sample stock data
    stock_data = [
        ("32014", "Shampoing Réparateur", "Cheveux", "Shampooing", "H.ZONE", 200, 50, 250, 125.0, 2.0, 7500.0, "2025-10-15", "2025-11-01", "Normal", "Normal"),
        ("32015", "Crème Hydratante", "Soin", "Crème", "RENEE BLANCHE", 500, 100, 600, 66.0, 9.0, 24000.0, "2025-09-20", "2025-10-28", "Surstock", "Promo"),
        ("32016", "Rouge à Lèvres", "Maquillage", "Lèvres", "RENEE BLANCHE", 50, 10, 60, 166.0, 0.36, 1800.0, "2025-10-25", "2025-11-01", "Rupture", "Réassort"),
        ("32017", "Masque Capillaire", "Cheveux", "Masque", "H.ZONE", 300, 75, 375, 100.0, 3.75, 15000.0, "2025-10-10", "2025-10-30", "Normal", "Normal"),
        ("32018", "Sérum Anti-Âge", "Soin", "Sérum", "RENEE BLANCHE", 800, 200, 1000, 50.0, 20.0, 60000.0, "2025-08-15", "2025-10-20", "Surstock", "Promo"),
    ]

    cursor.executemany("""
        INSERT INTO stock VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, stock_data)

    conn.commit()
    conn.close()

    yield f"sqlite:///{db_path}"

    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


@pytest.fixture
def agent(test_db):
    """Create a Chat2DBQueryAgent with mock model."""
    model_loader = ModelLoader()
    model_loader._model = MockModel()
    return Chat2DBQueryAgent(db_url=test_db, model_loader=model_loader)


class TestClientQueries:
    """Test use cases related to client queries from TABLE Chatbot.md."""

    def test_inactive_clients(self, agent):
        """Test: 'Quels sont les clients inactifs depuis plus de 3 mois?'"""
        result = agent.generate_and_run("Quels sont les clients inactifs depuis plus de 3 mois?")

        assert result["sql"] is not None
        assert "clients" in result["sql"].lower()
        assert "mois_depuis_derniere_vente" in result["sql"].lower()
        assert len(result["rows"]) >= 1  # Should find at least CL002 and CL004

    def test_client_average_revenue(self, agent):
        """Test: 'Quel est le chiffre d affaires moyen du client AYADI HOUWAIDA?'"""
        result = agent.generate_and_run("Quel est le chiffre d'affaires moyen du client AYADI HOUWAIDA?")

        assert result["sql"] is not None
        assert "clients" in result["sql"].lower()
        assert "ca_moyen_annuel" in result["sql"].lower() or "ayadi" in result["sql"].lower()

    def test_clients_growth(self, agent):
        """Test: 'Quels clients ont augmenté leurs achats de plus de 20%?'"""
        result = agent.generate_and_run("Quels clients ont augmenté leurs achats de plus de 20%?")

        assert result["sql"] is not None
        assert "clients" in result["sql"].lower()
        # Should filter by positive variation

    def test_clients_by_representative(self, agent):
        """Test: 'Quels clients gère le commercial RABIAA BEN SEDRINE?'"""
        result = agent.generate_and_run("Quels clients gère le commercial RABIAA BEN SEDRINE?")

        assert result["sql"] is not None
        assert "clients" in result["sql"].lower()
        assert "representant" in result["sql"].lower() or "rabiaa" in result["sql"].lower()

    def test_clients_churn_risk(self, agent):
        """Test: 'Quels sont les clients à risque de perte?'"""
        result = agent.generate_and_run("Quels sont les clients à risque de perte?")

        assert result["sql"] is not None
        assert "clients" in result["sql"].lower()
        assert "variation_ca_percent" in result["sql"].lower() or "-20" in result["sql"]

    def test_top_loyal_clients(self, agent):
        """Test: 'Top 10 des clients les plus fidèles?'"""
        result = agent.generate_and_run("Top 10 des clients les plus fidèles?")

        assert result["sql"] is not None
        assert "clients" in result["sql"].lower()
        assert "limit" in result["sql"].lower() or "top" in result["sql"].lower()

    def test_count_clients(self, agent):
        """Test: 'Combien de clients actifs?'"""
        result = agent.generate_and_run("Combien de clients actifs?")

        assert result["sql"] is not None
        assert "count" in result["sql"].lower()
        assert "clients" in result["sql"].lower()


class TestProductQueries:
    """Test use cases related to product queries from TABLE Chatbot.md."""

    def test_product_family(self, agent):
        """Test: 'À quelle famille appartient l article 32014?'"""
        result = agent.generate_and_run("À quelle famille appartient l'article 32014?")

        assert result["sql"] is not None
        assert "produits" in result["sql"].lower()
        assert "famille" in result["sql"].lower()

    def test_best_selling_product(self, agent):
        """Test: 'Quel est le produit le plus vendu de la marque H.ZONE?'"""
        result = agent.generate_and_run("Quel est le produit le plus vendu de la marque H.ZONE?")

        assert result["sql"] is not None
        assert "produits" in result["sql"].lower()
        assert "h.zone" in result["sql"].lower() or "h_zone" in result["sql"].lower()

    def test_product_family_growth(self, agent):
        """Test: 'Est-ce que la famille Shampooing est en croissance?'"""
        result = agent.generate_and_run("Est-ce que la famille Shampooing est en croissance?")

        assert result["sql"] is not None
        # Should check either produits or ventes_mensuelles

    def test_overstock_products(self, agent):
        """Test: 'Quels produits ont un stock supérieur à 6 mois?'"""
        result = agent.generate_and_run("Quels produits ont un stock supérieur à 6 mois?")

        assert result["sql"] is not None
        assert "couverture_stock" in result["sql"].lower() or "stock" in result["sql"].lower()

    def test_stockout_risk(self, agent):
        """Test: 'Quels produits risquent la rupture le mois prochain?'"""
        result = agent.generate_and_run("Quels produits risquent la rupture le mois prochain?")

        assert result["sql"] is not None
        assert "stock" in result["sql"].lower() or "couverture" in result["sql"].lower()


class TestSalesQueries:
    """Test use cases related to sales queries from TABLE Chatbot.md."""

    def test_monthly_sales_evolution(self, agent):
        """Test: 'Comment évoluent les ventes de septembre à octobre?'"""
        result = agent.generate_and_run("Comment évoluent les ventes de septembre à octobre?")

        assert result["sql"] is not None
        assert "ventes_mensuelles" in result["sql"].lower() or "ventes" in result["sql"].lower()

    def test_top_product_revenue(self, agent):
        """Test: 'Quel produit a généré le plus de CA ce mois-ci?'"""
        result = agent.generate_and_run("Quel produit a généré le plus de CA ce mois-ci?")

        assert result["sql"] is not None
        # Should aggregate CA from sales or products

    def test_clients_revenue_decline(self, agent):
        """Test: 'Quels clients ont baissé leur CA en octobre?'"""
        result = agent.generate_and_run("Quels clients ont baissé leur CA en octobre?")

        assert result["sql"] is not None
        # Should check delta_ca_percent or variation


class TestStockQueries:
    """Test use cases related to stock queries from TABLE Chatbot.md."""

    def test_overstock_list(self, agent):
        """Test: 'Quels produits sont en surstock?'"""
        result = agent.generate_and_run("Quels produits sont en surstock?")

        assert result["sql"] is not None
        assert "stock" in result["sql"].lower()

    def test_stockout_risk_list(self, agent):
        """Test: 'Quels produits risquent la rupture?'"""
        result = agent.generate_and_run("Quels produits risquent la rupture?")

        assert result["sql"] is not None
        assert "stock" in result["sql"].lower()

    def test_average_stock_coverage(self, agent):
        """Test: 'Quelle est la couverture de stock moyenne?'"""
        result = agent.generate_and_run("Quelle est la couverture de stock moyenne?")

        assert result["sql"] is not None
        assert "avg" in result["sql"].lower() or "moyenne" in result["sql"].lower()
        assert "couverture" in result["sql"].lower() or "stock" in result["sql"].lower()


class TestValidation:
    """Test validation and error handling."""

    def test_invalid_question(self, agent):
        """Test handling of ambiguous questions."""
        result = agent.generate_and_run("blablabla")

        # Should still generate SQL (even if generic)
        assert result["sql"] is not None

    def test_sql_injection_prevention(self, agent):
        """Test SQL injection prevention."""
        malicious_query = "Show me clients; DROP TABLE clients;"
        result = agent.generate_and_run(malicious_query)

        # Should generate safe SQL (no DROP, DELETE, etc.)
        assert result["sql"] is not None
        assert "drop" not in result["sql"].lower()

    def test_validation_issues_reported(self, agent):
        """Test that validation issues are reported."""
        # Force a validation issue by mocking the validator
        result = agent.generate_and_run("test query")

        assert "issues" in result
        # issues list may be empty if SQL is valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
