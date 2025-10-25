"""
Comprehensive Unit Tests for Chatbot Module

Tests cover all use cases from TABLE Chatbot.md document:
- CLIENT queries (inactive, CA, growth, risk, fidelity)
- PRODUCT queries (performance, stock, trends, recommendations)
- VENTES_MENSUELLES queries (evolution, comparisons, trends)
- STOCK queries (coverage, rupture, surstock)
- Business Rules validation
- KPI calculations
"""

import pytest
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Import modules to test
from core.business_rules import (
    BusinessRulesEngine,
    EntityType,
    AlertLevel,
    get_business_rules_engine
)
from core.table_schemas import (
    get_schema,
    format_schema_for_llm,
    infer_entity_type,
    get_column_list
)


class TestBusinessRulesEngine:
    """Test Business Rules Engine calculations and recommendations"""
    
    def test_client_inactive_rule(self):
        """Test: Client inactive >= 3 months triggers recommendation"""
        engine = get_business_rules_engine()
        
        data = {
            "mois_depuis_derniere_vente": 4,
            "ca_total": 50000
        }
        
        recommendations = engine.evaluate_rules(EntityType.CLIENT, data)
        
        assert len(recommendations) > 0
        inactive_rec = [r for r in recommendations if r["rule_name"] == "client_inactive"]
        assert len(inactive_rec) == 1
        assert inactive_rec[0]["alert_level"] == AlertLevel.HIGH.value
        assert "relance" in inactive_rec[0]["recommendation"].lower()
    
    def test_client_ca_decline_rule(self):
        """Test: Client CA decline > 10% triggers alert"""
        engine = get_business_rules_engine()
        
        data = {
            "delta_ca_pct": -15.5,
            "ca_total": 30000
        }
        
        recommendations = engine.evaluate_rules(EntityType.CLIENT, data)
        
        decline_rec = [r for r in recommendations if r["rule_name"] == "client_ca_decline"]
        assert len(decline_rec) == 1
        assert decline_rec[0]["alert_level"] == AlertLevel.HIGH.value
    
    def test_client_high_growth_rule(self):
        """Test: Client growth > 20% triggers consolidation recommendation"""
        engine = get_business_rules_engine()
        
        data = {
            "delta_ca_pct": 25.0,
            "ca_total": 75000
        }
        
        recommendations = engine.evaluate_rules(EntityType.CLIENT, data)
        
        growth_rec = [r for r in recommendations if r["rule_name"] == "client_high_growth"]
        assert len(growth_rec) == 1
        assert "consolider" in growth_rec[0]["recommendation"].lower()
    
    def test_client_at_risk_rule(self):
        """Test: Client with CA decline + low frequency = at risk"""
        engine = get_business_rules_engine()
        
        data = {
            "delta_ca_pct": -18.0,
            "frequence_achat_moyenne": 1.5,
            "ca_total": 40000
        }
        
        recommendations = engine.evaluate_rules(EntityType.CLIENT, data)
        
        risk_rec = [r for r in recommendations if r["rule_name"] == "client_at_risk"]
        assert len(risk_rec) == 1
        assert risk_rec[0]["alert_level"] == AlertLevel.CRITICAL.value
        assert "risque" in risk_rec[0]["recommendation"].lower()
    
    def test_product_stock_critical_rule(self):
        """Test: Coverage < 1 month = critical stock alert"""
        engine = get_business_rules_engine()
        
        data = {
            "couverture_stock": 0.8,
            "stock_terme": 50
        }
        
        recommendations = engine.evaluate_rules(EntityType.PRODUCT, data)
        
        critical_rec = [r for r in recommendations if r["rule_name"] == "product_stock_critical"]
        assert len(critical_rec) == 1
        assert critical_rec[0]["alert_level"] == AlertLevel.CRITICAL.value
        assert "réassortir" in critical_rec[0]["recommendation"].lower()
    
    def test_product_overstock_rule(self):
        """Test: Coverage > 6 months = overstock alert"""
        engine = get_business_rules_engine()
        
        data = {
            "couverture_stock": 8.5,
            "stock_terme": 1000
        }
        
        recommendations = engine.evaluate_rules(EntityType.PRODUCT, data)
        
        overstock_rec = [r for r in recommendations if r["rule_name"] == "product_overstock"]
        assert len(overstock_rec) == 1
        assert "promotion" in overstock_rec[0]["recommendation"].lower() or \
               "déstockage" in overstock_rec[0]["recommendation"].lower()
    
    def test_product_sales_decline_rule(self):
        """Test: Product quantity decline > 15% triggers investigation"""
        engine = get_business_rules_engine()
        
        data = {
            "variation_qte_pct": -18.0,
            "qte_totale_vendue": 500
        }
        
        recommendations = engine.evaluate_rules(EntityType.PRODUCT, data)
        
        decline_rec = [r for r in recommendations if r["rule_name"] == "product_sales_decline"]
        assert len(decline_rec) == 1
        assert "enquête" in decline_rec[0]["recommendation"].lower()
    
    def test_product_high_demand_rule(self):
        """Test: High demand + low stock = urgent restock"""
        engine = get_business_rules_engine()
        
        data = {
            "variation_qte_pct": 25.0,
            "couverture_stock": 2.0
        }
        
        recommendations = engine.evaluate_rules(EntityType.PRODUCT, data)
        
        demand_rec = [r for r in recommendations if r["rule_name"] == "product_high_demand"]
        assert len(demand_rec) == 1
        assert "stock" in demand_rec[0]["recommendation"].lower()
    
    def test_kpi_ca_moyen_annuel(self):
        """Test: KPI calculation for average annual CA"""
        engine = get_business_rules_engine()
        
        rows = [
            {"ca_ht_net": 10000, "annee": 2023},
            {"ca_ht_net": 12000, "annee": 2023},
            {"ca_ht_net": 15000, "annee": 2024},
            {"ca_ht_net": 18000, "annee": 2024},
        ]
        
        kpis = engine.calculate_kpis(rows, ["ca_moyen_annuel"])
        
        assert "ca_moyen_annuel" in kpis
        # Should be (10000+12000+15000+18000) / 2 years = 27500
        assert kpis["ca_moyen_annuel"]["value"] == pytest.approx(27500.0, rel=0.1)
    
    def test_kpi_couverture_stock(self):
        """Test: Stock coverage calculation"""
        engine = get_business_rules_engine()
        
        rows = [
            {
                "stock_terme": 600,
                "rotation_mensuelle": 100
            }
        ]
        
        kpis = engine.calculate_kpis(rows, ["couverture_stock"])
        
        assert "couverture_stock" in kpis
        assert kpis["couverture_stock"]["value"] == 6.0
    
    def test_generate_insights_client(self):
        """Test: Complete insights generation for client entity"""
        engine = get_business_rules_engine()
        
        rows = [
            {
                "code_client": "CL001",
                "ca_ht_net": 5000,
                "annee": 2024,
                "mois": 10
            },
            {
                "code_client": "CL001",
                "ca_ht_net": 4000,
                "annee": 2024,
                "mois": 9
            }
        ]
        
        insights = engine.generate_insights(
            entity_type=EntityType.CLIENT,
            rows=rows,
            question="Analyser le client CL001"
        )
        
        assert "kpis" in insights
        assert "recommendations" in insights
        assert "summary" in insights
        assert insights["row_count"] == 2
    
    def test_no_false_positives(self):
        """Test: Healthy data should not trigger alerts"""
        engine = get_business_rules_engine()
        
        # Healthy client data
        data = {
            "mois_depuis_derniere_vente": 1,
            "delta_ca_pct": 5.0,
            "frequence_achat_moyenne": 3.0,
            "ca_total": 50000
        }
        
        recommendations = engine.evaluate_rules(EntityType.CLIENT, data)
        
        # Should only get positive growth recommendation if any
        assert len(recommendations) == 0 or all(
            r["alert_level"] not in [AlertLevel.CRITICAL.value, AlertLevel.HIGH.value]
            for r in recommendations
        )


class TestTableSchemas:
    """Test table schema definitions and utilities"""
    
    def test_get_client_schema(self):
        """Test: CLIENT schema is defined correctly"""
        schema = get_schema("clients")
        
        assert schema is not None
        assert schema.logical_name == "clients"
        assert schema.primary_key == "Code_Client"
        
        column_names = [col.name for col in schema.columns]
        assert "Code_Client" in column_names
        assert "Intitulé_Client" in column_names
        assert "CA_Total" in column_names
        assert "Statut_Client" in column_names
        assert "Mois_Depuis_Derniere_Vente" in column_names
    
    def test_get_product_schema(self):
        """Test: PRODUITS schema is defined correctly"""
        schema = get_schema("produits")
        
        assert schema is not None
        assert schema.logical_name == "produits"
        assert schema.primary_key == "Référence_Article"
        
        column_names = [col.name for col in schema.columns]
        assert "Référence_Article" in column_names
        assert "Couverture_Stock" in column_names
        assert "Rotation_Mensuelle" in column_names
        assert "Recommandation" in column_names
    
    def test_format_schema_for_llm(self):
        """Test: Schema formatting for LLM prompts"""
        formatted = format_schema_for_llm(["clients", "produits"])
        
        assert "TABLE: clients" in formatted
        assert "TABLE: produits" in formatted
        assert "Code_Client" in formatted
        assert "Référence_Article" in formatted
        assert "Règles métier" in formatted or "business" in formatted.lower()
    
    def test_infer_entity_type_client(self):
        """Test: Entity type inference for client questions"""
        questions = [
            "Quels sont les clients inactifs ?",
            "Top 10 des clients les plus fidèles",
            "Quel commercial gère ce client ?",
            "Liste des clients à risque de perte"
        ]
        
        for q in questions:
            entity = infer_entity_type(q)
            assert entity == "clients", f"Failed for: {q}"
    
    def test_infer_entity_type_product(self):
        """Test: Entity type inference for product questions"""
        questions = [
            "Quel est le produit le plus vendu de H.ZONE ?",
            "Quels articles nécessitent une promotion ?",
            "La famille Shampooing est-elle en croissance ?",
            "Liste des produits RENEE BLANCHE"
        ]
        
        for q in questions:
            entity = infer_entity_type(q)
            assert entity == "produits", f"Failed for: {q}"
    
    def test_infer_entity_type_stock(self):
        """Test: Entity type inference for stock questions"""
        questions = [
            "Quels produits ont un stock supérieur à 6 mois ?",
            "Liste des articles en rupture",
            "Quelle est la couverture de stock moyenne ?",
            "Produits qui risquent la rupture"
        ]
        
        for q in questions:
            entity = infer_entity_type(q)
            assert entity in ["stock", "produits"], f"Failed for: {q}"
    
    def test_infer_entity_type_sales(self):
        """Test: Entity type inference for sales questions"""
        questions = [
            "Quel est le CA total du mois dernier ?",
            "Compare les ventes de septembre et octobre",
            "Évolution du chiffre d'affaires",
        ]
        
        for q in questions:
            entity = infer_entity_type(q)
            assert entity in ["ventes_mensuelles", "general"], f"Failed for: {q}"
        
        # "Top 5 des familles" can be inferred as products or sales - both are valid
        family_question = "Top 5 des familles en croissance"
        entity = infer_entity_type(family_question)
        assert entity in ["ventes_mensuelles", "general", "produits"], \
            f"Failed for: {family_question}"
    
    def test_get_column_list(self):
        """Test: Column list extraction"""
        columns = get_column_list("clients")
        
        assert len(columns) > 0
        assert "Code_Client" in columns
        assert "CA_Total" in columns


class TestUseCaseScenarios:
    """Integration tests for complete use case scenarios"""
    
    def test_scenario_inactive_clients(self):
        """
        Use Case: "Quels sont les clients inactifs depuis plus de 3 mois ?"
        Expected: Filter by Mois_Depuis_Derniere_Vente >= 3
        """
        engine = get_business_rules_engine()
        
        clients = [
            {"code_client": "CL001", "mois_depuis_derniere_vente": 5, "ca_total": 20000},
            {"code_client": "CL002", "mois_depuis_derniere_vente": 1, "ca_total": 30000},
            {"code_client": "CL003", "mois_depuis_derniere_vente": 6, "ca_total": 15000},
        ]
        
        inactive_clients = []
        for client in clients:
            recs = engine.evaluate_rules(EntityType.CLIENT, client)
            inactive_recs = [r for r in recs if r["rule_name"] == "client_inactive"]
            if inactive_recs:
                inactive_clients.append(client)
        
        assert len(inactive_clients) == 2
        assert all(c["mois_depuis_derniere_vente"] >= 3 for c in inactive_clients)
    
    def test_scenario_client_ca_average(self):
        """
        Use Case: "Quel est le chiffre d'affaires moyen du client AYADI HOUWAIDA ?"
        Expected: Calculate CA_Moyen_Annuel
        """
        engine = get_business_rules_engine()
        
        client_sales = [
            {"ca_ht_net": 5000, "annee": 2023, "mois": 1},
            {"ca_ht_net": 6000, "annee": 2023, "mois": 2},
            {"ca_ht_net": 7000, "annee": 2024, "mois": 1},
            {"ca_ht_net": 8000, "annee": 2024, "mois": 2},
        ]
        
        kpis = engine.calculate_kpis(client_sales, ["ca_moyen_annuel"])
        
        # Total: 26000 / 2 years = 13000
        assert kpis["ca_moyen_annuel"]["value"] == pytest.approx(13000.0, rel=0.1)
    
    def test_scenario_client_growth_20pct(self):
        """
        Use Case: "Quels clients ont augmenté leurs achats de plus de 20% ?"
        Expected: Filter by delta_ca_pct > 20
        """
        engine = get_business_rules_engine()
        
        clients = [
            {"code_client": "CL001", "delta_ca_pct": 25.0},
            {"code_client": "CL002", "delta_ca_pct": 10.0},
            {"code_client": "CL003", "delta_ca_pct": 30.0},
        ]
        
        growing_clients = []
        for client in clients:
            recs = engine.evaluate_rules(EntityType.CLIENT, client)
            growth_recs = [r for r in recs if r["rule_name"] == "client_high_growth"]
            if growth_recs:
                growing_clients.append(client)
        
        assert len(growing_clients) == 2
        assert all(c["delta_ca_pct"] > 20 for c in growing_clients)
    
    def test_scenario_product_stock_coverage_6months(self):
        """
        Use Case: "Quels produits ont un stock supérieur à 6 mois ?"
        Expected: Filter by Couverture_Stock > 6
        """
        engine = get_business_rules_engine()
        
        products = [
            {"ref_article": "P001", "couverture_stock": 8.5, "stock_terme": 1000},
            {"ref_article": "P002", "couverture_stock": 3.0, "stock_terme": 300},
            {"ref_article": "P003", "couverture_stock": 12.0, "stock_terme": 2400},
        ]
        
        overstock_products = []
        for product in products:
            recs = engine.evaluate_rules(EntityType.PRODUCT, product)
            overstock_recs = [r for r in recs if r["rule_name"] == "product_overstock"]
            if overstock_recs:
                overstock_products.append(product)
        
        assert len(overstock_products) == 2
        assert all(p["couverture_stock"] > 6 for p in overstock_products)
    
    def test_scenario_product_rupture_risk(self):
        """
        Use Case: "Quels produits risquent la rupture le mois prochain ?"
        Expected: Filter by Couverture_Stock < 1
        """
        engine = get_business_rules_engine()
        
        products = [
            {"ref_article": "P001", "couverture_stock": 0.5, "stock_terme": 50},
            {"ref_article": "P002", "couverture_stock": 3.0, "stock_terme": 300},
            {"ref_article": "P003", "couverture_stock": 0.8, "stock_terme": 80},
        ]
        
        rupture_products = []
        for product in products:
            recs = engine.evaluate_rules(EntityType.PRODUCT, product)
            rupture_recs = [r for r in recs if r["rule_name"] == "product_stock_critical"]
            if rupture_recs:
                rupture_products.append(product)
        
        assert len(rupture_products) == 2
        assert all(p["couverture_stock"] < 1 for p in rupture_products)
    
    def test_scenario_product_sales_decline_15pct(self):
        """
        Use Case: "Quels produits ont perdu plus de 15% de ventes ?"
        Expected: Filter by Variation_Qte_% < -15
        """
        engine = get_business_rules_engine()
        
        products = [
            {"ref_article": "P001", "variation_qte_pct": -20.0},
            {"ref_article": "P002", "variation_qte_pct": -5.0},
            {"ref_article": "P003", "variation_qte_pct": -18.0},
        ]
        
        declining_products = []
        for product in products:
            recs = engine.evaluate_rules(EntityType.PRODUCT, product)
            decline_recs = [r for r in recs if r["rule_name"] == "product_sales_decline"]
            if decline_recs:
                declining_products.append(product)
        
        assert len(declining_products) == 2
        assert all(p["variation_qte_pct"] < -15 for p in declining_products)


class TestIntegration:
    """Integration tests with InsightAgent"""
    
    def test_insight_agent_with_business_rules(self):
        """Test InsightAgent integration with Business Rules Engine"""
        from agents.insight_agent import summarize_results
        
        rows = [
            {
                "code_client": "CL001",
                "ca_ht_net": 5000,
                "mois_depuis_derniere_vente": 4
            },
            {
                "code_client": "CL002",
                "ca_ht_net": 6000,
                "mois_depuis_derniere_vente": 1
            }
        ]
        
        result = summarize_results(
            rows=rows,
            columns=["code_client", "ca_ht_net", "mois_depuis_derniere_vente"],
            question="Analyser les clients inactifs",
            sql="SELECT * FROM t_clients WHERE mois_depuis_derniere_vente >= 3"
        )
        
        assert "summary" in result
        assert "recommendations" in result
        assert "kpis" in result
        assert result["row_count"] == 2
    
    def test_no_hardcoded_values_in_recommendations(self):
        """Ensure recommendations are data-driven, not hardcoded"""
        engine = get_business_rules_engine()
        
        # Test with different threshold values
        test_cases = [
            {"mois_depuis_derniere_vente": 3, "should_trigger": True},
            {"mois_depuis_derniere_vente": 2, "should_trigger": False},
            {"mois_depuis_derniere_vente": 6, "should_trigger": True},
        ]
        
        for case in test_cases:
            recs = engine.evaluate_rules(EntityType.CLIENT, case)
            inactive_recs = [r for r in recs if r["rule_name"] == "client_inactive"]
            
            if case["should_trigger"]:
                assert len(inactive_recs) > 0, f"Failed to trigger for {case}"
            else:
                assert len(inactive_recs) == 0, f"False positive for {case}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
