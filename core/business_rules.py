"""
Business Rules Engine

Implements all business logic, KPI calculations, and recommendations
without hardcoding values. Rules are defined declaratively and can be
configured or extended without modifying agent code.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from datetime import datetime, timedelta


class AlertLevel(Enum):
    """Severity levels for alerts and recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EntityType(Enum):
    """Types of business entities"""
    CLIENT = "client"
    PRODUCT = "product"
    FAMILY = "family"
    BRAND = "brand"
    SALES = "sales"
    STOCK = "stock"
    ZONE = "zone"


@dataclass
class BusinessRule:
    """A single business rule with condition and action"""
    name: str
    entity_type: EntityType
    condition: Callable[[Dict[str, Any]], bool]
    recommendation: str
    alert_level: AlertLevel
    kpi_fields: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class KPIDefinition:
    """Definition of a calculated KPI"""
    name: str
    calculation: Callable[[List[Dict]], Any]
    description: str
    unit: Optional[str] = None
    format_string: str = "{:.2f}"


class BusinessRulesEngine:
    """
    Central engine for business rules, KPI calculations, and recommendations.
    All rules are data-driven and configurable.
    """
    
    def __init__(self):
        self.rules: List[BusinessRule] = []
        self.kpis: Dict[str, KPIDefinition] = {}
        self._register_default_rules()
        self._register_default_kpis()
    
    def _register_default_rules(self):
        """Register all business rules from the requirements document"""
        
        # CLIENT RULES
        self.add_rule(BusinessRule(
            name="client_inactive",
            entity_type=EntityType.CLIENT,
            condition=lambda data: data.get("mois_depuis_derniere_vente", 0) >= 3,
            recommendation="Relance commerciale ciblée pour réactiver le client",
            alert_level=AlertLevel.HIGH,
            kpi_fields=["mois_depuis_derniere_vente"],
            description="Client inactif depuis 3 mois ou plus"
        ))
        
        self.add_rule(BusinessRule(
            name="client_ca_decline",
            entity_type=EntityType.CLIENT,
            condition=lambda data: data.get("delta_ca_pct", 0) < -10,
            recommendation="Analyser les causes de la baisse et proposer actions correctives",
            alert_level=AlertLevel.HIGH,
            kpi_fields=["delta_ca_pct", "ca_total"],
            description="Baisse significative du CA client (> 10%)"
        ))
        
        self.add_rule(BusinessRule(
            name="client_high_growth",
            entity_type=EntityType.CLIENT,
            condition=lambda data: data.get("delta_ca_pct", 0) > 20,
            recommendation="Consolider la relation et identifier opportunités additionnelles",
            alert_level=AlertLevel.MEDIUM,
            kpi_fields=["delta_ca_pct", "ca_total"],
            description="Forte croissance client (> 20%)"
        ))
        
        self.add_rule(BusinessRule(
            name="client_at_risk",
            entity_type=EntityType.CLIENT,
            condition=lambda data: (
                data.get("delta_ca_pct", 0) < -15 and 
                data.get("frequence_achat_moyenne", 0) < 2
            ),
            recommendation="Client à risque de perte - action commerciale urgente",
            alert_level=AlertLevel.CRITICAL,
            kpi_fields=["delta_ca_pct", "frequence_achat_moyenne"],
            description="Client en forte baisse avec faible fréquence d'achat"
        ))
        
        # PRODUCT RULES
        self.add_rule(BusinessRule(
            name="product_stock_critical",
            entity_type=EntityType.PRODUCT,
            condition=lambda data: data.get("couverture_stock", 99) < 1,
            recommendation="Réassortir immédiatement - risque de rupture",
            alert_level=AlertLevel.CRITICAL,
            kpi_fields=["couverture_stock", "stock_terme"],
            description="Couverture de stock inférieure à 1 mois"
        ))
        
        self.add_rule(BusinessRule(
            name="product_overstock",
            entity_type=EntityType.PRODUCT,
            condition=lambda data: data.get("couverture_stock", 0) > 6,
            recommendation="Lancer promotion ou déstockage",
            alert_level=AlertLevel.MEDIUM,
            kpi_fields=["couverture_stock", "stock_terme"],
            description="Surstock - couverture supérieure à 6 mois"
        ))
        
        self.add_rule(BusinessRule(
            name="product_sales_decline",
            entity_type=EntityType.PRODUCT,
            condition=lambda data: data.get("variation_qte_pct", 0) < -15,
            recommendation="Enquête sur la cause (prix, rupture, saison, concurrence)",
            alert_level=AlertLevel.HIGH,
            kpi_fields=["variation_qte_pct", "qte_totale_vendue"],
            description="Baisse significative des ventes produit (> 15%)"
        ))
        
        self.add_rule(BusinessRule(
            name="product_high_demand",
            entity_type=EntityType.PRODUCT,
            condition=lambda data: (
                data.get("variation_qte_pct", 0) > 20 and
                data.get("couverture_stock", 99) < 3
            ),
            recommendation="Réévaluer le stock - forte demande et stock faible",
            alert_level=AlertLevel.HIGH,
            kpi_fields=["variation_qte_pct", "couverture_stock"],
            description="Forte demande avec stock insuffisant"
        ))
        
        self.add_rule(BusinessRule(
            name="product_losing_customers",
            entity_type=EntityType.PRODUCT,
            condition=lambda data: (
                data.get("nb_clients_actifs", 0) > 0 and
                data.get("delta_clients_pct", 0) < -20
            ),
            recommendation="Action marketing ciblée - perte d'intérêt client",
            alert_level=AlertLevel.MEDIUM,
            kpi_fields=["nb_clients_actifs", "delta_clients_pct"],
            description="Perte significative de clients acheteurs"
        ))
        
        # SALES RULES
        self.add_rule(BusinessRule(
            name="sales_monthly_decline",
            entity_type=EntityType.SALES,
            condition=lambda data: data.get("delta_ca_mois_pct", 0) < -10,
            recommendation="Alerte baisse mensuelle - analyser segments touchés",
            alert_level=AlertLevel.HIGH,
            kpi_fields=["delta_ca_mois_pct", "ca_ht_net"],
            description="Baisse mensuelle significative du CA (> 10%)"
        ))
        
        self.add_rule(BusinessRule(
            name="sales_yearly_decline",
            entity_type=EntityType.SALES,
            condition=lambda data: data.get("delta_ca_annee_pct", 0) < -5,
            recommendation="Alerte structurelle - révision stratégie commerciale",
            alert_level=AlertLevel.CRITICAL,
            kpi_fields=["delta_ca_annee_pct"],
            description="Baisse annuelle persistante"
        ))
        
        self.add_rule(BusinessRule(
            name="high_variability",
            entity_type=EntityType.SALES,
            condition=lambda data: data.get("coefficient_variation", 0) > 0.7,
            recommendation="Variabilité élevée - stabiliser la demande ou revoir pricing",
            alert_level=AlertLevel.MEDIUM,
            kpi_fields=["coefficient_variation"],
            description="Forte variabilité des ventes"
        ))
    
    def _register_default_kpis(self):
        """Register all KPI calculations"""
        
        # CLIENT KPIs
        self.add_kpi(KPIDefinition(
            name="ca_moyen_annuel",
            calculation=lambda rows: (
                sum(r.get("ca_ht_net", 0) for r in rows) / 
                len(set(r.get("annee") for r in rows))
                if rows and any(r.get("annee") for r in rows) else 0
            ),
            description="Chiffre d'affaires moyen annuel",
            unit="TND",
            format_string="{:,.2f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="frequence_achat_moyenne",
            calculation=lambda rows: (
                len(set(r.get("ndocument") or r.get("date") for r in rows)) /
                max(1, len(set(r.get("mois") for r in rows if r.get("mois"))))
                if rows else 0
            ),
            description="Fréquence d'achat moyenne (commandes/mois)",
            unit="commandes/mois",
            format_string="{:.1f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="mois_depuis_derniere_vente",
            calculation=lambda rows: self._calculate_months_since_last_sale(rows),
            description="Nombre de mois depuis la dernière vente",
            unit="mois",
            format_string="{:.0f}"
        ))
        
        # PRODUCT KPIs
        self.add_kpi(KPIDefinition(
            name="couverture_stock",
            calculation=lambda rows: self._calculate_stock_coverage(rows),
            description="Couverture de stock en mois",
            unit="mois",
            format_string="{:.1f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="rotation_mensuelle",
            calculation=lambda rows: (
                statistics.mean([r.get("qte_vendu", 0) for r in rows[-3:]])
                if len(rows) >= 3 else
                sum(r.get("qte_vendu", 0) for r in rows) / max(1, len(rows))
            ),
            description="Rotation mensuelle moyenne (3 derniers mois)",
            unit="unités",
            format_string="{:.1f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="nb_clients_actifs",
            calculation=lambda rows: len(set(
                r.get("code_client") for r in rows if r.get("code_client")
            )),
            description="Nombre de clients actifs",
            unit="clients",
            format_string="{:.0f}"
        ))
        
        # VARIATION KPIs
        self.add_kpi(KPIDefinition(
            name="delta_ca_pct",
            calculation=lambda rows: self._calculate_percentage_change(
                rows, "ca_ht_net", period="month"
            ),
            description="Variation du CA en %",
            unit="%",
            format_string="{:+.1f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="delta_qte_pct",
            calculation=lambda rows: self._calculate_percentage_change(
                rows, "qte_vendu", period="month"
            ),
            description="Variation de la quantité en %",
            unit="%",
            format_string="{:+.1f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="coefficient_variation",
            calculation=lambda rows: self._calculate_coefficient_variation(rows, "ca_ht_net"),
            description="Coefficient de variation",
            unit="",
            format_string="{:.2f}"
        ))
        
        self.add_kpi(KPIDefinition(
            name="tendance_3m",
            calculation=lambda rows: self._calculate_trend(rows, "ca_ht_net", periods=3),
            description="Tendance sur 3 mois (moyenne mobile)",
            unit="TND",
            format_string="{:,.2f}"
        ))
    
    def add_rule(self, rule: BusinessRule):
        """Add a business rule to the engine"""
        self.rules.append(rule)
    
    def add_kpi(self, kpi: KPIDefinition):
        """Add a KPI definition to the engine"""
        self.kpis[kpi.name] = kpi
    
    def evaluate_rules(
        self, 
        entity_type: EntityType, 
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all rules for a given entity type and data.
        Returns list of triggered recommendations.
        """
        recommendations = []
        
        for rule in self.rules:
            if rule.entity_type != entity_type:
                continue
            
            try:
                if rule.condition(data):
                    recommendations.append({
                        "rule_name": rule.name,
                        "recommendation": rule.recommendation,
                        "alert_level": rule.alert_level.value,
                        "description": rule.description,
                        "relevant_kpis": {
                            k: data.get(k) for k in rule.kpi_fields if k in data
                        }
                    })
            except Exception as e:
                # Log but don't fail on rule evaluation errors
                import logging
                logging.getLogger(__name__).debug(
                    f"Rule {rule.name} evaluation failed: {e}"
                )
        
        return recommendations
    
    def calculate_kpis(
        self, 
        rows: List[Dict[str, Any]], 
        kpi_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate specified KPIs (or all if None) from data rows.
        Returns dict of {kpi_name: calculated_value}
        """
        if kpi_names is None:
            kpi_names = list(self.kpis.keys())
        
        results = {}
        for name in kpi_names:
            if name not in self.kpis:
                continue
            
            try:
                kpi_def = self.kpis[name]
                value = kpi_def.calculation(rows)
                results[name] = {
                    "value": value,
                    "formatted": kpi_def.format_string.format(value),
                    "unit": kpi_def.unit,
                    "description": kpi_def.description
                }
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(
                    f"KPI {name} calculation failed: {e}"
                )
                results[name] = {
                    "value": None,
                    "formatted": "N/A",
                    "unit": None,
                    "description": f"Calculation failed: {e}"
                }
        
        return results
    
    def generate_insights(
        self,
        entity_type: EntityType,
        rows: List[Dict[str, Any]],
        question: str = ""
    ) -> Dict[str, Any]:
        """
        Generate complete business insights: KPIs + recommendations.
        This is the main entry point for analysis.
        """
        # Calculate KPIs
        kpis = self.calculate_kpis(rows)
        
        # Build data dict with calculated values for rule evaluation
        data = {k: v["value"] for k, v in kpis.items()}
        
        # Add raw aggregates
        if rows:
            data.update({
                "ca_total": sum(r.get("ca_ht_net", 0) for r in rows),
                "qte_totale_vendue": sum(r.get("qte_vendu", 0) for r in rows),
                "nb_lignes": len(rows)
            })
        
        # Evaluate rules
        recommendations = self.evaluate_rules(entity_type, data)
        
        # Generate summary
        summary = self._generate_summary(entity_type, kpis, recommendations, rows)
        
        return {
            "kpis": kpis,
            "recommendations": recommendations,
            "summary": summary,
            "entity_type": entity_type.value,
            "row_count": len(rows)
        }
    
    def _generate_summary(
        self,
        entity_type: EntityType,
        kpis: Dict[str, Any],
        recommendations: List[Dict],
        rows: List[Dict]
    ) -> str:
        """Generate natural language summary"""
        parts = []
        
        if not rows:
            return "Aucune donnée trouvée pour cette analyse."
        
        parts.append(f"Analyse de {len(rows)} enregistrements.")
        
        # Add key metrics based on entity type
        if entity_type == EntityType.CLIENT:
            if "ca_moyen_annuel" in kpis:
                parts.append(
                    f"CA moyen annuel: {kpis['ca_moyen_annuel']['formatted']} "
                    f"{kpis['ca_moyen_annuel']['unit']}."
                )
        elif entity_type == EntityType.PRODUCT:
            if "couverture_stock" in kpis:
                parts.append(
                    f"Couverture de stock: {kpis['couverture_stock']['formatted']} "
                    f"{kpis['couverture_stock']['unit']}."
                )
        
        # Add alert count
        if recommendations:
            critical = sum(1 for r in recommendations if r["alert_level"] == "critical")
            high = sum(1 for r in recommendations if r["alert_level"] == "high")
            if critical:
                parts.append(f"⚠️ {critical} alerte(s) critique(s) détectée(s).")
            elif high:
                parts.append(f"⚠️ {high} alerte(s) haute priorité détectée(s).")
        
        return " ".join(parts)
    
    # Helper calculation methods
    
    def _calculate_months_since_last_sale(self, rows: List[Dict]) -> float:
        """Calculate months since last sale date"""
        if not rows:
            return 999.0
        
        dates = []
        for r in rows:
            date_str = r.get("date")
            if date_str:
                try:
                    # Handle various date formats
                    if isinstance(date_str, str):
                        # Try ISO format first
                        try:
                            dt = datetime.fromisoformat(date_str.split()[0])
                        except:
                            # Try other common formats
                            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
                                try:
                                    dt = datetime.strptime(date_str, fmt)
                                    break
                                except:
                                    continue
                            else:
                                continue
                        dates.append(dt)
                except:
                    continue
        
        if not dates:
            return 999.0
        
        last_date = max(dates)
        today = datetime.now()
        delta = today - last_date
        return delta.days / 30.0
    
    def _calculate_stock_coverage(self, rows: List[Dict]) -> float:
        """Calculate stock coverage in months"""
        # Expects rows with stock_terme and rotation_mensuelle
        if not rows:
            return 0.0
        
        # Use most recent row
        latest = rows[-1] if rows else {}
        stock = latest.get("stock_terme", 0)
        rotation = latest.get("rotation_mensuelle", 0)
        
        if rotation <= 0:
            return 999.0 if stock > 0 else 0.0
        
        return stock / rotation
    
    def _calculate_percentage_change(
        self, 
        rows: List[Dict], 
        field: str, 
        period: str = "month"
    ) -> float:
        """Calculate percentage change between periods"""
        if len(rows) < 2:
            return 0.0
        
        # Sort by date if available
        sorted_rows = sorted(
            rows,
            key=lambda r: r.get("date", "") or r.get("mois", "") or "",
            reverse=True
        )
        
        if len(sorted_rows) < 2:
            return 0.0
        
        current = sorted_rows[0].get(field, 0) or 0
        previous = sorted_rows[1].get(field, 0) or 0
        
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        
        return ((current - previous) / previous) * 100.0
    
    def _calculate_coefficient_variation(
        self, 
        rows: List[Dict], 
        field: str
    ) -> float:
        """Calculate coefficient of variation (std/mean)"""
        if len(rows) < 2:
            return 0.0
        
        values = [r.get(field, 0) or 0 for r in rows]
        values = [v for v in values if v != 0]  # Remove zeros
        
        if not values or len(values) < 2:
            return 0.0
        
        try:
            mean = statistics.mean(values)
            if mean == 0:
                return 0.0
            std = statistics.stdev(values)
            return std / abs(mean)
        except:
            return 0.0
    
    def _calculate_trend(
        self, 
        rows: List[Dict], 
        field: str, 
        periods: int = 3
    ) -> float:
        """Calculate moving average trend"""
        if len(rows) < periods:
            return sum(r.get(field, 0) for r in rows) / max(1, len(rows))
        
        recent = rows[-periods:]
        return sum(r.get(field, 0) for r in recent) / periods


# Global singleton instance
_engine: Optional[BusinessRulesEngine] = None


def get_business_rules_engine() -> BusinessRulesEngine:
    """Get or create global business rules engine instance"""
    global _engine
    if _engine is None:
        _engine = BusinessRulesEngine()
    return _engine
