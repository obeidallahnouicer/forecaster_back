"""
Business Rules Engine - Applies business logic and rules to metrics.

Rules include:
- Rupture detection: Couverture_Stock < 1 month
- Baisse CA: ΔCA_% < -10%
- Surstock detection: Couverture_Stock > 6 months
- Produit en recul: ΔQte_% < -15%
- Client inactif: Pas de ventes depuis N mois
- Recommandations basées sur les seuils métier
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .metrics_calculator import ProductMetrics

logger = logging.getLogger("agents.rules_engine")


class RuleType(Enum):
    """Types of business rules."""
    STOCK_RUPTURE = "stock_rupture"
    STOCK_SURSTOCK = "stock_surstock"
    CA_BAISSE = "ca_baisse"
    CA_HAUSSE = "ca_hausse"
    QTE_BAISSE = "qte_baisse"
    QTE_HAUSSE = "qte_hausse"
    CLIENT_INACTIF = "client_inactif"
    PRODUIT_RISQUE = "produit_risque"
    OPPORTUNITE = "opportunite"


class Severity(Enum):
    """Rule severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RuleResult:
    """Result of applying a business rule."""
    rule_type: RuleType
    triggered: bool
    severity: Severity
    message: str
    recommendation: Optional[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_type': self.rule_type.value,
            'triggered': self.triggered,
            'severity': self.severity.value,
            'message': self.message,
            'recommendation': self.recommendation,
            'metrics': self.metrics
        }


class RulesEngine:
    """
    Applies business rules to metrics and generates recommendations.
    
    Business rules:
    - Couverture_Stock < 1 ⇒ Risque rupture (CRITICAL)
    - Couverture_Stock > 6 ⇒ Surstock (HIGH)
    - ΔCA_% < -10% ⇒ Baisse CA (HIGH)
    - ΔQte_% < -15% ⇒ Produit en recul (HIGH)
    - ΔCA_% > 20% ⇒ Forte croissance (OPPORTUNITE)
    - Client sans achat 3+ mois ⇒ Client inactif (MEDIUM)
    """
    
    def __init__(self):
        """Initialize rules engine."""
        # Thresholds (configurable)
        self.RUPTURE_THRESHOLD = 1.0  # months
        self.SURSTOCK_THRESHOLD = 6.0  # months
        self.CA_BAISSE_THRESHOLD = -10.0  # %
        self.CA_HAUSSE_THRESHOLD = 20.0  # %
        self.QTE_BAISSE_THRESHOLD = -15.0  # %
        self.QTE_HAUSSE_THRESHOLD = 25.0  # %
        self.CLIENT_INACTIF_MONTHS = 3
        
        logger.info("RulesEngine initialized")
    
    def evaluate_product(self, metrics: ProductMetrics) -> List[RuleResult]:
        """
        Evaluate all rules for a product.
        
        Args:
            metrics: Product metrics
            
        Returns:
            List of triggered rule results
        """
        results = []
        
        # Check stock rupture
        if metrics.couverture_stock is not None:
            if metrics.couverture_stock < self.RUPTURE_THRESHOLD:
                results.append(RuleResult(
                    rule_type=RuleType.STOCK_RUPTURE,
                    triggered=True,
                    severity=Severity.CRITICAL,
                    message=f"Risque de rupture de stock: couverture de {metrics.couverture_stock:.1f} mois",
                    recommendation="Réapprovisionner d'urgence. Contacter fournisseur et accélérer livraison.",
                    metrics={
                        'couverture_stock': metrics.couverture_stock,
                        'stock_actuel': metrics.stock_actuel,
                        'rotation_mensuelle': metrics.rotation_mensuelle
                    }
                ))
        
        # Check surstock
        if metrics.couverture_stock is not None:
            if metrics.couverture_stock > self.SURSTOCK_THRESHOLD:
                results.append(RuleResult(
                    rule_type=RuleType.STOCK_SURSTOCK,
                    triggered=True,
                    severity=Severity.HIGH,
                    message=f"Surstock détecté: couverture de {metrics.couverture_stock:.1f} mois",
                    recommendation="Réduire commandes. Envisager promotions ou déstockage.",
                    metrics={
                        'couverture_stock': metrics.couverture_stock,
                        'stock_actuel': metrics.stock_actuel
                    }
                ))
        
        # Check CA baisse
        if metrics.delta_ca_pct < self.CA_BAISSE_THRESHOLD:
            results.append(RuleResult(
                rule_type=RuleType.CA_BAISSE,
                triggered=True,
                severity=Severity.HIGH,
                message=f"Baisse significative du CA: {metrics.delta_ca_pct:.1f}%",
                recommendation="Analyser causes (prix, concurrence, saisonnalité). Envisager actions marketing.",
                metrics={
                    'delta_ca_pct': metrics.delta_ca_pct,
                    'ca_actuel': metrics.ca_actuel,
                    'ca_precedent': metrics.ca_precedent
                }
            ))
        
        # Check CA hausse (opportunité)
        if metrics.delta_ca_pct > self.CA_HAUSSE_THRESHOLD:
            results.append(RuleResult(
                rule_type=RuleType.CA_HAUSSE,
                triggered=True,
                severity=Severity.INFO,
                message=f"Forte croissance du CA: +{metrics.delta_ca_pct:.1f}%",
                recommendation="Opportunité de croissance. Assurer stock suffisant et renforcer positionnement.",
                metrics={
                    'delta_ca_pct': metrics.delta_ca_pct,
                    'ca_actuel': metrics.ca_actuel
                }
            ))
        
        # Check quantité baisse
        if metrics.delta_qte_pct < self.QTE_BAISSE_THRESHOLD:
            results.append(RuleResult(
                rule_type=RuleType.QTE_BAISSE,
                triggered=True,
                severity=Severity.HIGH,
                message=f"Produit en recul: {metrics.delta_qte_pct:.1f}% de baisse des quantités",
                recommendation="Évaluer pertinence du produit. Considérer phase-out si tendance continue.",
                metrics={
                    'delta_qte_pct': metrics.delta_qte_pct,
                    'qte_actuelle': metrics.qte_actuelle,
                    'qte_precedente': metrics.qte_precedente
                }
            ))
        
        # Check produit à risque (combinaison de facteurs)
        risk_score = 0
        if metrics.couverture_stock and metrics.couverture_stock < 1.5:
            risk_score += 2
        if metrics.delta_ca_pct < -5:
            risk_score += 1
        if metrics.statut_ventes == 'declin':
            risk_score += 1
        
        if risk_score >= 3:
            results.append(RuleResult(
                rule_type=RuleType.PRODUIT_RISQUE,
                triggered=True,
                severity=Severity.HIGH,
                message="Produit à risque élevé: combinaison de facteurs négatifs",
                recommendation="Attention prioritaire requise. Audit complet du produit recommandé.",
                metrics={
                    'risk_score': risk_score,
                    'facteurs': {
                        'stock_faible': metrics.couverture_stock < 1.5 if metrics.couverture_stock else False,
                        'ca_baisse': metrics.delta_ca_pct < -5,
                        'ventes_declin': metrics.statut_ventes == 'declin'
                    }
                }
            ))
        
        return results
    
    def evaluate_client(self, client_metrics: Dict[str, Any]) -> List[RuleResult]:
        """
        Evaluate rules for a client.
        
        Args:
            client_metrics: Client metrics dictionary
            
        Returns:
            List of triggered rule results
        """
        results = []
        
        # Check client inactif
        if not client_metrics.get('actif'):
            mois_inactif = client_metrics.get('mois_depuis_derniere_vente', 0)
            
            if mois_inactif >= self.CLIENT_INACTIF_MONTHS:
                severity = Severity.HIGH if mois_inactif >= 6 else Severity.MEDIUM
                
                results.append(RuleResult(
                    rule_type=RuleType.CLIENT_INACTIF,
                    triggered=True,
                    severity=severity,
                    message=f"Client inactif depuis {mois_inactif} mois",
                    recommendation=f"Action de réactivation requise. Contacter commercial ou proposer offre spéciale.",
                    metrics={
                        'mois_inactif': mois_inactif,
                        'derniere_transaction': client_metrics.get('derniere_transaction'),
                        'ca_historique': client_metrics.get('ca_total')
                    }
                ))
        
        return results
    
    def get_global_recommendations(
        self,
        product_results: List[Dict[str, Any]],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate global recommendations from product analysis.
        
        Args:
            product_results: List of product analysis results with rules
            max_recommendations: Maximum number of recommendations
            
        Returns:
            Prioritized recommendations
        """
        recommendations = []
        
        # Count critical issues
        critical_ruptures = sum(
            1 for p in product_results
            for r in p.get('rules', [])
            if r.get('rule_type') == RuleType.STOCK_RUPTURE.value and r.get('triggered')
        )
        
        if critical_ruptures > 0:
            recommendations.append({
                'priority': 1,
                'category': 'STOCK',
                'action': f"URGENT: {critical_ruptures} produit(s) en risque de rupture",
                'details': "Réapprovisionner immédiatement les produits identifiés.",
                'severity': 'critical'
            })
        
        # Count surstocks
        surstocks = sum(
            1 for p in product_results
            for r in p.get('rules', [])
            if r.get('rule_type') == RuleType.STOCK_SURSTOCK.value and r.get('triggered')
        )
        
        if surstocks > 0:
            recommendations.append({
                'priority': 2,
                'category': 'STOCK',
                'action': f"{surstocks} produit(s) en surstock",
                'details': "Planifier actions de déstockage ou réduire commandes.",
                'severity': 'high'
            })
        
        # Count declining products
        declining = sum(
            1 for p in product_results
            for r in p.get('rules', [])
            if r.get('rule_type') in [RuleType.CA_BAISSE.value, RuleType.QTE_BAISSE.value] and r.get('triggered')
        )
        
        if declining > 0:
            recommendations.append({
                'priority': 3,
                'category': 'VENTES',
                'action': f"{declining} produit(s) en baisse de performance",
                'details': "Analyser causes et envisager actions correctives ou phase-out.",
                'severity': 'high'
            })
        
        # Count growth opportunities
        opportunities = sum(
            1 for p in product_results
            for r in p.get('rules', [])
            if r.get('rule_type') == RuleType.CA_HAUSSE.value and r.get('triggered')
        )
        
        if opportunities > 0:
            recommendations.append({
                'priority': 4,
                'category': 'OPPORTUNITE',
                'action': f"{opportunities} produit(s) en forte croissance",
                'details': "Capitaliser sur la croissance: assurer stock et marketing.",
                'severity': 'info'
            })
        
        # Sort by priority and limit
        recommendations.sort(key=lambda x: x['priority'])
        return recommendations[:max_recommendations]
    
    def format_answer(
        self,
        query: str,
        data_source: str,
        columns_used: List[str],
        metrics: Dict[str, Any],
        rules_triggered: List[RuleResult],
        answer_text: str
    ) -> Dict[str, Any]:
        """
        Format final answer according to business chatbot spec.
        
        Args:
            query: Original user question
            data_source: Table/source used (VENTES, STOCK, etc.)
            columns_used: Columns queried
            metrics: Computed metrics
            rules_triggered: Business rules that were triggered
            answer_text: Natural language answer
            
        Returns:
            Formatted response dictionary
        """
        return {
            'query': query,
            'table_used': data_source,
            'columns_used': columns_used,
            'metrics': metrics,
            'business_rules_triggered': [r.to_dict() for r in rules_triggered if r.triggered],
            'answer': answer_text,
            'source': 'multi_agent_rag',
            'validation': 'passed',
            'timestamp': pd.Timestamp.now().isoformat()
        }


# Global singleton
_rules_engine: Optional[RulesEngine] = None


def get_rules_engine() -> RulesEngine:
    """Get or create global RulesEngine instance."""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine()
    return _rules_engine
