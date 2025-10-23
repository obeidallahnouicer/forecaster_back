"""
Intent Classifier - Routes questions to appropriate data sources.

Classifies user questions into business categories:
- CLIENT: Questions about customers
- PRODUIT: Questions about products
- STOCK: Questions about inventory
- VENTES_MENSUELLES: Questions about sales trends

Sub-types:
- identification: "Quel est le client CL025?"
- activité/inactivité: "Quels clients inactifs depuis 3 mois?"
- performance: "Quels produits ont augmenté de 20%?"
- risque/rupture: "Quels produits risquent une rupture?"
- recommandation: "Quelle action pour la famille Shampooing?"
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("agents.intent_classifier")


class QuestionType(Enum):
    """Main question types."""
    CLIENT = "CLIENT"
    PRODUIT = "PRODUIT"
    STOCK = "STOCK"
    VENTES_MENSUELLES = "VENTES_MENSUELLES"
    GENERAL = "GENERAL"


class SubType(Enum):
    """Question sub-types."""
    IDENTIFICATION = "identification"
    ACTIVITE = "activité"
    INACTIVITE = "inactivité"
    PERFORMANCE = "performance"
    RISQUE = "risque"
    RUPTURE = "rupture"
    SURSTOCK = "surstock"
    RECOMMANDATION = "recommandation"
    TENDANCE = "tendance"
    COMPARAISON = "comparaison"


@dataclass
class Intent:
    """Classified intent with confidence."""
    question_type: QuestionType
    sub_type: Optional[SubType]
    entities: Dict[str, Any]
    confidence: float
    requires_aggregation: bool
    requires_time_comparison: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question_type': self.question_type.value,
            'sub_type': self.sub_type.value if self.sub_type else None,
            'entities': self.entities,
            'confidence': self.confidence,
            'requires_aggregation': self.requires_aggregation,
            'requires_time_comparison': self.requires_time_comparison
        }


class IntentClassifier:
    """
    Deterministic intent classifier using pattern matching and rules.
    
    Classifies questions into business categories and extracts entities
    like client codes, product references, time periods, thresholds, etc.
    """
    
    def __init__(self):
        """Initialize intent classifier."""
        # Pattern definitions for entity extraction
        self.patterns = {
            'client_code': r'\b(CL\d{4,})\b',
            'product_ref': r'\b([A-Z0-9]{6,})\b',
            'percentage': r'(\d+)\s*%',
            'months': r'(\d+)\s*mois',
            'famille': r'(?:famille|gamme)\s+(\w+)',
            'marque': r'(?:marque)\s+(\w+)',
        }
        
        # Keyword patterns for classification
        self.client_keywords = [
            'client', 'clients', 'acheteur', 'acheteurs', 'distributeur',
            'code client', 'clientèle'
        ]
        
        self.product_keywords = [
            'produit', 'produits', 'article', 'articles', 'référence',
            'ref', 'gamme', 'famille', 'sous-famille', 'marque'
        ]
        
        self.stock_keywords = [
            'stock', 'stocks', 'inventaire', 'rupture', 'surstock',
            'disponibilité', 'couverture', 'réappro', 'réapprovisionnement'
        ]
        
        self.ventes_keywords = [
            'vente', 'ventes', 'chiffre d\'affaire', 'ca', 'ca ht',
            'quantité vendu', 'volume', 'tendance', 'évolution', 'croissance'
        ]
        
        # Sub-type keywords
        self.identification_keywords = ['quel est', 'qui est', 'c\'est quoi', 'identifie']
        self.activite_keywords = ['actif', 'actifs', 'activité', 'achète', 'achètent']
        self.inactivite_keywords = ['inactif', 'inactifs', 'inactivité', 'n\'achète pas', 'ne vend pas']
        self.performance_keywords = ['augmenté', 'augmentation', 'croissance', 'croissant', 'meilleur', 'top']
        self.risque_keywords = ['risque', 'danger', 'alerte', 'critique', 'problème']
        self.rupture_keywords = ['rupture', 'manque', 'insuffisant', 'bas stock']
        self.surstock_keywords = ['surstock', 'excès', 'trop', 'surstockage']
        self.recommandation_keywords = ['recommande', 'conseil', 'action', 'que faire', 'stratégie']
        self.tendance_keywords = ['tendance', 'évolution', 'progression', 'baisse', 'hausse']
        self.comparaison_keywords = ['compare', 'comparaison', 'vs', 'versus', 'différence']
        
        logger.info("IntentClassifier initialized")
    
    def classify(self, query: str) -> Intent:
        """
        Classify user query into intent.
        
        Args:
            query: User question
            
        Returns:
            Intent with classification and entities
        """
        query_lower = query.lower()
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine main question type
        question_type, type_confidence = self._classify_question_type(query_lower, entities)
        
        # Determine sub-type
        sub_type, sub_confidence = self._classify_sub_type(query_lower, question_type)
        
        # Determine if aggregation needed
        requires_aggregation = self._requires_aggregation(query_lower)
        
        # Determine if time comparison needed
        requires_time_comparison = self._requires_time_comparison(query_lower)
        
        # Calculate overall confidence
        confidence = (type_confidence + sub_confidence) / 2.0
        
        intent = Intent(
            question_type=question_type,
            sub_type=sub_type,
            entities=entities,
            confidence=confidence,
            requires_aggregation=requires_aggregation,
            requires_time_comparison=requires_time_comparison
        )
        
        logger.info(f"Classified query: type={question_type.value}, sub_type={sub_type.value if sub_type else None}, confidence={confidence:.2f}")
        
        return intent
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query."""
        entities = {}
        
        # Extract client codes
        client_matches = re.findall(self.patterns['client_code'], query)
        if client_matches:
            entities['client_codes'] = client_matches
        
        # Extract product references
        product_matches = re.findall(self.patterns['product_ref'], query)
        if product_matches:
            entities['product_refs'] = product_matches
        
        # Extract percentages
        percentage_matches = re.findall(self.patterns['percentage'], query)
        if percentage_matches:
            entities['percentages'] = [int(p) for p in percentage_matches]
        
        # Extract months
        month_matches = re.findall(self.patterns['months'], query)
        if month_matches:
            entities['months'] = [int(m) for m in month_matches]
        
        # Extract famille
        famille_matches = re.findall(self.patterns['famille'], query, re.IGNORECASE)
        if famille_matches:
            entities['famille'] = famille_matches[0]
        
        # Extract marque
        marque_matches = re.findall(self.patterns['marque'], query, re.IGNORECASE)
        if marque_matches:
            entities['marque'] = marque_matches[0]
        
        return entities
    
    def _classify_question_type(self, query: str, entities: Dict[str, Any]) -> Tuple[QuestionType, float]:
        """Classify main question type."""
        scores = {
            QuestionType.CLIENT: 0.0,
            QuestionType.PRODUIT: 0.0,
            QuestionType.STOCK: 0.0,
            QuestionType.VENTES_MENSUELLES: 0.0,
            QuestionType.GENERAL: 0.0
        }
        
        # Score based on keywords
        for keyword in self.client_keywords:
            if keyword in query:
                scores[QuestionType.CLIENT] += 1.0
        
        for keyword in self.product_keywords:
            if keyword in query:
                scores[QuestionType.PRODUIT] += 1.0
        
        for keyword in self.stock_keywords:
            if keyword in query:
                scores[QuestionType.STOCK] += 1.5  # Stock keywords are more specific
        
        for keyword in self.ventes_keywords:
            if keyword in query:
                scores[QuestionType.VENTES_MENSUELLES] += 1.0
        
        # Boost based on entities
        if 'client_codes' in entities:
            scores[QuestionType.CLIENT] += 2.0
        
        if 'product_refs' in entities:
            scores[QuestionType.PRODUIT] += 2.0
        
        # Stock-specific patterns
        if any(w in query for w in ['rupture', 'surstock', 'couverture']):
            scores[QuestionType.STOCK] += 2.0
        
        # Get top score
        max_score = max(scores.values())
        
        if max_score == 0:
            return QuestionType.GENERAL, 0.3
        
        question_type = max(scores, key=scores.get)
        confidence = min(1.0, max_score / 3.0)  # Normalize
        
        return question_type, confidence
    
    def _classify_sub_type(self, query: str, question_type: QuestionType) -> Tuple[Optional[SubType], float]:
        """Classify question sub-type."""
        scores = {}
        
        # Check for identification
        if any(kw in query for kw in self.identification_keywords):
            scores[SubType.IDENTIFICATION] = 1.0
        
        # Check for activity/inactivity
        if any(kw in query for kw in self.activite_keywords):
            scores[SubType.ACTIVITE] = 1.0
        
        if any(kw in query for kw in self.inactivite_keywords):
            scores[SubType.INACTIVITE] = 1.5  # More specific
        
        # Check for performance
        if any(kw in query for kw in self.performance_keywords):
            scores[SubType.PERFORMANCE] = 1.0
        
        # Check for risk/rupture/surstock
        if any(kw in query for kw in self.risque_keywords):
            scores[SubType.RISQUE] = 1.0
        
        if any(kw in query for kw in self.rupture_keywords):
            scores[SubType.RUPTURE] = 1.5
        
        if any(kw in query for kw in self.surstock_keywords):
            scores[SubType.SURSTOCK] = 1.5
        
        # Check for recommendation
        if any(kw in query for kw in self.recommandation_keywords):
            scores[SubType.RECOMMANDATION] = 1.0
        
        # Check for tendance
        if any(kw in query for kw in self.tendance_keywords):
            scores[SubType.TENDANCE] = 1.0
        
        # Check for comparison
        if any(kw in query for kw in self.comparaison_keywords):
            scores[SubType.COMPARAISON] = 1.0
        
        if not scores:
            return None, 0.5
        
        max_score = max(scores.values())
        sub_type = max(scores, key=scores.get)
        confidence = min(1.0, max_score / 2.0)
        
        return sub_type, confidence
    
    def _requires_aggregation(self, query: str) -> bool:
        """Check if query requires aggregation."""
        aggregation_keywords = [
            'quels', 'combien', 'total', 'somme', 'moyenne', 'liste',
            'tous', 'toutes', 'nombre'
        ]
        return any(kw in query for kw in aggregation_keywords)
    
    def _requires_time_comparison(self, query: str) -> bool:
        """Check if query requires time-based comparison."""
        time_keywords = [
            'depuis', 'pendant', 'dernier', 'mois', 'année', 'trimestre',
            'évolution', 'tendance', 'croissance', 'augmenté', 'diminué'
        ]
        return any(kw in query for kw in time_keywords)
    
    def get_suggested_columns(self, intent: Intent) -> List[str]:
        """
        Get suggested columns for query based on intent.
        
        Args:
            intent: Classified intent
            
        Returns:
            List of column names to include in query
        """
        columns = []
        
        if intent.question_type == QuestionType.CLIENT:
            columns = [
                'Code_Client', 'Intitule_Client', 'Categorie_Client',
                'Zone', 'Representant'
            ]
            if intent.requires_aggregation:
                columns.extend(['CA_HT_NET', 'Qte_Vendu', 'Date'])
        
        elif intent.question_type == QuestionType.PRODUIT:
            columns = [
                'Ref_Article', 'Designation', 'Marque', 'Famille', 'Sous_Famille'
            ]
            if intent.requires_aggregation:
                columns.extend(['CA_HT_NET', 'Qte_Vendu', 'Date'])
        
        elif intent.question_type == QuestionType.STOCK:
            columns = [
                'Référence_Article', 'Désignation', 'Stock_à_terme'
            ]
        
        elif intent.question_type == QuestionType.VENTES_MENSUELLES:
            columns = [
                'Ref_Article', 'Designation', 'Date', 'Mois', 'Annee',
                'CA_HT_NET', 'Qte_Vendu'
            ]
        
        return columns


# Global singleton
_intent_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create global IntentClassifier instance."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
