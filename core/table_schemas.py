"""
Table Schemas and Business Metadata

Defines all table structures, column mappings, and business semantics
for the chatbot. This ensures the SQL generation and validation use
accurate schema information aligned with business requirements.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ColumnDefinition:
    """Definition of a database column with business metadata"""
    name: str
    type: str  # SQL type
    description: str
    business_meaning: str = ""
    examples: List[str] = field(default_factory=list)
    is_numeric: bool = False
    is_date: bool = False
    is_id: bool = False


@dataclass
class TableSchema:
    """Complete schema for a business table"""
    name: str
    logical_name: str  # Name used in prompts
    description: str
    columns: List[ColumnDefinition]
    primary_key: str = ""
    sample_questions: List[str] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)


# Define all table schemas based on requirements document

CLIENT_SCHEMA = TableSchema(
    name="t_clients",
    logical_name="clients",
    description="Table contenant les informations clients et leur historique d'activité",
    primary_key="Code_Client",
    columns=[
        ColumnDefinition(
            name="Code_Client",
            type="TEXT",
            description="Identifiant unique du client",
            is_id=True,
            examples=["CL001", "CL025"]
        ),
        ColumnDefinition(
            name="Intitulé_Client",
            type="TEXT",
            description="Nom commercial du client",
            examples=["AYADI HOUWAIDA", "BEN ZIDI"]
        ),
        ColumnDefinition(
            name="Categorie_Client",
            type="TEXT",
            description="Type de client (GMS, Superette, Détaillant, etc.)",
            examples=["GMS", "Superette", "Détaillant"]
        ),
        ColumnDefinition(
            name="Zone",
            type="TEXT",
            description="Zone géographique du client",
            examples=["Nord", "Centre", "Sud"]
        ),
        ColumnDefinition(
            name="Gouvernorat",
            type="TEXT",
            description="Gouvernorat du client",
            examples=["Tunis", "Sfax", "Sousse"]
        ),
        ColumnDefinition(
            name="Représentant",
            type="TEXT",
            description="Commercial en charge du client",
            examples=["RABIAA BEN SEDRINE", "Mohamed Ali"]
        ),
        ColumnDefinition(
            name="Date_Premiere_Vente",
            type="DATE",
            description="Date de la première vente au client (ancienneté)",
            is_date=True,
            examples=["2020-01-15", "2019-06-20"]
        ),
        ColumnDefinition(
            name="Date_Derniere_Vente",
            type="DATE",
            description="Date de la dernière commande",
            is_date=True,
            examples=["2025-10-01", "2025-07-15"]
        ),
        ColumnDefinition(
            name="Mois_Depuis_Derniere_Vente",
            type="INTEGER",
            description="Nombre de mois depuis la dernière vente",
            is_numeric=True,
            business_meaning="Indicateur d'inactivité client",
            examples=["0", "3", "6"]
        ),
        ColumnDefinition(
            name="CA_Total",
            type="REAL",
            description="Chiffre d'affaires total du client",
            is_numeric=True,
            business_meaning="Valeur totale du client",
            examples=["50000.00", "125000.00"]
        ),
        ColumnDefinition(
            name="CA_Moyen_Annuel",
            type="REAL",
            description="Chiffre d'affaires moyen par an",
            is_numeric=True,
            business_meaning="Performance moyenne annuelle",
            examples=["25000.00", "60000.00"]
        ),
        ColumnDefinition(
            name="Frequence_Achat_Moyenne",
            type="REAL",
            description="Fréquence d'achat en commandes par mois",
            is_numeric=True,
            business_meaning="Régularité du client",
            examples=["2.5", "4.0", "1.2"]
        ),
        ColumnDefinition(
            name="Statut_Client",
            type="TEXT",
            description="Statut actuel: Actif, Inactif, En baisse",
            examples=["Actif", "Inactif", "En baisse"]
        ),
        ColumnDefinition(
            name="Annees_Actives",
            type="INTEGER",
            description="Nombre d'années d'activité avec l'entreprise",
            is_numeric=True,
            examples=["1", "3", "5"]
        ),
    ],
    sample_questions=[
        "Quels sont les clients inactifs depuis plus de 3 mois ?",
        "Quel est le chiffre d'affaires moyen du client AYADI HOUWAIDA ?",
        "Quels clients ont augmenté leurs achats de plus de 20% cette année ?",
        "Quels clients gère le commercial RABIAA BEN SEDRINE ?",
        "Quels sont les clients à risque de perte ?",
        "Top 10 des clients les plus fidèles ?",
    ],
    business_rules=[
        "Inactivité >= 3 mois → Client inactif → Action: Relance commerciale",
        "ΔCA% < -10% → Baisse significative → Action: Analyse causes",
        "ΔCA% > +20% → Forte croissance → Action: Consolider relation",
        "Fréquence < 1 commande/mois → Faible engagement",
    ]
)

PRODUCT_SCHEMA = TableSchema(
    name="t_produits",
    logical_name="produits",
    description="Table des produits avec performance et état des stocks",
    primary_key="Référence_Article",
    columns=[
        ColumnDefinition(
            name="Référence_Article",
            type="TEXT",
            description="Code produit unique",
            is_id=True,
            examples=["32014", "ART001"]
        ),
        ColumnDefinition(
            name="Désignation",
            type="TEXT",
            description="Nom commercial du produit",
            examples=["ARGAN 250ML", "Shampooing Volume"]
        ),
        ColumnDefinition(
            name="Famille",
            type="TEXT",
            description="Catégorie principale du produit",
            examples=["Cheveux", "Maquillage", "Coiffage"]
        ),
        ColumnDefinition(
            name="Sous_Famille",
            type="TEXT",
            description="Sous-catégorie du produit",
            examples=["Shampooing", "Rouge à lèvres", "Gel"]
        ),
        ColumnDefinition(
            name="Marque",
            type="TEXT",
            description="Marque du produit",
            examples=["H.ZONE", "RENEE BLANCHE"]
        ),
        ColumnDefinition(
            name="Qte_Totale_Vendue",
            type="REAL",
            description="Quantité totale vendue sur la période",
            is_numeric=True,
            examples=["1000", "500"]
        ),
        ColumnDefinition(
            name="CA_Total",
            type="REAL",
            description="Chiffre d'affaires total généré",
            is_numeric=True,
            examples=["50000.00", "25000.00"]
        ),
        ColumnDefinition(
            name="CA_Mensuel_Moyen",
            type="REAL",
            description="Moyenne mensuelle des ventes",
            is_numeric=True,
            examples=["4166.67", "2083.33"]
        ),
        ColumnDefinition(
            name="Rotation_Mensuelle",
            type="REAL",
            description="Moyenne des ventes sur 3 derniers mois",
            is_numeric=True,
            business_meaning="Vitesse d'écoulement du produit",
            examples=["100", "50"]
        ),
        ColumnDefinition(
            name="Stock_Terme",
            type="REAL",
            description="Stock actuel disponible",
            is_numeric=True,
            examples=["500", "1000"]
        ),
        ColumnDefinition(
            name="Couverture_Stock",
            type="REAL",
            description="Stock_Terme / Rotation_Mensuelle (durée en mois)",
            is_numeric=True,
            business_meaning="Nombre de mois de stock disponible",
            examples=["5.0", "20.0", "0.5"]
        ),
        ColumnDefinition(
            name="Nb_Clients_Actifs",
            type="INTEGER",
            description="Nombre de clients ayant acheté sur 3 mois",
            is_numeric=True,
            examples=["15", "30"]
        ),
        ColumnDefinition(
            name="Variation_CA_%",
            type="REAL",
            description="Évolution du CA vs. mois précédent (%)",
            is_numeric=True,
            examples=["-15.5", "+20.0"]
        ),
        ColumnDefinition(
            name="Variation_Qte_%",
            type="REAL",
            description="Évolution quantité vs. mois précédent (%)",
            is_numeric=True,
            examples=["-10.0", "+15.0"]
        ),
        ColumnDefinition(
            name="Tendance_3m",
            type="REAL",
            description="Moyenne mobile des ventes sur 3 mois",
            is_numeric=True,
            examples=["4500.00", "3000.00"]
        ),
        ColumnDefinition(
            name="Recommandation",
            type="TEXT",
            description="Action suggérée: Promo, Réassort, Normal",
            examples=["Promo", "Réassort", "Normal"]
        ),
    ],
    sample_questions=[
        "À quelle famille appartient l'article 32014 ?",
        "Quel est le produit le plus vendu de la marque H.ZONE ?",
        "Est-ce que la famille Shampooing est en croissance ?",
        "Quels produits ont un stock supérieur à 6 mois ?",
        "Quels produits risquent la rupture le mois prochain ?",
        "Combien de clients ont acheté ce produit ce trimestre ?",
        "Quelle sous-famille contribue le plus au CA de RENEE BLANCHE ?",
        "Quels articles nécessitent une promotion ?",
    ],
    business_rules=[
        "Couverture_Stock > 6 mois → Surstock → Action: Promo/Déstockage",
        "Couverture_Stock < 1 mois → Rupture → Action: Réassort prioritaire",
        "Variation_Qte% < -15% → Baisse ventes → Action: Enquête cause",
        "Variation_Qte% > +20% → Forte demande → Action: Réévaluer stock",
        "Nb_Clients_Actifs ↓ → Perte intérêt → Action: Marketing ciblé",
    ]
)

VENTES_MENSUELLES_SCHEMA = TableSchema(
    name="t_ventes_mensuelles",
    logical_name="ventes_mensuelles",
    description="Agrégation mensuelle des ventes avec indicateurs de performance",
    columns=[
        ColumnDefinition(
            name="Code_Client",
            type="TEXT",
            description="Identifiant client",
            is_id=True
        ),
        ColumnDefinition(
            name="Categorie_Client",
            type="TEXT",
            description="Type de client",
            examples=["GMS", "Superette"]
        ),
        ColumnDefinition(
            name="Marque",
            type="TEXT",
            description="Marque du produit",
            examples=["H.ZONE", "RENEE BLANCHE"]
        ),
        ColumnDefinition(
            name="Famille",
            type="TEXT",
            description="Famille de produit",
            examples=["Cheveux", "Maquillage"]
        ),
        ColumnDefinition(
            name="Sous_Famille",
            type="TEXT",
            description="Sous-famille de produit",
            examples=["Shampooing", "Gel"]
        ),
        ColumnDefinition(
            name="Référence_Article",
            type="TEXT",
            description="Code article",
            is_id=True
        ),
        ColumnDefinition(
            name="Désignation",
            type="TEXT",
            description="Nom du produit"
        ),
        ColumnDefinition(
            name="Année",
            type="INTEGER",
            description="Année de la vente",
            is_numeric=True,
            examples=["2024", "2025"]
        ),
        ColumnDefinition(
            name="Mois",
            type="INTEGER",
            description="Mois de la vente (1-12)",
            is_numeric=True,
            examples=["1", "10"]
        ),
        ColumnDefinition(
            name="CA_HT_NET",
            type="REAL",
            description="Chiffre d'affaires HT net mensuel",
            is_numeric=True,
            examples=["10000.00", "25000.00"]
        ),
        ColumnDefinition(
            name="Qte_Vendu",
            type="REAL",
            description="Quantité vendue mensuelle",
            is_numeric=True,
            examples=["100", "250"]
        ),
        ColumnDefinition(
            name="CA_Mois_Precedent",
            type="REAL",
            description="CA du mois précédent",
            is_numeric=True
        ),
        ColumnDefinition(
            name="Qte_Mois_Precedent",
            type="REAL",
            description="Quantité du mois précédent",
            is_numeric=True
        ),
        ColumnDefinition(
            name="ΔCA_Mois",
            type="REAL",
            description="Variation absolue du CA (mois)",
            is_numeric=True
        ),
        ColumnDefinition(
            name="ΔCA_%",
            type="REAL",
            description="Variation relative du CA (%)",
            is_numeric=True,
            examples=["-10.5", "+15.0"]
        ),
        ColumnDefinition(
            name="ΔQte_Mois",
            type="REAL",
            description="Variation absolue quantité",
            is_numeric=True
        ),
        ColumnDefinition(
            name="ΔQte_%",
            type="REAL",
            description="Variation relative quantité (%)",
            is_numeric=True
        ),
        ColumnDefinition(
            name="ΔCA_Année",
            type="REAL",
            description="Variation CA vs même mois N-1",
            is_numeric=True
        ),
        ColumnDefinition(
            name="ΔQte_Année",
            type="REAL",
            description="Variation quantité vs N-1",
            is_numeric=True
        ),
        ColumnDefinition(
            name="Tendance_3m",
            type="REAL",
            description="Moyenne mobile sur 3 mois",
            is_numeric=True
        ),
    ],
    sample_questions=[
        "Quel est le chiffre d'affaires total du mois dernier ?",
        "Quels clients ont perdu plus de 10% de CA en octobre ?",
        "Quelle famille de produits est en baisse depuis 3 mois ?",
        "Quels produits n'ont pas été vendus depuis juin ?",
        "Compare les ventes de septembre 2025 et septembre 2024",
        "Top 5 des sous-familles en croissance ce trimestre",
    ],
    business_rules=[
        "ΔCA% < -10% → Alerte baisse → Relance commerciale",
        "ΔQte% < -15% → Baisse volume → Action promo",
        "CA_HT_NET = 0 et durée > 3 mois → Inactif",
        "ΔCA_Année < 0 → Alerte structurelle",
    ]
)

STOCK_SCHEMA = TableSchema(
    name="t_stock",
    logical_name="stock",
    description="État des stocks produits à date donnée",
    columns=[
        ColumnDefinition(
            name="Référence_Article",
            type="TEXT",
            description="Code produit",
            is_id=True
        ),
        ColumnDefinition(
            name="Désignation",
            type="TEXT",
            description="Nom du produit"
        ),
        ColumnDefinition(
            name="Famille",
            type="TEXT",
            description="Famille produit"
        ),
        ColumnDefinition(
            name="Sous_Famille",
            type="TEXT",
            description="Sous-famille produit"
        ),
        ColumnDefinition(
            name="Marque",
            type="TEXT",
            description="Marque du produit"
        ),
        ColumnDefinition(
            name="Depot",
            type="TEXT",
            description="Lieu de stockage",
            examples=["PRINCIPAL", "SECONDAIRE"]
        ),
        ColumnDefinition(
            name="Stock_Terme",
            type="REAL",
            description="Stock physique actuel",
            is_numeric=True,
            examples=["500", "1000", "0"]
        ),
        ColumnDefinition(
            name="Stock_Mois_Precedent",
            type="REAL",
            description="Stock du mois précédent",
            is_numeric=True
        ),
        ColumnDefinition(
            name="ΔStock_%",
            type="REAL",
            description="Variation stock en %",
            is_numeric=True
        ),
        ColumnDefinition(
            name="Qte_Vendue_Mois",
            type="REAL",
            description="Quantité vendue sur période",
            is_numeric=True
        ),
        ColumnDefinition(
            name="Rotation_Mensuelle",
            type="REAL",
            description="Moyenne ventes 3 mois",
            is_numeric=True
        ),
        ColumnDefinition(
            name="Couverture_Stock",
            type="REAL",
            description="Stock_Terme / Rotation (mois)",
            is_numeric=True,
            business_meaning="Durée du stock disponible"
        ),
        ColumnDefinition(
            name="Alerte_Stock",
            type="TEXT",
            description="Rupture, Surstock, Normal",
            examples=["Rupture", "Surstock", "Normal"]
        ),
        ColumnDefinition(
            name="Date_Releve",
            type="DATE",
            description="Date du relevé de stock",
            is_date=True
        ),
    ],
    sample_questions=[
        "Quel est le stock du produit ARGAN 250ML ?",
        "Quels produits ont une couverture < 1 mois ?",
        "Quels articles ont un stock de plus de 6 mois ?",
        "Comment a évolué le stock de la famille Coiffage ?",
        "Quel est le taux de rotation moyen par marque ?",
        "Ce produit est vendu alors qu'il n'y a plus de stock ?",
        "Le dépôt principal a-t-il du stock pour ce produit ?",
    ],
    business_rules=[
        "Stock_Terme = 0 → Rupture totale → Réassort immédiat",
        "Couverture_Stock < 1 → Risque rupture → Priorité achat",
        "1 <= Couverture <= 3 → Zone optimale",
        "Couverture_Stock > 6 → Surstock → Déstockage/promo",
        "ΔStock% > +50% → Hausse anormale → Surproduction",
    ]
)

# All schemas registry
ALL_SCHEMAS = {
    "clients": CLIENT_SCHEMA,
    "produits": PRODUCT_SCHEMA,
    "ventes_mensuelles": VENTES_MENSUELLES_SCHEMA,
    "stock": STOCK_SCHEMA,
    "ventes_cleann": TableSchema(  # Backward compatibility
        name="t_ventes_cleann",
        logical_name="ventes_cleann",
        description="Table des ventes détaillées (vue transactionnelle)",
        columns=[
            ColumnDefinition(name=col, type="TEXT", description="")
            for col in [
                "Client_Principal", "Code_Client", "Intitule_client",
                "Categorie_Client", "Pays", "Zone", "Gouvernorat",
                "Adresse", "Representant", "NDocument", "Type_Document",
                "Date", "Mois", "Annee", "Marque", "Famille", "Sous_Famille",
                "Ref_Article", "Designation", "Qte_Vendu", "CA_HT_BRUT",
                "Tx_Remise", "CA_HT_NET"
            ]
        ]
    )
}


def get_schema(table_name: str) -> TableSchema:
    """Get schema definition for a table"""
    return ALL_SCHEMAS.get(table_name)


def get_all_schemas() -> Dict[str, TableSchema]:
    """Get all table schemas"""
    return ALL_SCHEMAS.copy()


def format_schema_for_llm(schemas: List[str] = None) -> str:
    """
    Format schemas as text for LLM prompts.
    If schemas list is provided, only include those tables.
    """
    if schemas is None:
        schemas = list(ALL_SCHEMAS.keys())
    
    lines = ["=== TABLES DISPONIBLES ===\n"]
    
    for schema_name in schemas:
        schema = ALL_SCHEMAS.get(schema_name)
        if not schema:
            continue
        
        lines.append(f"\n## TABLE: {schema.logical_name}")
        lines.append(f"Description: {schema.description}")
        lines.append(f"Nom physique: {schema.name}")
        
        if schema.primary_key:
            lines.append(f"Clé primaire: {schema.primary_key}")
        
        lines.append("\nColonnes:")
        for col in schema.columns:
            line = f"  - {col.name} ({col.type}): {col.description}"
            if col.business_meaning:
                line += f" | {col.business_meaning}"
            if col.examples:
                line += f" | Ex: {', '.join(col.examples[:2])}"
            lines.append(line)
        
        if schema.business_rules:
            lines.append("\nRègles métier:")
            for rule in schema.business_rules[:3]:
                lines.append(f"  • {rule}")
        
        lines.append("")
    
    return "\n".join(lines)


def get_column_list(table_name: str) -> List[str]:
    """Get list of column names for a table"""
    schema = get_schema(table_name)
    if not schema:
        return []
    return [col.name for col in schema.columns]


def infer_entity_type(question: str, schema_names: List[str] = None) -> str:
    """
    Infer which entity type the question is about.
    Returns: 'clients', 'produits', 'ventes_mensuelles', 'stock', or 'general'
    """
    question_lower = question.lower()
    
    # Client keywords
    client_keywords = [
        "client", "customer", "clientèle", "acheteur", "inactif",
        "fidèle", "fidélité", "commercial", "représentant"
    ]
    
    # Product keywords
    product_keywords = [
        "produit", "article", "référence", "famille", "sous-famille",
        "marque", "h.zone", "renee blanche", "shampooing", "maquillage",
        "coiffage", "désignation"
    ]
    
    # Stock keywords
    stock_keywords = [
        "stock", "rupture", "couverture", "réassort", "inventaire",
        "surstock", "dépôt", "disponible"
    ]
    
    # Sales keywords
    sales_keywords = [
        "vente", "ca", "chiffre d'affaires", "revenu", "mensuel",
        "annuel", "trimestre", "évolution", "croissance", "baisse"
    ]
    
    # Count matches
    client_score = sum(1 for kw in client_keywords if kw in question_lower)
    product_score = sum(1 for kw in product_keywords if kw in question_lower)
    stock_score = sum(1 for kw in stock_keywords if kw in question_lower)
    sales_score = sum(1 for kw in sales_keywords if kw in question_lower)
    
    # Return highest scoring category
    scores = {
        "clients": client_score,
        "produits": product_score,
        "stock": stock_score,
        "ventes_mensuelles": sales_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return "general"
    
    # If stock and product both high, prefer stock
    if stock_score > 0 and product_score > 0:
        return "stock"
    
    return max(scores.items(), key=lambda x: x[1])[0]
