"""Deterministic recommender that produces simple recommendations from analysis data.

This module provides rule-based recommendations so the system can offer
actionable suggestions without calling an LLM.
"""
from typing import Dict, Any, List


def produce_recommendations(analysis_data: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    """Produce simple deterministic recommendations based on analysis output.

    Rules (simple examples):
    - Drop candidate: products with stability_score < 0.35 and overall_mean low (bottom quartile)
    - Promote: products with stability_score >= 0.8 and overall_mean in top quartile
    - Monitor: products with disagreement > 0.2
    """
    products = analysis_data.get('products', {})
    if not products:
        return []

    # Compute list of (product, overall_mean)
    means = [(p, v.get('overall_mean', 0.0), v.get('stability_score', 0.0), v.get('disagreement', 0.0)) for p, v in products.items()]
    # sort by mean
    sorted_by_mean = sorted(means, key=lambda x: x[1])
    total = len(sorted_by_mean)
    recommendations = []

    # bottom quartile threshold index
    bottom_idx = max(0, total // 4)
    top_idx = max(0, total - (total // 4))

    for i, (p, mean, stability, disagreement) in enumerate(sorted_by_mean):
        rec = None
        if stability < 0.35 and i <= bottom_idx:
            rec = {
                'product': p,
                'action': 'drop_candidate',
                'reason': f'Low stability ({stability:.2f}) and low forecast mean ({mean:.2f})'
            }
        elif stability >= 0.8 and i >= top_idx:
            rec = {
                'product': p,
                'action': 'promote',
                'reason': f'High stability ({stability:.2f}) and strong mean ({mean:.2f})'
            }
        elif disagreement > 0.2:
            rec = {
                'product': p,
                'action': 'monitor',
                'reason': f'High model disagreement ({disagreement:.3f})'
            }

        if rec:
            recommendations.append(rec)

    # If no recommendations derived by rules, fallback to recommending top N by mean
    if not recommendations:
        for p, mean, stability, disagreement in sorted_by_mean[-top_n:][::-1]:
            recommendations.append({'product': p, 'action': 'monitor', 'reason': f'Top by mean ({mean:.2f})'})

    return recommendations
