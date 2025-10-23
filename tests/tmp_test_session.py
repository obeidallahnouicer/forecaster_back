import sys
sys.path.append(r'c:\Users\onouicer\Desktop\slimback\forecaster_back')
from core.registry import REGISTRY
from core.context_manager import get_context_manager

info = REGISTRY.create_session_from_file('forecast-summary.csv')
print('Created session id:', info.session_id)
cm = get_context_manager()
ctx = cm.get_or_create_context(info.session_id, query='First question')
print('Context query_history after first:', ctx.query_history)
ctx2 = cm.get_or_create_context(info.session_id, query='Second question')
print('Context query_history after second:', ctx2.query_history)
