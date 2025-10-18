import traceback
from rag_chatbot import chatbot

if __name__ == '__main__':
    try:
        res = chatbot.chat('what product got best prediction for next year ?', thread_id='default')
        print('CHAT RESULT:', res)
    except Exception as e:
        print('CHAT FAILED:', e)
        traceback.print_exc()
