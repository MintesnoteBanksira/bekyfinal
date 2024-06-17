import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .utils import preprocess_input, extract_symptoms, predict_intent, chat_with_bot, predict_model, symptoms_list

stored_symptoms = []

@csrf_exempt
def get_symptoms(request):
    symptoms_formatted = [symptom.replace("_", " ") for symptom in symptoms_list]
    return JsonResponse(symptoms_formatted, safe=False)

from .utils import chat_with_bot, preprocess_input
from asgiref.sync import async_to_sync

@csrf_exempt
@require_POST
def chat(request):
    try:
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        user_input = body.get('message', '')
        print(f"User input: {user_input}")

        # Use the chat_with_bot function
        rasa_responses = async_to_sync(chat_with_bot)(user_input)
        if rasa_responses:
            message = [resp.get("text") for resp in rasa_responses if "text" in resp]
            return JsonResponse({
                'response': message,
            })

        return JsonResponse({
            'response': "I could not process your input properly.",
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
