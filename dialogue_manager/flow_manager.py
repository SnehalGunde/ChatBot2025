# def get_response(intent):
#     if intent == "bad_day":
#         return (
#             "That sounds really tough. I'm here for you. "
#             "Would you like to try a calming breathing exercise or talk more about it?"
#         )

#     elif intent == "pain_check":
#         return (
#             "I'm sorry to hear you're in pain. "
#             "On a scale of 1 to 10, how would you rate your pain right now?"
#         )

#     elif intent == "relaxation_request":
#         return (
#             "Of course. Let’s begin a simple relaxation technique. "
#             "Close your eyes, take a deep breath, and let’s breathe together."
#         )

#     elif intent == "exercise_request":
#         return (
#             "I recommend starting with gentle stretches for your neck and shoulders. "
#             "Would you like a video or step-by-step instructions?"
#         )

#     elif intent == "emotional_support":
#         return (
#             "You’re not alone in this. I’m always here to listen. "
#             "Would talking about what you're feeling help a bit right now?"
#         )

#     elif intent == "greeting":
#         return (
#             "Hi there! It’s good to hear from you. How are you feeling physically and emotionally today?"
#         )

#     elif intent == "pain_rating":
#         return (
#             "Thanks for sharing. Based on your rating, I can suggest either gentle stretches or calming techniques. "
#             "Would you prefer movement or mindfulness right now?"
#         )

#     elif intent == "confirm_exercise":
#         return (
#             "Great! Let’s begin. Start by sitting comfortably. "
#             "Take a deep breath in through your nose... and exhale slowly through your mouth... "
#             "Do this for a few minutes. "
#         )

#     elif intent == "unknown":
#         return (
#             "Hmm, I didn't quite understand that. Could you try saying it another way?"
#         )

#     else:
#         return (
#             "I’m here for you, even if I didn’t quite catch that. "
#             "Could you rephrase or let me know how you're feeling?"
#         )

# dialogue_manager/flow_manager.py

user_memory = {
    "last_pain_location": None
}

def extract_pain_location(user_input):
    """Extract common body part from user input for exercise recommendations."""
    user_input = user_input.lower()
    if "shoulder" in user_input:
        return "shoulder"
    elif "foot" in user_input or "feet" in user_input or "leg" in user_input:
        return "foot"
    elif "back" in user_input:
        return "back"
    elif "knee" in user_input or "knees" in user_input:
        return "knee"
    else:
        return None

def get_response(intent, user_input=None):
    # Update memory if user mentions a body part
    if user_input:
        location = extract_pain_location(user_input)
        if location:
            user_memory["last_pain_location"] = location

    if intent == "bad_day":
        return (
            "That sounds really tough. I'm here for you. "
            "Would you like to try a calming breathing exercise or talk more about it?"
        )

    elif intent == "pain_check":
        return (
            "I'm sorry to hear you're in pain. "
            "On a scale of 1 to 10, how would you rate your pain right now?"
        )

    elif intent == "relaxation_request":
        return (
            "Of course. Let’s begin a simple relaxation technique. "
            "Close your eyes, take a deep breath, and let’s breathe together."
        )

    elif intent == "rating_response" or intent == "exercise_request":
        location = user_memory.get("last_pain_location", None)
        if location == "shoulder":
            return "Let's try gentle shoulder rolls and stretches. Would you like step-by-step guidance?"
        elif location == "foot":
            return "Try simple leg raises and ankle rotations. Want me to walk you through it?"
        elif location == "back":
            return "Some light back stretches can help. Would you like a video or written steps?"
        elif location == "knee":
            return "For knee pain, try seated leg extensions and hamstring stretches. Shall we begin?"
        else:
            return "I recommend starting with gentle full-body stretches. Would you like a video or step-by-step instructions?"

    elif intent == "emotional_support":
        return (
            "You’re not alone in this. I’m always here to listen. "
            "Would talking about what you're feeling help a bit right now?"
        )

    elif intent == "greeting":
        return (
            "Hi there! It’s good to hear from you. How are you feeling physically and emotionally today?"
        )

    elif intent == "pain_rating":
        return (
            "Thanks for sharing. Based on your rating, I can suggest either gentle stretches or calming techniques. "
            "Would you prefer movement or mindfulness right now?"
        )

    elif intent == "confirm_exercise":
        return (
            "Great! Let’s begin. Start by sitting comfortably. "
            "Take a deep breath in through your nose... and exhale slowly through your mouth... "
            "Do this for a few minutes."
        )

    elif intent == "unknown":
        return (
            "Hmm, I didn't quite understand that. Could you try saying it another way?"
        )

    else:
        return (
            "I’m here for you, even if I didn’t quite catch that. "
            "Could you rephrase or let me know how you're feeling?"
        )
