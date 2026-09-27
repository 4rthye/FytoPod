import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def get_treatment(disease: str, severity: str) -> dict:
    """
    Calls LLaMA 3 via Groq to get treatment recommendations
    for a detected plant disease and severity level.
    """
    treatment_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a plant disease expert. Give concise treatment recommendations."
            },
            {
                "role": "user",
                "content": (
                    f"Plant has {disease} at {severity} severity. "
                    f"Give 3 treatment recommendations in bullet points."
                )
            }
        ],
        max_tokens=250
    )
    treatment = treatment_response.choices[0].message.content

    followup_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate 4 follow-up questions a user might ask about plant disease treatment."
            },
            {
                "role": "user",
                "content": (
                    f"Disease: {disease}, Severity: {severity}, "
                    f"Treatment: {treatment}. Generate 4 numbered follow-up questions."
                )
            }
        ],
        max_tokens=200
    )
    followups = followup_response.choices[0].message.content

    return {
        "disease":    disease,
        "severity":   severity,
        "treatment":  treatment,
        "followups":  followups
    }


def answer_followup(disease: str, severity: str,
                    treatment: str, question: str) -> str:
    """
    Handles a follow-up question from the user about
    the diagnosed disease and recommended treatment.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a plant disease expert. Answer follow-up questions "
                    "about plant disease treatment concisely and practically."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Disease: {disease}, Severity: {severity}. "
                    f"Treatment given: {treatment}. "
                    f"User question: {question}"
                )
            }
        ],
        max_tokens=200
    )
    return response.choices[0].message.content