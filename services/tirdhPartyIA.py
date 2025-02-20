from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


def getAns(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens = 500
    )

    return completion.choices[0].message.content
