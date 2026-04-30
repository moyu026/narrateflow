import os

from google import genai


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    print("Available Gemini models:")
    for index, model in enumerate(client.models.list(), start=1):
        actions = getattr(model, "supported_actions", None)
        if actions is None:
            actions = getattr(model, "supported_generation_methods", [])
        print(f"- {model.name}: {', '.join(actions)}")
        if index >= 10:
            break

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly: ok",
    )
    print(f"\nGenerate content test: {response.text.strip()}")


if __name__ == "__main__":
    main()
