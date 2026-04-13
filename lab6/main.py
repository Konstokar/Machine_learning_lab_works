from llm import ask_llama


def chat():
    print("LLaMA Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        answer = ask_llama(user_input)

        print("AI:", answer, "\n")


if __name__ == "__main__":
    chat()