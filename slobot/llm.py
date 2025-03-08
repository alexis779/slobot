import openai

class Llm:
    def inference(self, prompt):
        return openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=100
        ).choices[0].text