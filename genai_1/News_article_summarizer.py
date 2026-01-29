from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize_news(article):
    inputs = tokenizer(
        article,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ---- INPUT ----
print("Paste news article (type END on a new line to finish):")
lines = []

while True:
    line = input()
    if line.strip().upper() == "END":
        break
    lines.append(line)

article = " ".join(lines)

print("\n--- News Summary ---\n")
print(summarize_news(article))
