from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from .config import MODEL_NAME, DATASET_PATH, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, BATCH_SIZE, NUM_EPOCHS, OUTPUT_DIR

def preprocess_function(examples, tokenizer):
    """
    Tokenizes the input and target texts for BART.
    """
    inputs = examples['text']
    targets = examples['summary']
    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Load dataset
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["text", "summary"]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        evaluation_strategy="no",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
