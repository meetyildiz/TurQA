from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from TurQA.process import prepare_train_features, prepare_validation_features
from TurQA.answer import postprocess_qa_predictions
from TurQA.settings import *



def train(num_train_epochs=1, push_to_hub=False):
    datasets = load_dataset(dataset, download_mode='force_redownload', ignore_verifications=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)



    tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)


    args = TrainingArguments(
        f"TurQA-{model_name}-finetuned-toqad",
        evaluation_strategy = "epoch",
        learning_rate=2e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=push_to_hub,
    )



    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )


    trainer.train()


    trainer.save_model(f"TurQA-{model_name}-finetuned-toqad")




    validation_features = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["validation"].column_names
    )



    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))



    final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)


    metric = load_metric("squad_v2" if squad_v2 else "squad")



    if squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    metrics = metric.compute(predictions=formatted_predictions, references=references)

    print(metrics)


    if push_to_hub:
        trainer.push_to_hub()
    
    return metrics
#from transformers import AutoModelForQuestionAnswering
#model = AutoModelForQuestionAnswering.from_pretrained("sgugger/my-awesome-model")
