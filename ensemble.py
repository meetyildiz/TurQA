import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

models_dict = {
    "bert": ("dbmdz/bert-base-turkish-cased", "/content/drive/MyDrive/QA_Models/TurQA-bert-base-turkish-cased-finetuned-toqad-aug"),
    "bert-128k": ("dbmdz/bert-base-turkish-128k-cased", "/content/drive/MyDrive/QA_Models/TurQA-bert-base-turkish-128k-cased-finetuned-toqad-aug"),
    "convbert": ("dbmdz/convbert-base-turkish-cased", "/content/drive/MyDrive/QA_Models/TurQA-convbert-base-turkish-cased-finetuned-toqad-aug"),
    "electra": ("dbmdz/electra-base-turkish-cased-discriminator", "/content/drive/MyDrive/QA_Models/TurQA-electra-base-turkish-cased-discriminator-finetuned-toqad-aug"),
    "xlm": ("xlm-roberta-base", "/content/drive/MyDrive/QA_Models/TurQA-xlm-roberta-base-finetuned-toqad-aug"),
}



dataset = "meetyildiz/toqad-aug"
datasets = load_dataset(dataset)

metric = load_metric("squad")
references_valid = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
references_test = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["test"]]




def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


from tqdm.auto import tqdm
import collections
import numpy as np

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()
    possible_ans = []
    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        predictions[example["id"]] = best_answer["text"]

        possible_ans.append({"id": example_index, "answers": valid_answers})
    return predictions, possible_ans




valid_results_dict = {}
test_results_dict = {}


for m_key in models_dict:
    print("*"*20)
    print(m_key)
    tok_name = models_dict[m_key][0]
    model_name = models_dict[m_key][1]

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    pad_on_right = tokenizer.padding_side == "right"

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    max_length = 512 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    batch_size = 16

    tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

    #dataset_name = dataset.split("/")[-1]
    args = TrainingArguments(
        f"TurQA-{model_name}-finetuned-toqad-aug",
        evaluation_strategy = "steps",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )


    from transformers import default_data_collator

    data_collator = default_data_collator


    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    validation_features = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["validation"].column_names
    )
    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    final_predictions, possible_ans = postprocess_qa_predictions(datasets["validation"], validation_features, raw_predictions.predictions)
    valid_results_dict[m_key] = (final_predictions, possible_ans)

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    valid_metric = metric.compute(predictions=formatted_predictions, references=references_valid)

    print(valid_metric)

    validation_features = datasets["test"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["test"].column_names
    )
    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    final_predictions, possible_ans = postprocess_qa_predictions(datasets["test"], validation_features, raw_predictions.predictions)
    test_results_dict[m_key] = (final_predictions, possible_ans)

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    test_metric = metric.compute(predictions=formatted_predictions, references=references_test)

    print(test_metric)

from operator import itemgetter
import pandas as pd


def get_ensemble_scores(references, results_dict):
    ensemble_answers = []
    i = 0
    for sampl in references:
        id = sampl["id"]
        

        for model_key in models_dict:

            ordered_scores = sorted(results_dict[model_key][1][i]["answers"], key=itemgetter('score'), reverse=True)[0:20]
            for item in ordered_scores: item['id']=id
            for item in ordered_scores: item['model']=model_key
            ensemble_answers.extend(ordered_scores)

        i = i + 1

    return pd.DataFrame(ensemble_answers)


valid_ensemble_scores = get_ensemble_scores(references_valid, valid_results_dict)
test_ensemble_scores = get_ensemble_scores(references_test, test_results_dict)



valid_mean = valid_ensemble_scores.pivot_table(index=["id", "text"], columns = "model", values= "score", aggfunc='mean').fillna(-10).stack().reset_index(name='score').groupby(["id", "text"])["score"].mean().reset_index()
test_mean = test_ensemble_scores.pivot_table(index=["id", "text"], columns = "model", values= "score", aggfunc='mean').fillna(-10).stack().reset_index(name='score').groupby(["id", "text"])["score"].mean().reset_index()

final_valid_mean = valid_mean.loc[valid_mean.reset_index().groupby(['id'])['score'].idxmax()][["id", "text"]]
final_test_mean = test_mean.loc[test_mean.reset_index().groupby(['id'])['score'].idxmax()][["id", "text"]]

final_valid_mean = [{"id": v["id"], "prediction_text": v["text"]} for k, v in final_valid_mean.iterrows()]
final_test_mean = [{"id": v["id"], "prediction_text": v["text"]} for k, v in final_test_mean.iterrows()]


print(metric.compute(predictions=final_valid_mean, references=references_valid))
print(metric.compute(predictions=final_test_mean, references=references_test))


final_valid_max = valid_ensemble_scores.loc[valid_ensemble_scores.reset_index().groupby(['id'])['score'].idxmax()][["id", "text"]]
final_test_max = test_ensemble_scores.loc[test_ensemble_scores.reset_index().groupby(['id'])['score'].idxmax()][["id", "text"]]

final_valid_max = [{"id": v["id"], "prediction_text": v["text"]} for k, v in final_valid_max.iterrows()]
final_test_max = [{"id": v["id"], "prediction_text": v["text"]} for k, v in final_test_max.iterrows()]

print(metric.compute(predictions=final_valid_max, references=references_valid))
print(metric.compute(predictions=final_test_max, references=references_test))


valid_sum = valid_ensemble_scores.groupby(["id", "text"])["score"].sum().reset_index()
test_sum = test_ensemble_scores.groupby(["id", "text"])["score"].sum().reset_index()

final_valid_sum = valid_sum.loc[valid_sum.reset_index().groupby(['id'])['score'].idxmax()][["id", "text"]]
final_test_sum = test_sum.loc[test_sum.reset_index().groupby(['id'])['score'].idxmax()][["id", "text"]]

final_valid_sum = [{"id": v["id"], "prediction_text": v["text"]} for k, v in final_valid_sum.iterrows()]
final_test_sum = [{"id": v["id"], "prediction_text": v["text"]} for k, v in final_test_sum.iterrows()]

print(metric.compute(predictions=final_valid_sum, references=references_valid))
print(metric.compute(predictions=final_test_sum, references=references_test))