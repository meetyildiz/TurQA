squad_v2 = False
model_checkpoint = "dbmdz/bert-base-turkish-cased"
dataset = "meetyildiz/toqad"
batch_size = 256
max_length = 512 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
n_best_size = 20
max_answer_length = 32

model_name = model_checkpoint.split("/")[-1]
dataset_name = dataset.split("/")[-1]