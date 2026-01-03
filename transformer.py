path = kagglehub.dataset_download(
    "roblexnana/the-babi-tasks-for-nlp-qa-system"
)


print(path)


BASE_PATH = "/kaggle/input/the-babi-tasks-for-nlp-qa-system"


import os

train_file = os.path.join(
    BASE_PATH,
    "tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt"
)

test_file = os.path.join(
    BASE_PATH,
    "tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt"
)

print(train_file)
print(test_file)


# STEP 2: Parse bAbI dataset
def parse_babi(file_path):
    stories, questions, answers = [], [], []
    story = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            nid, text = line.split(" ", 1)

            if nid == "1":
                story = []

            if "\t" in text:
                q, a, _ = text.split("\t")
                stories.append(" ".join(story))
                questions.append(q)
                answers.append(a)
            else:
                story.append(text)

    return stories, questions, answers


train_stories, train_questions, train_answers = parse_babi(train_file)
test_stories, test_questions, test_answers = parse_babi(test_file)

print("Train samples:", len(train_stories))
print("Test samples :", len(test_stories))


# STEP 3: Tokenization & padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_VOCAB = 10000

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_stories + train_questions)

max_story_len = max(len(s.split()) for s in train_stories)
max_question_len = max(len(q.split()) for q in train_questions)

def vectorize(stories, questions, answers):
    s = tokenizer.texts_to_sequences(stories)
    q = tokenizer.texts_to_sequences(questions)

    a = np.array([
        tokenizer.word_index.get(ans, 0) for ans in answers
    ])

    s = pad_sequences(s, maxlen=max_story_len)
    q = pad_sequences(q, maxlen=max_question_len)

    return s, q, a


x_story, x_question, y = vectorize(
    train_stories, train_questions, train_answers
)

x_story_test, x_question_test, y_test = vectorize(
    test_stories, test_questions, test_answers
)

print("Vectorization done ✅")


# STEP 4: Build Transformer-based QA model
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    Concatenate
)
from tensorflow.keras.models import Model


# Transformer Encoder
def transformer_encoder(x, head_size, num_heads, ff_dim):
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size
    )(x, x)

    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(x.shape[-1])(ffn)

    return LayerNormalization(epsilon=1e-6)(x + ffn)


# Model
embed_dim = 64
vocab_size = MAX_VOCAB

story_input = Input(shape=(max_story_len,))
question_input = Input(shape=(max_question_len,))

embedding = Embedding(vocab_size, embed_dim)

story_emb = embedding(story_input)
question_emb = embedding(question_input)

story_encoded = transformer_encoder(
    story_emb, head_size=32, num_heads=2, ff_dim=64
)

question_encoded = transformer_encoder(
    question_emb, head_size=32, num_heads=2, ff_dim=64
)

qa_attention = MultiHeadAttention(
    num_heads=2, key_dim=32
)(
    query=question_encoded,
    key=story_encoded,
    value=story_encoded
)

story_vec = GlobalAveragePooling1D()(qa_attention)
question_vec = GlobalAveragePooling1D()(question_encoded)

merged = Concatenate()([story_vec, question_vec])

output = Dense(vocab_size, activation="softmax")(merged)

model = Model(
    inputs=[story_input, question_input],
    outputs=output
)



# STEP 5: Compile, train, evaluate
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


model.fit(
    [x_story, x_question],
    y,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)



loss, acc = model.evaluate(
    [x_story_test, x_question_test],
    y_test
)

print("Final Transformer QA Accuracy:", acc)
