# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:39:28.427498Z","iopub.execute_input":"2021-09-03T06:39:28.427884Z","iopub.status.idle":"2021-09-03T06:39:35.295609Z","shell.execute_reply.started":"2021-09-03T06:39:28.427802Z","shell.execute_reply":"2021-09-03T06:39:35.294682Z"},"jupyter":{"outputs_hidden":false}}
# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
import tensorflow as tf
import re
import numpy as np
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:23.601077Z","iopub.execute_input":"2021-09-03T06:40:23.601427Z","iopub.status.idle":"2021-09-03T06:40:26.644174Z","shell.execute_reply.started":"2021-09-03T06:40:23.601396Z","shell.execute_reply":"2021-09-03T06:40:26.643193Z"},"jupyter":{"outputs_hidden":false}}
!unzip ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:26.645923Z","iopub.execute_input":"2021-09-03T06:40:26.646288Z","iopub.status.idle":"2021-09-03T06:40:27.532988Z","shell.execute_reply.started":"2021-09-03T06:40:26.646232Z","shell.execute_reply":"2021-09-03T06:40:27.532052Z"},"jupyter":{"outputs_hidden":false}}
df=pd.read_csv('train.csv')

df = df.sample(frac=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:27.534664Z","iopub.execute_input":"2021-09-03T06:40:27.534968Z","iopub.status.idle":"2021-09-03T06:40:27.559963Z","shell.execute_reply.started":"2021-09-03T06:40:27.534934Z","shell.execute_reply":"2021-09-03T06:40:27.559056Z"},"jupyter":{"outputs_hidden":false}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:28.093852Z","iopub.execute_input":"2021-09-03T06:40:28.094177Z","iopub.status.idle":"2021-09-03T06:40:28.101234Z","shell.execute_reply.started":"2021-09-03T06:40:28.094148Z","shell.execute_reply":"2021-09-03T06:40:28.100291Z"},"jupyter":{"outputs_hidden":false}}
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:29.992031Z","iopub.execute_input":"2021-09-03T06:40:29.992399Z","iopub.status.idle":"2021-09-03T06:40:42.571652Z","shell.execute_reply.started":"2021-09-03T06:40:29.992355Z","shell.execute_reply":"2021-09-03T06:40:42.570752Z"},"jupyter":{"outputs_hidden":false}}
df['comment_text'] = df['comment_text'].map(lambda x : clean_text(x))

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:42.572982Z","iopub.execute_input":"2021-09-03T06:40:42.573313Z","iopub.status.idle":"2021-09-03T06:40:42.602675Z","shell.execute_reply.started":"2021-09-03T06:40:42.573262Z","shell.execute_reply":"2021-09-03T06:40:42.601783Z"},"jupyter":{"outputs_hidden":false}}
train_sentences = df["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_y = df[list_classes].values

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:40:42.60558Z","iopub.execute_input":"2021-09-03T06:40:42.605931Z","iopub.status.idle":"2021-09-03T06:40:42.619542Z","shell.execute_reply.started":"2021-09-03T06:40:42.605897Z","shell.execute_reply":"2021-09-03T06:40:42.618623Z"},"jupyter":{"outputs_hidden":false}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:41:56.787013Z","iopub.execute_input":"2021-09-03T06:41:56.787372Z","iopub.status.idle":"2021-09-03T06:42:26.724167Z","shell.execute_reply.started":"2021-09-03T06:41:56.78734Z","shell.execute_reply":"2021-09-03T06:42:26.723258Z"},"jupyter":{"outputs_hidden":false}}
# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 128

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
#config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:42:26.725885Z","iopub.execute_input":"2021-09-03T06:42:26.726454Z","iopub.status.idle":"2021-09-03T06:42:31.196128Z","shell.execute_reply.started":"2021-09-03T06:42:26.726415Z","shell.execute_reply":"2021-09-03T06:42:31.195211Z"},"jupyter":{"outputs_hidden":false}}
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
x = bert.bert(inputs)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:31:32.576672Z","iopub.execute_input":"2021-09-03T06:31:32.576998Z","iopub.status.idle":"2021-09-03T06:31:32.583775Z","shell.execute_reply.started":"2021-09-03T06:31:32.576967Z","shell.execute_reply":"2021-09-03T06:31:32.582781Z"},"jupyter":{"outputs_hidden":false}}
x

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:42:53.212177Z","iopub.execute_input":"2021-09-03T06:42:53.212684Z","iopub.status.idle":"2021-09-03T06:42:53.263293Z","shell.execute_reply.started":"2021-09-03T06:42:53.212646Z","shell.execute_reply":"2021-09-03T06:42:53.262506Z"},"jupyter":{"outputs_hidden":false}}
#x2 =Dense(512, activation='relu')(x[1])
x2 = GlobalAveragePooling1D()(x[0])
#x3 = Dropout(0.5)(x2)
y =Dense(len(list_classes), activation='sigmoid', name='outputs')(x2)

model = Model(inputs=inputs, outputs=y)
#model.layers[2].trainable = False

# Take a look at the model
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:45:12.630973Z","iopub.execute_input":"2021-09-03T06:45:12.631346Z","iopub.status.idle":"2021-09-03T06:45:12.651626Z","shell.execute_reply.started":"2021-09-03T06:45:12.631313Z","shell.execute_reply":"2021-09-03T06:45:12.650738Z"},"jupyter":{"outputs_hidden":false}}
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:45:15.533989Z","iopub.execute_input":"2021-09-03T06:45:15.534344Z","iopub.status.idle":"2021-09-03T06:45:54.749497Z","shell.execute_reply.started":"2021-09-03T06:45:15.534313Z","shell.execute_reply":"2021-09-03T06:45:54.748573Z"},"jupyter":{"outputs_hidden":false}}
# Tokenize the input 
x = tokenizer(
    text=list(train_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T06:45:54.750916Z","iopub.execute_input":"2021-09-03T06:45:54.751251Z","iopub.status.idle":"2021-09-03T07:22:24.080812Z","shell.execute_reply.started":"2021-09-03T06:45:54.751215Z","shell.execute_reply":"2021-09-03T07:22:24.079967Z"},"jupyter":{"outputs_hidden":false}}
history = model.fit(
    x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    #x={'input_ids': x['input_ids']},
    y={'outputs': train_y},
    validation_split=0.1,
    batch_size=32,
    epochs=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:22:45.174842Z","iopub.execute_input":"2021-09-03T07:22:45.175181Z","iopub.status.idle":"2021-09-03T07:22:57.283863Z","shell.execute_reply.started":"2021-09-03T07:22:45.175149Z","shell.execute_reply":"2021-09-03T07:22:57.282987Z"},"jupyter":{"outputs_hidden":false}}
test_df=pd.read_csv("test.csv")
test_df['comment_text']=test_df['comment_text'].map(lambda x : clean_text(x))
test_sentences = test_df["comment_text"].fillna("CVxTz").values

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:22:57.285423Z","iopub.execute_input":"2021-09-03T07:22:57.285768Z","iopub.status.idle":"2021-09-03T07:23:33.034983Z","shell.execute_reply.started":"2021-09-03T07:22:57.285733Z","shell.execute_reply":"2021-09-03T07:23:33.034093Z"},"jupyter":{"outputs_hidden":false}}
test_x = tokenizer(
    text=list(test_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:23:33.03734Z","iopub.execute_input":"2021-09-03T07:23:33.037678Z","iopub.status.idle":"2021-09-03T07:23:34.301662Z","shell.execute_reply.started":"2021-09-03T07:23:33.037645Z","shell.execute_reply":"2021-09-03T07:23:34.299497Z"},"jupyter":{"outputs_hidden":false}}
del test_sentences
del x
del df

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:23:34.303314Z","iopub.execute_input":"2021-09-03T07:23:34.303711Z","iopub.status.idle":"2021-09-03T07:23:34.553147Z","shell.execute_reply.started":"2021-09-03T07:23:34.303662Z","shell.execute_reply":"2021-09-03T07:23:34.552283Z"},"jupyter":{"outputs_hidden":false}}
import gc
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:23:34.554746Z","iopub.execute_input":"2021-09-03T07:23:34.555357Z","iopub.status.idle":"2021-09-03T07:23:34.569373Z","shell.execute_reply.started":"2021-09-03T07:23:34.555311Z","shell.execute_reply":"2021-09-03T07:23:34.568544Z"},"jupyter":{"outputs_hidden":false}}
test_x

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:23:34.570659Z","iopub.execute_input":"2021-09-03T07:23:34.571021Z","iopub.status.idle":"2021-09-03T07:36:17.039007Z","shell.execute_reply.started":"2021-09-03T07:23:34.570977Z","shell.execute_reply":"2021-09-03T07:36:17.038076Z"},"jupyter":{"outputs_hidden":false}}
predictions=model.predict(x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']},batch_size=32)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:36:52.858151Z","iopub.execute_input":"2021-09-03T07:36:52.858525Z","iopub.status.idle":"2021-09-03T07:36:52.865317Z","shell.execute_reply.started":"2021-09-03T07:36:52.858487Z","shell.execute_reply":"2021-09-03T07:36:52.864295Z"},"jupyter":{"outputs_hidden":false}}
predictions

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:37:04.092956Z","iopub.execute_input":"2021-09-03T07:37:04.09328Z","iopub.status.idle":"2021-09-03T07:37:05.524104Z","shell.execute_reply.started":"2021-09-03T07:37:04.093239Z","shell.execute_reply":"2021-09-03T07:37:05.523291Z"},"jupyter":{"outputs_hidden":false}}
submission=pd.DataFrame(predictions,columns=list_classes)
submission['id'] = test_df['id']
submission=submission[['id']+(list_classes)]
submission.to_csv("submission.csv", index=False)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:37:10.173743Z","iopub.execute_input":"2021-09-03T07:37:10.174066Z","iopub.status.idle":"2021-09-03T07:37:10.192606Z","shell.execute_reply.started":"2021-09-03T07:37:10.174035Z","shell.execute_reply":"2021-09-03T07:37:10.191666Z"},"jupyter":{"outputs_hidden":false}}
submission

# %% [code] {"execution":{"iopub.status.busy":"2021-09-03T07:42:33.336242Z","iopub.execute_input":"2021-09-03T07:42:33.336657Z","iopub.status.idle":"2021-09-03T07:42:34.153284Z","shell.execute_reply.started":"2021-09-03T07:42:33.336624Z","shell.execute_reply":"2021-09-03T07:42:34.152198Z"},"jupyter":{"outputs_hidden":false}}
!ls

# %% [code] {"jupyter":{"outputs_hidden":false}}
