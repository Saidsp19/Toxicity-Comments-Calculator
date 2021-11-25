from PyQt5 import QtCore, QtGui, QtWidgets

# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

from keras.models import load_model


# Load model 
model = load_model('model.h5')

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

def pre_processing_and_predict(txt_data):
    test_df['comment_text'].map(lambda x : clean_text(x))
    test_sentences = test_df["comment_text"].fillna("CVxTz").values
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
    
    predictions=model.predict(x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']},batch_size=32)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)

        font_title = QtGui.QFont('Arial', 20)
        font_class = QtGui.QFont('Arial', 10)
        font_value = QtGui.QFont('Arial', 10)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 30, 550, 50))
        self.label.setText("AI-Powered Toxicity calculator on  comments")
        self.label.setStyleSheet("background-color: lightgreen")
        self.label.setFont(font_title)

        self.textBrowser = QtWidgets.QTextEdit(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(40, 110, 730, 200))

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(350, 320, 151, 51))
        self.pushButton.setText("Prediction")
        self.pushButton.setStyleSheet("background-color: red")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 400, 120, 20))
        self.label_2.setText("Toxic")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(170, 400, 120, 20))
        self.label_3.setText("Severe toxic")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(290, 400, 120, 20))
        self.label_4.setText("Obscene")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(410, 400, 120, 20))
        self.label_5.setText("Threat")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(530, 400, 120, 20))
        self.label_6.setText("Insult")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(650, 400, 120, 20))
        self.label_7.setText("Identity_hate")

        self.lcdNumber = QtWidgets.QLabel(self.centralwidget)
        self.lcdNumber.setGeometry(QtCore.QRect(50, 460, 120, 30))
        self.lcdNumber.setText("0.91")

        self.lcdNumber_2 = QtWidgets.QLabel(self.centralwidget)
        self.lcdNumber_2.setGeometry(QtCore.QRect(170, 460, 120, 30))
        self.lcdNumber_2.setText("0.75")

        self.lcdNumber_3 = QtWidgets.QLabel(self.centralwidget)
        self.lcdNumber_3.setGeometry(QtCore.QRect(290, 460, 120, 30))
        self.lcdNumber_3.setText("0.120")

        self.lcdNumber_4 = QtWidgets.QLabel(self.centralwidget)
        self.lcdNumber_4.setGeometry(QtCore.QRect(410, 460, 120, 30))
        self.lcdNumber_4.setText("0.81")

        self.lcdNumber_5 = QtWidgets.QLabel(self.centralwidget)
        self.lcdNumber_5.setGeometry(QtCore.QRect(530, 460, 120, 30))
        self.lcdNumber_5.setText("9.1")

        self.lcdNumber_6 = QtWidgets.QLabel(self.centralwidget)
        self.lcdNumber_6.setGeometry(QtCore.QRect(650, 460, 120, 30))
        self.lcdNumber_6.setText("0.54")

        MainWindow.setCentralWidget(self.centralwidget)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
