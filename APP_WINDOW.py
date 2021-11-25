from PyQt5 import QtCore, QtGui, QtWidgets


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
