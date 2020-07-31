import sys
import predict
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from tkinter import *
from PIL import ImageTk, Image


def Init():
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.brush_size = 1
        self.brush_color = Qt.white
        self.last_point = QPoint()
        self.initUI()

    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        # Create 'Clear' button
        btn1 = QPushButton('Clear', self)
        btn1.move(300, 330)
        btn1.resize(btn1.sizeHint())
        btn1.clicked.connect(self.clear)

        # Create 'Predict' button
        btn2 = QPushButton('Predict', self)
        btn2.move(300, 365)
        btn2.resize(btn2.sizeHint())
        btn2.clicked.connect(self.predict)

        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save)

        load_action = QAction('Load', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load)

        filemenu.addAction(save_action)
        filemenu.addAction(load_action)

        self.setWindowTitle('HDR Program')
        self.setGeometry(300, 300, 400, 400)
        self.show()

    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

    def save(self):
        fpath, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        if fpath:
            self.image.save(fpath)

    def load(self):
        root = Tk()
        root.title("Load Image")
        root.resizable(400,400)
        filename, _ = QFileDialog.getOpenFileName(self, 'Load Image', '')
        if filename:
            load_image = PhotoImage(file=filename)
            label = Label(root, image=load_image)
            label.pack()
            root.mainloop()

    def clear(self):
        self.image.fill(Qt.black)
        self.update()

    def predict(self):
        self.image.save("draw.png")
        predict.Predict()