import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap


class ImageSwitcher(QWidget):
    def __init__(self):
        super().__init__()
        self.next_button = None
        self.pixmap = None
        self.image_label = None
        self.layout = None
        self.images = None
        self.image_index = None
        self.initUI()

    def initUI(self):
        self.image_index = 0
        self.images = ['/home/hagikehappy/图片/问题-显示不够宽.png',
                       '/home/hagikehappy/图片/问题-显示不够长.png',
                       '/home/hagikehappy/图片/宪法卫士.png']

        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.pixmap = QPixmap(self.images[self.image_index])
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)
        # self.image_label.setFixedSize(400, 300)  # 设置图片显示大小

        self.next_button = QPushButton('Next Image', self)
        self.next_button.clicked.connect(self.next_image)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.next_button)

        self.setLayout(self.layout)

        self.setWindowTitle('Image Switcher')
        self.setGeometry(100, 100, 450, 350)

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.images)
        self.pixmap = QPixmap(self.images[self.image_index])
        self.image_label.setPixmap(self.pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageSwitcher()
    ex.show()
    sys.exit(app.exec_())
