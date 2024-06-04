# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mainwindow_six.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1269, 1064)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.calibrate_sensor_text = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calibrate_sensor_text.sizePolicy().hasHeightForWidth())
        self.calibrate_sensor_text.setSizePolicy(sizePolicy)
        self.calibrate_sensor_text.setObjectName("calibrate_sensor_text")
        self.gridLayout_3.addWidget(self.calibrate_sensor_text, 5, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.sensor5_min = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor5_min.sizePolicy().hasHeightForWidth())
        self.sensor5_min.setSizePolicy(sizePolicy)
        self.sensor5_min.setObjectName("sensor5_min")
        self.gridLayout_2.addWidget(self.sensor5_min, 9, 3, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 6, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_13.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 6, 0, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(170, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_18.setPalette(palette)
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 8, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 198, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 198, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_7.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 2, 0, 1, 1)
        self.sensor6_max = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor6_max.sizePolicy().hasHeightForWidth())
        self.sensor6_max.setSizePolicy(sizePolicy)
        self.sensor6_max.setObjectName("sensor6_max")
        self.gridLayout_2.addWidget(self.sensor6_max, 10, 3, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_2.addWidget(self.pushButton_3, 13, 3, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 1, 1, 1)
        self.calibrate_sensor = QtWidgets.QPushButton(self.centralwidget)
        self.calibrate_sensor.setObjectName("calibrate_sensor")
        self.gridLayout_2.addWidget(self.calibrate_sensor, 14, 3, 1, 1)
        self.showsensor1 = QtWidgets.QCheckBox(self.centralwidget)
        self.showsensor1.setChecked(True)
        self.showsensor1.setObjectName("showsensor1")
        self.gridLayout_2.addWidget(self.showsensor1, 1, 0, 1, 1)
        self.sensor6_min = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor6_min.sizePolicy().hasHeightForWidth())
        self.sensor6_min.setSizePolicy(sizePolicy)
        self.sensor6_min.setObjectName("sensor6_min")
        self.gridLayout_2.addWidget(self.sensor6_min, 11, 3, 1, 1)
        self.sensor4_max = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor4_max.sizePolicy().hasHeightForWidth())
        self.sensor4_max.setSizePolicy(sizePolicy)
        self.sensor4_max.setObjectName("sensor4_max")
        self.gridLayout_2.addWidget(self.sensor4_max, 6, 3, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_16.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 13, 0, 1, 1)
        self.sensor1_max = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor1_max.sizePolicy().hasHeightForWidth())
        self.sensor1_max.setSizePolicy(sizePolicy)
        self.sensor1_max.setObjectName("sensor1_max")
        self.gridLayout_2.addWidget(self.sensor1_max, 0, 3, 1, 1)
        self.sensor3_max = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor3_max.sizePolicy().hasHeightForWidth())
        self.sensor3_max.setSizePolicy(sizePolicy)
        self.sensor3_max.setObjectName("sensor3_max")
        self.gridLayout_2.addWidget(self.sensor3_max, 4, 3, 1, 1)
        self.sensor1_min = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor1_min.sizePolicy().hasHeightForWidth())
        self.sensor1_min.setSizePolicy(sizePolicy)
        self.sensor1_min.setObjectName("sensor1_min")
        self.gridLayout_2.addWidget(self.sensor1_min, 1, 3, 1, 1)
        self.showforce = QtWidgets.QCheckBox(self.centralwidget)
        self.showforce.setChecked(True)
        self.showforce.setObjectName("showforce")
        self.gridLayout_2.addWidget(self.showforce, 14, 0, 1, 1)
        self.showsensor3 = QtWidgets.QCheckBox(self.centralwidget)
        self.showsensor3.setChecked(True)
        self.showsensor3.setObjectName("showsensor3")
        self.gridLayout_2.addWidget(self.showsensor3, 5, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_19.setPalette(palette)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 10, 0, 1, 1)
        self.sensor2_min = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor2_min.sizePolicy().hasHeightForWidth())
        self.sensor2_min.setSizePolicy(sizePolicy)
        self.sensor2_min.setObjectName("sensor2_min")
        self.gridLayout_2.addWidget(self.sensor2_min, 3, 3, 1, 1)
        self.showsensor5 = QtWidgets.QCheckBox(self.centralwidget)
        self.showsensor5.setChecked(True)
        self.showsensor5.setObjectName("showsensor5")
        self.gridLayout_2.addWidget(self.showsensor5, 9, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setObjectName("label_23")
        self.gridLayout_2.addWidget(self.label_23, 11, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_3.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.sensor3_min = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor3_min.sizePolicy().hasHeightForWidth())
        self.sensor3_min.setSizePolicy(sizePolicy)
        self.sensor3_min.setObjectName("sensor3_min")
        self.gridLayout_2.addWidget(self.sensor3_min, 5, 3, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 5, 1, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setObjectName("label_21")
        self.gridLayout_2.addWidget(self.label_21, 10, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_10.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 4, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 4, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 1, 1, 1)
        self.showsensor2 = QtWidgets.QCheckBox(self.centralwidget)
        self.showsensor2.setChecked(True)
        self.showsensor2.setObjectName("showsensor2")
        self.gridLayout_2.addWidget(self.showsensor2, 3, 0, 1, 1)
        self.sensor2_max = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor2_max.sizePolicy().hasHeightForWidth())
        self.sensor2_max.setSizePolicy(sizePolicy)
        self.sensor2_max.setObjectName("sensor2_max")
        self.gridLayout_2.addWidget(self.sensor2_max, 2, 3, 1, 1)
        self.showsensor6 = QtWidgets.QCheckBox(self.centralwidget)
        self.showsensor6.setChecked(True)
        self.showsensor6.setObjectName("showsensor6")
        self.gridLayout_2.addWidget(self.showsensor6, 11, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 3, 1, 1, 1)
        self.sensor5_max = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor5_max.sizePolicy().hasHeightForWidth())
        self.sensor5_max.setSizePolicy(sizePolicy)
        self.sensor5_max.setObjectName("sensor5_max")
        self.gridLayout_2.addWidget(self.sensor5_max, 8, 3, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 8, 1, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setObjectName("label_22")
        self.gridLayout_2.addWidget(self.label_22, 9, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 2, 1, 1, 1)
        self.showsensor4 = QtWidgets.QCheckBox(self.centralwidget)
        self.showsensor4.setChecked(True)
        self.showsensor4.setObjectName("showsensor4")
        self.gridLayout_2.addWidget(self.showsensor4, 7, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 7, 1, 1, 1)
        self.sensor4_min = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sensor4_min.sizePolicy().hasHeightForWidth())
        self.sensor4_min.setSizePolicy(sizePolicy)
        self.sensor4_min.setObjectName("sensor4_min")
        self.gridLayout_2.addWidget(self.sensor4_min, 7, 3, 1, 1)
        self.track = QtWidgets.QPushButton(self.centralwidget)
        self.track.setObjectName("track")
        self.gridLayout_2.addWidget(self.track, 13, 1, 1, 1)
        self.track_reset = QtWidgets.QPushButton(self.centralwidget)
        self.track_reset.setObjectName("track_reset")
        self.gridLayout_2.addWidget(self.track_reset, 14, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 2, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem, 1, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.savedata = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.savedata.sizePolicy().hasHeightForWidth())
        self.savedata.setSizePolicy(sizePolicy)
        self.savedata.setObjectName("savedata")
        self.gridLayout_4.addWidget(self.savedata, 0, 0, 1, 1)
        self.save_standard = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.save_standard.sizePolicy().hasHeightForWidth())
        self.save_standard.setSizePolicy(sizePolicy)
        self.save_standard.setObjectName("save_standard")
        self.gridLayout_4.addWidget(self.save_standard, 0, 1, 1, 1)
        self.clear_receive = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_receive.sizePolicy().hasHeightForWidth())
        self.clear_receive.setSizePolicy(sizePolicy)
        self.clear_receive.setObjectName("clear_receive")
        self.gridLayout_4.addWidget(self.clear_receive, 1, 1, 1, 1)
        self.save_force = QtWidgets.QPushButton(self.centralwidget)
        self.save_force.setObjectName("save_force")
        self.gridLayout_4.addWidget(self.save_force, 1, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_4, 7, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.com = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.com.sizePolicy().hasHeightForWidth())
        self.com.setSizePolicy(sizePolicy)
        self.com.setObjectName("com")
        self.gridLayout.addWidget(self.com, 0, 1, 1, 1)
        self.scancom = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scancom.sizePolicy().hasHeightForWidth())
        self.scancom.setSizePolicy(sizePolicy)
        self.scancom.setObjectName("scancom")
        self.gridLayout.addWidget(self.scancom, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.bps = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bps.sizePolicy().hasHeightForWidth())
        self.bps.setSizePolicy(sizePolicy)
        self.bps.setObjectName("bps")
        self.bps.addItem("")
        self.bps.addItem("")
        self.gridLayout.addWidget(self.bps, 2, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 3, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 3, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem1, 6, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_3.addItem(spacerItem2, 3, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.plotM = PlotWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotM.sizePolicy().hasHeightForWidth())
        self.plotM.setSizePolicy(sizePolicy)
        self.plotM.setMinimumSize(QtCore.QSize(0, 0))
        self.plotM.setObjectName("plotM")
        self.verticalLayout.addWidget(self.plotM)
        self.plot_force = PlotWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_force.sizePolicy().hasHeightForWidth())
        self.plot_force.setSizePolicy(sizePolicy)
        self.plot_force.setMinimumSize(QtCore.QSize(0, 200))
        self.plot_force.setMaximumSize(QtCore.QSize(16777215, 200))
        self.plot_force.setObjectName("plot_force")
        self.plot_force.setYRange(0, 2000)
        self.verticalLayout.addWidget(self.plot_force)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout_7.setHorizontalSpacing(6)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_6.setHorizontalSpacing(6)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_15.setFont(font)
        self.label_15.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_15.setObjectName("label_15")
        self.gridLayout_6.addWidget(self.label_15, 0, 0, 1, 1)
        self.receive_data = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.receive_data.sizePolicy().hasHeightForWidth())
        self.receive_data.setSizePolicy(sizePolicy)
        self.receive_data.setObjectName("receive_data")
        self.gridLayout_6.addWidget(self.receive_data, 1, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_6, 3, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.control_force = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.control_force.sizePolicy().hasHeightForWidth())
        self.control_force.setSizePolicy(sizePolicy)
        self.control_force.setObjectName("control_force")
        self.gridLayout_5.addWidget(self.control_force, 2, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.gridLayout_5.addWidget(self.label_17, 0, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.gridLayout_5.addWidget(self.label_24, 0, 1, 1, 1)
        self.standard_data = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.standard_data.sizePolicy().hasHeightForWidth())
        self.standard_data.setSizePolicy(sizePolicy)
        self.standard_data.setObjectName("standard_data")
        self.gridLayout_5.addWidget(self.standard_data, 2, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_5, 3, 1, 1, 1)
        self.gridLayout_7.setColumnStretch(0, 1)
        self.gridLayout_7.setColumnStretch(1, 2)
        self.verticalLayout_2.addLayout(self.gridLayout_7)
        self.verticalLayout_2.setStretch(0, 6)
        self.verticalLayout_2.setStretch(1, 1)
        self.gridLayout_8.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.track_force = PlotWidget(self.centralwidget)
        self.track_force.setObjectName("track_force")
        self.gridLayout_8.addWidget(self.track_force, 1, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1269, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_12.setText(_translate("MainWindow", "Max"))
        self.label_13.setText(_translate("MainWindow", "Sensor4"))
        self.label_18.setText(_translate("MainWindow", "Sensor5"))
        self.label_7.setText(_translate("MainWindow", "Sensor2"))
        self.pushButton_3.setText(_translate("MainWindow", "Normalize"))
        self.label_5.setText(_translate("MainWindow", "Max"))
        self.calibrate_sensor.setText(_translate("MainWindow", "Calibrate"))
        self.showsensor1.setText(_translate("MainWindow", "Show"))
        self.label_16.setText(_translate("MainWindow", "Force"))
        self.showforce.setText(_translate("MainWindow", "Show"))
        self.showsensor3.setText(_translate("MainWindow", "Show"))
        self.label_19.setText(_translate("MainWindow", "Sensor6"))
        self.showsensor5.setText(_translate("MainWindow", "Show"))
        self.label_23.setText(_translate("MainWindow", "Min"))
        self.label_3.setText(_translate("MainWindow", "Sensor1"))
        self.label_11.setText(_translate("MainWindow", "Min"))
        self.label_21.setText(_translate("MainWindow", "Max"))
        self.label_10.setText(_translate("MainWindow", "Sensor3"))
        self.label_9.setText(_translate("MainWindow", "Max"))
        self.label_4.setText(_translate("MainWindow", "Min"))
        self.showsensor2.setText(_translate("MainWindow", "Show"))
        self.showsensor6.setText(_translate("MainWindow", "Show"))
        self.label_8.setText(_translate("MainWindow", "Min"))
        self.label_20.setText(_translate("MainWindow", "Max"))
        self.label_22.setText(_translate("MainWindow", "Min"))
        self.label_6.setText(_translate("MainWindow", "Max"))
        self.showsensor4.setText(_translate("MainWindow", "Show"))
        self.label_14.setText(_translate("MainWindow", "Min"))
        self.track.setText(_translate("MainWindow", "Track"))
        self.track_reset.setText(_translate("MainWindow", "Track reset"))
        self.savedata.setText(_translate("MainWindow", "Save raw"))
        self.save_standard.setText(_translate("MainWindow", "Save normalization"))
        self.clear_receive.setText(_translate("MainWindow", "Clear receiving area"))
        self.save_force.setText(_translate("MainWindow", "Save force"))
        self.label.setText(_translate("MainWindow", "COM"))
        self.scancom.setText(_translate("MainWindow", "Scan"))
        self.label_2.setText(_translate("MainWindow", "BaudRate"))
        self.bps.setItemText(0, _translate("MainWindow", "115200"))
        self.bps.setItemText(1, _translate("MainWindow", "256000"))
        self.pushButton.setText(_translate("MainWindow", "Open"))
        self.pushButton_2.setText(_translate("MainWindow", "Close"))
        self.label_15.setText(_translate("MainWindow", "Raw data"))
        self.label_17.setText(_translate("MainWindow", "Normalized data"))
        self.label_24.setText(_translate("MainWindow", "Control force"))
from pyqtgraph import PlotWidget