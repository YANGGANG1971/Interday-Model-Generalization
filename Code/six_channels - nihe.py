import sys
import time
import numpy as np
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from Mainwindow_six import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QDialog
import serial
import serial.tools.list_ports
import pyqtgraph as pg
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import joblib


# 定义一个线程类
class New_Thread(QThread):
    # 自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(int)
    min = pyqtSignal(int)
    min_check = pyqtSignal(int)
    finish = pyqtSignal(int)
    # 带一个参数t
    def __init__(self):
        super(New_Thread, self).__init__()
    # run函数是子线程中的操作，线程启动后开始执行
    def run(self):
        TIM1 = [5, 4, 3, 2, 1]
        for i in TIM1:
            time.sleep(1)
            # 发射自定义信号
            # 通过emit函数将参数i传递给主线程，触发自定义信号
            self.finishSignal.emit(i)  # 注意这里与_signal = pyqtSignal(str)中的类型相同
        # min
        for i in [60, 59, 58, 57]:
            time.sleep(1)
            self.min.emit(i)
        self.min_check.emit(1)
        for i in range(56, 5, -1):
            time.sleep(1)
            self.min.emit(i)
        self.min_check.emit(0)
        for i in [5, 4, 3, 2, 1]:
            time.sleep(1)
            self.min.emit(i)
        time.sleep(1)
        self.finish.emit(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.mSerial = serial.Serial()
        self.thread = New_Thread()  # 实例化一个线程
        self.mTimer2 = QTimer()  # 定时器接收数据
        self.setWindowTitle("串口数据波形显示工具")
        self.PosX = 0
        self.dataIndex = 0  # 数据列表当前索引
        self.i_min = 0
        self.i_mid = 0
        self.i_max = 0
        self.dataMaxLength = 300  # 数据列表最大长度
        self.trackMaxLength = 3300  # 力追踪列表最大长度
        self.dataheader = b''  # 数据帧开头

        self.min_1 = 0
        self.max_1 = 4096
        self.min_2 = 0
        self.max_2 = 4096
        self.min_3 = 0
        self.max_3 = 4096
        self.min_4 = 0
        self.max_4 = 4096
        self.min_5 = 0
        self.max_5 = 4096
        self.min_6 = 0
        self.max_6 = 4096
        self.recvdata = ''  # 接收内容
        self.standard_check = 0
        self.calibrate_check = 1
        self.track_check = 0
        self.control_check = 0
        self.fit_check = 0
        self.min_check = 0
        self.calibrate_flag = 0
        self.force = 0  # 目标力
        self.hand_force = 0  # 人手力
        self.prosthetic_force = 0  # 假手力
        self.sensor = np.zeros(6, dtype=float)  # 光波导信号
        # self.x_min = 0.0  # 三次抓握采集的数值
        # self.x_mid = 0.0
        # self.x_max = 0.0
        # self.y_min = 0.0
        # self.y_mid = 0.0
        # self.y_max = 0.0
        self.A = 0.0  # 方程系数
        self.B = 0.0
        self.C = 0.0

        self.sensor1 = np.zeros(shape=(1, 0))
        self.sensor2 = np.zeros(shape=(1, 0))
        self.sensor3 = np.zeros(shape=(1, 0))
        self.sensor4 = np.zeros(shape=(1, 0))
        self.sensor5 = np.zeros(shape=(1, 0))
        self.sensor6 = np.zeros(shape=(1, 0))
        self.grip_force = np.zeros(shape=(1, 0))

        self.dataX = np.zeros(self.dataMaxLength, dtype=int)
        self.dataY = np.zeros(self.dataMaxLength, dtype=int)
        self.dataZ = np.zeros(self.dataMaxLength, dtype=int)
        self.dataH = np.zeros(self.dataMaxLength, dtype=int)
        self.dataI = np.zeros(self.dataMaxLength, dtype=int)
        self.dataJ = np.zeros(self.dataMaxLength, dtype=int)
        # 力传感器数据
        self.dataK = np.zeros(self.dataMaxLength, dtype=int)
        self.dataL = np.zeros(self.dataMaxLength, dtype=int)
        # 力跟随数据
        self.dataM = np.zeros(self.trackMaxLength, dtype=int)
        self.dataN = np.zeros(1, dtype=int)

        # self.dataM[0:self.trackMaxLength // 7] = 100
        # self.dataM[self.trackMaxLength // 7:2 * self.trackMaxLength // 7] = 500
        # self.dataM[2 * self.trackMaxLength // 7:3 * self.trackMaxLength // 7] = 1000
        # self.dataM[3 * self.trackMaxLength // 7:4 * self.trackMaxLength // 7] = 1500
        # self.dataM[4 * self.trackMaxLength // 7:5 * self.trackMaxLength // 7] = 1000
        # self.dataM[5 * self.trackMaxLength // 7:6 * self.trackMaxLength // 7] = 500
        # self.dataM[6 * self.trackMaxLength // 7:self.trackMaxLength] = 100

        self.dataM[0:self.trackMaxLength // 11] = 100
        self.dataM[self.trackMaxLength // 11:2 * self.trackMaxLength // 11] = 400
        self.dataM[2 * self.trackMaxLength // 11:3 * self.trackMaxLength // 11] = 800
        self.dataM[3 * self.trackMaxLength // 11:4 * self.trackMaxLength // 11] = 1200
        self.dataM[4 * self.trackMaxLength // 11:5 * self.trackMaxLength // 11] = 1600
        self.dataM[5 * self.trackMaxLength // 11:6 * self.trackMaxLength // 11] = 2000
        self.dataM[6 * self.trackMaxLength // 11:7 * self.trackMaxLength // 11] = 1600
        self.dataM[7 * self.trackMaxLength // 11:8 * self.trackMaxLength // 11] = 1200
        self.dataM[8 * self.trackMaxLength // 11:9 * self.trackMaxLength // 11] = 800
        self.dataM[9 * self.trackMaxLength // 11:10 * self.trackMaxLength // 11] = 400
        self.dataM[10 * self.trackMaxLength // 11:self.trackMaxLength] = 100

        self.ui.sensor1_min.textChanged.connect(self.sensor1_min_change)
        self.ui.sensor1_max.textChanged.connect(self.sensor1_max_change)
        self.ui.sensor2_min.textChanged.connect(self.sensor2_min_change)
        self.ui.sensor2_max.textChanged.connect(self.sensor2_max_change)
        self.ui.sensor3_min.textChanged.connect(self.sensor3_min_change)
        self.ui.sensor3_max.textChanged.connect(self.sensor3_max_change)
        self.ui.sensor4_min.textChanged.connect(self.sensor4_min_change)
        self.ui.sensor4_max.textChanged.connect(self.sensor4_max_change)
        self.ui.sensor5_min.textChanged.connect(self.sensor5_min_change)
        self.ui.sensor5_max.textChanged.connect(self.sensor5_max_change)
        self.ui.sensor6_min.textChanged.connect(self.sensor6_min_change)
        self.ui.sensor6_max.textChanged.connect(self.sensor6_max_change)
        self.ui.scancom.clicked.connect(self.ScanComPort)
        self.ui.pushButton.clicked.connect(self.OpenComPort)
        self.ui.pushButton_2.clicked.connect(self.CloseComPort)
        self.ui.pushButton_3.clicked.connect(self.standard_data)
        self.ui.pushButton_3.setCheckable(False)
        self.ui.clear_receive.clicked.connect(self.clear_receive)
        self.ui.savedata.clicked.connect(self.savedata)
        self.ui.save_standard.clicked.connect(self.save_standard)
        self.ui.save_force.clicked.connect(self.save_force)
        self.ui.calibrate_sensor.clicked.connect(self.calibrate_sensor)
        self.ui.track.clicked.connect(self.track)
        self.ui.track_reset.clicked.connect(self.track_reset)

        # self.ui.plotM.setBackground("white")  # 改变示波器背景颜色
        self.canvasX = self.ui.plotM.plot(self.dataX, pen=pg.mkPen(color='r', width=1))
        self.canvasY = self.ui.plotM.plot(self.dataY, pen=pg.mkPen(color='g', width=1))
        self.canvasZ = self.ui.plotM.plot(self.dataZ, pen=pg.mkPen(color='b', width=1))
        self.canvasH = self.ui.plotM.plot(self.dataH, pen=pg.mkPen(color='y', width=1))
        self.canvasI = self.ui.plotM.plot(self.dataI, pen=pg.mkPen(color='purple', width=1))
        self.canvasJ = self.ui.plotM.plot(self.dataJ, pen=pg.mkPen(color='peru', width=1))
        self.canvasK = self.ui.plot_force.plot(self.dataK, pen=pg.mkPen(color='w', width=1))
        self.canvasL = self.ui.plot_force.plot(self.dataL, pen=pg.mkPen(color='r', width=1))
        # 力跟随
        self.canvasM = self.ui.track_force.plot(self.dataM, pen=pg.mkPen(color='w', width=1))
        self.canvasN = self.ui.track_force.plot(self.dataN, pen=pg.mkPen(color='r', width=1))

        self.fit1 = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', alpha=0.01,
                                 max_iter=1000, batch_size=64)
        self.MinMax = preprocessing.MinMaxScaler()

    # 接受通过emit传来的信息，执行相应操作
    def Change(self, msg):
        self.ui.calibrate_sensor_text.append(f"{msg}s......")

    def min(self, msg):
        if self.min_check == 1:
            self.ui.calibrate_sensor_text.append("Please start gripping the dynamometer")
        self.ui.calibrate_sensor_text.append(f"{msg}s......")
        self.min_check = 0

    def check_min(self, msg):
        self.calibrate_flag = msg

    def calibra_finish(self):
        self.ui.calibrate_sensor_text.append(f"Sensor calibration is complete, click the button again to start controlling the prosthetic hand")

    def calibrate_sensor(self):
        if self.control_check == 0:
            self.ui.calibrate_sensor.setText("control")
            self.control_check = 1
            if self.calibrate_check == 1:
                # 将线程thread的信号finishSignal和UI主线程中的槽函数Change进行连接
                self.thread.finishSignal.connect(self.Change)
                self.thread.min.connect(self.min)
                self.thread.min_check.connect(self.check_min)
                self.thread.finish.connect(self.calibra_finish)
                # 启动线程，执行线程类中run函数
                self.thread.start()
                self.ui.calibrate_sensor_text.append("Calibration starts after 5 seconds")

                self.min_check = 1

                self.calibrate_check = 0
        else:
            # print('End')
            self.ui.calibrate_sensor.setText("Calibration Sensor")
            self.thread.terminate()  # 终止线程
            self.calibrate_check = 1
            self.control_check = 0
            self.fit_model()

    def savedata(self):
        dlg = QFileDialog()
        filenames = dlg.getSaveFileName(None, "保存日志文件", None, "Txt files(*.txt)")

        try:
            with open(file=filenames[0], mode='w', encoding='utf-8') as file:
                file.write(self.ui.receive_data.toPlainText())
        except:
            QMessageBox.critical(self, '日志异常', '保存日志文件失败！')

    def save_standard(self):
        dlg = QFileDialog()
        filenames = dlg.getSaveFileName(None, "保存日志文件", None, "Txt files(*.txt)")

        try:
            with open(file=filenames[0], mode='w', encoding='utf-8') as file:
                file.write(self.ui.standard_data.toPlainText())
        except:
            QMessageBox.critical(self, '日志异常', '保存日志文件失败！')

    def save_force(self):
        dlg = QFileDialog()
        filenames = dlg.getSaveFileName(None, "保存日志文件", None, "Txt files(*.txt)")

        try:
            with open(file=filenames[0], mode='w', encoding='utf-8') as file:
                file.write(self.ui.control_force.toPlainText())
        except:
            QMessageBox.critical(self, '日志异常', '保存日志文件失败！')

    def standard_data(self):
        if not self.ui.pushButton_3.isChecked():
            self.standard_check = 1
            self.ui.pushButton_3.setCheckable(True)
            self.ui.pushButton_3.setText("正常显示")
            self.dataX = [0] * self.dataMaxLength
            self.dataY = [0] * self.dataMaxLength
            self.dataZ = [0] * self.dataMaxLength
            self.dataH = [0] * self.dataMaxLength
            self.dataI = [0] * self.dataMaxLength
            self.dataJ = [0] * self.dataMaxLength
            self.dataK = [0] * self.dataMaxLength
            self.dataL = [0] * self.dataMaxLength
            self.dataIndex = 0
        if self.ui.pushButton_3.isChecked():
            self.standard_check = 0
            self.ui.pushButton_3.setCheckable(False)
            self.ui.pushButton_3.setText("归一化")

    def clear_receive(self):
        self.ui.receive_data.clear()
        self.ui.standard_data.clear()
        self.dataIndex = 0
        self.i_min = 0
        self.dataX = [0] * self.dataMaxLength
        self.dataY = [0] * self.dataMaxLength
        self.dataZ = [0] * self.dataMaxLength
        self.dataH = [0] * self.dataMaxLength
        self.dataI = [0] * self.dataMaxLength
        self.dataJ = [0] * self.dataMaxLength
        self.dataK = [0] * self.dataMaxLength
        self.dataL = [0] * self.dataMaxLength
        self.dataN = [0] * self.trackMaxLength
        if not self.mSerial.isOpen():
            self.canvasX.setData([0] * self.dataMaxLength)
            self.canvasY.setData([0] * self.dataMaxLength)
            self.canvasZ.setData([0] * self.dataMaxLength)
            self.canvasH.setData([0] * self.dataMaxLength)
            self.canvasI.setData([0] * self.dataMaxLength)
            self.canvasJ.setData([0] * self.dataMaxLength)
            self.canvasK.setData([0] * self.dataMaxLength)
            self.canvasL.setData([0] * self.dataMaxLength)
            self.canvasN.setData([0] * self.trackMaxLength)

    def fit_model(self):
        self.min_1 = min(self.sensor1)
        self.max_1 = max(self.sensor1)
        self.min_2 = min(self.sensor2)
        self.max_2 = max(self.sensor2)
        self.min_3 = min(self.sensor3)
        self.max_3 = max(self.sensor3)
        self.min_4 = min(self.sensor4)
        self.max_4 = max(self.sensor4)
        self.min_5 = min(self.sensor5)
        self.max_5 = max(self.sensor5)
        self.min_6 = min(self.sensor6)
        self.max_6 = max(self.sensor6)
        x_data = np.hstack((self.sensor1.reshape(-1, 1), self.sensor2.reshape(-1, 1), self.sensor3.reshape(-1, 1),
                            self.sensor4.reshape(-1, 1), self.sensor5.reshape(-1, 1), self.sensor6.reshape(-1, 1)))
        y_data = self.grip_force.reshape(-1, 1)
        x_data_std = self.MinMax.fit_transform(x_data)
        y_data_std = self.MinMax.fit_transform(y_data)
        # print("x_data : ", x_data)
        # print("x_data's shape : ", x_data.shape)
        # print("y_data : ", y_data)
        # print("y_data's shape : ", y_data.shape)
        # print("x_data_std : ", x_data_std)
        # print("x_data_std's shape : ", x_data_std.shape)
        # print("y_data_std : ", y_data_std)
        # print("y_data_std's shape : ", y_data_std.shape)
        self.fit1.fit(x_data_std, y_data_std.ravel())
        self.fit_check = 1
        self.mTimer2.timeout.connect(self.control_force_send)
        self.mTimer2.start(20)  # 打开串口接收定时器，周期为20ms


    def sensor1_min_change(self):
        if self.ui.sensor1_min.text():
            self.min_1 = int(self.ui.sensor1_min.text())
        else:
            self.min_1 = 0

    def sensor1_max_change(self):
        if self.ui.sensor1_max.text():
            self.max_1 = int(self.ui.sensor1_max.text())
        else:
            self.max_1 = 4096

    def sensor2_min_change(self):
        if self.ui.sensor2_min.text():
            self.min_2 = int(self.ui.sensor2_min.text())
        else:
            self.min_2 = 0

    def sensor2_max_change(self):
        if self.ui.sensor2_max.text():
            self.max_2 = int(self.ui.sensor2_max.text())
        else:
            self.max_2 = 4096

    def sensor3_min_change(self):
        if self.ui.sensor3_min.text():
            self.min_3 = int(self.ui.sensor3_min.text())
        else:
            self.min_3 = 0

    def sensor3_max_change(self):
        if self.ui.sensor3_max.text():
            self.max_3 = int(self.ui.sensor3_max.text())
        else:
            self.max_3 = 4096

    def sensor4_min_change(self):
        if self.ui.sensor4_min.text():
            self.min_4 = int(self.ui.sensor4_min.text())
        else:
            self.min_4 = 0

    def sensor4_max_change(self):
        if self.ui.sensor4_max.text():
            self.max_4 = int(self.ui.sensor4_max.text())
        else:
            self.max_4 = 4096

    def sensor5_min_change(self):
        if self.ui.sensor4_min.text():
            self.min_5 = int(self.ui.sensor5_min.text())
        else:
            self.min_5 = 0

    def sensor5_max_change(self):
        if self.ui.sensor4_max.text():
            self.max_5 = int(self.ui.sensor5_max.text())
        else:
            self.max_5 = 4096

    def sensor6_min_change(self):
        if self.ui.sensor4_min.text():
            self.min_6 = int(self.ui.sensor6_min.text())
        else:
            self.min_6 = 0

    def sensor6_max_change(self):
        if self.ui.sensor4_max.text():
            self.max_6 = int(self.ui.sensor6_max.text())
        else:
            self.max_6 = 4096

    def ScanComPort(self):
        self.ui.com.clear()
        self.portDict = {}
        portlist = list(serial.tools.list_ports.comports())
        for port in portlist:
            self.portDict["%s" % port[0]] = "%s" % port[1]
            self.ui.com.addItem(port[0])
        if len(self.portDict) == 0:
            QMessageBox.critical(self, "警告", "未找到串口设备！", QMessageBox.Cancel, QMessageBox.Cancel)
        pass

    def OpenComPort(self):
        self.mSerial.port = self.ui.com.currentText()
        self.mSerial.baudrate = int(self.ui.bps.currentText())
        self.mSerial.bytesize = 8  # 设置数据位
        self.mSerial.stopbits = 1  # 设置停止位
        self.mSerial.parity = "N"  # 设置校验位
        if self.mSerial.isOpen():
            QMessageBox.warning(self, "提示", "串口已打开", QMessageBox.Cancel, QMessageBox.Cancel)
        else:
            try:
                self.ui.pushButton.setEnabled(False)
                self.mSerial.open()
            except:
                QMessageBox.critical(self, "警告", "串口打开失败！", QMessageBox.Cancel, QMessageBox.Cancel)
                self.ui.pushButton.setEnabled(True)
        self.mTimer1 = QTimer()  # 定时器接收数据
        self.mTimer1.timeout.connect(self.ReceiverPortData)
        self.mTimer1.start(8)  # 打开串口接收定时器，周期为1ms

        pass

    def CloseComPort(self):
        self.mTimer1.stop()
        if self.mTimer2.isActive():
            self.mTimer2.stop()
        if self.mSerial.isOpen:
            self.ui.pushButton.setEnabled(True)
            self.mSerial.close()
        pass

    def control_force_send(self):
        sensor = self.sensor.reshape(1, -1)
        self.force = self.fit1.predict(sensor)
        # load model
        # fit2 = joblib.load('D:/pythonproject/GPR/model_0817_2.pkl')
        # self.force = fit2.predict(sensor)
        # 反归一化
        self.force = self.MinMax.inverse_transform(self.force.reshape(1, -1))
        # print("sensor : ", sensor)
        # print("sensor's shape : ", sensor.shape)
        # print("self.force : ", self.force)
        # print("self.force's shape : ", self.force.shape)
        if self.force < 0:
            mw_value = 50  # 因为假手力传感器没归零
        else:
            mw_value = int(self.force[0, 0])
        # hexStr = 'AABB' + '{:04X}'.format(int(self.force[0, 0]))

        mw_set = [0xAA, 0xBB, 0x00, 0x00]  # 初始设置值为0W
        # 数据结构：头1 头2  地址 命令1  数高  数低  尾1  尾2
        if (mw_value >= 0) and (mw_value <= 32767):
            y_h = mw_value >> 8  # 输入功率整数值转换为16进制，整形数>>8将得到高位十六进制数字节（16进制数<<8将得到高位整形数字节）
            y_l = mw_value & 0xff  # 十六进制低位字节
            mw_set[2] = y_h  # 功率值高字节替换
            mw_set[3] = y_l  # 功率值低字节替换
        if self.mSerial.inWaiting():
            self.mSerial.write(mw_set)  # 输出功率设定值
            print(mw_set)
        # self.mSerial.write(bytes_hex)
        # bytes_hex = bytes.fromhex(hexStr)
        # print("bytes_hex's length : ", len(bytes_hex))
        # if len(bytes_hex) == 4:
        #     if self.mSerial.inWaiting():
        #         self.mSerial.write(bytes_hex)
        #         print(bytes_hex)
        # str_1 = str(self.sensor[0]) + ',' + str(self.sensor[1]) + ',' + str(self.sensor[2]) + ',' + str(self.sensor[3]) + ',' + str(self.sensor[4]) + ',' + str(self.sensor[5]) + ',' + str(self.force)
        # self.ui.control_force.append(str_1.replace('\r\n', ''))
        str_1 = str(float(sensor[0, 0])) + ',' + str(float(sensor[0, 1])) + ',' + str(float(sensor[0, 2])) + ',' + \
                str(float(sensor[0, 3])) + ',' + str(float(sensor[0, 4])) + ',' + str(float(sensor[0, 5])) + ',' + \
                str(mw_value) + ',' + str(self.hand_force) + ',' + str(self.prosthetic_force)
        self.ui.standard_data.append(self.recvdata.decode('utf-8').replace('\r\n', ''))
        self.ui.control_force.append(str_1.replace('\r\n', ''))
        if self.track_check:
            self.dataN = np.append(self.dataN, self.prosthetic_force)
            self.canvasN.setData(self.dataN)

    def track(self):
        self.track_check = 1

    def track_reset(self):
        self.track_check = 0
        self.dataN = np.zeros(1, dtype=int)
        self.canvasN.setData(self.dataN)

    def ReceiverPortData(self):
        '''
        接收串口数据，并解析出每一个数据项更新到波形图
        数据帧格式'$$:95.68,195.04,-184.0\r\n'
        每个数据帧以b'$$:'开头，每个数据项以','分割
        '''
        try:
            n = self.mSerial.inWaiting()
        except:
            self.CloseComPort()
        if n:
            # 端口缓存内有数据
            try:
                self.recvdata = self.mSerial.readline()  # 读取一行数据
                self.ui.receive_data.append(self.recvdata.decode('utf-8').replace('\r\n', ''))
                if self.recvdata.decode('UTF-8').startswith(self.dataheader.decode('UTF-8')):
                    rawdata = self.recvdata[len(self.dataheader): len(self.recvdata) - 2]
                    data = rawdata.split(b',')
                    self.hand_force = int(data[6])
                    self.prosthetic_force = int(data[7])
                    if self.fit_check:
                        self.sensor[0] = round((np.clip(int(data[0]), self.min_1, self.max_1) - self.min_1) /
                                               (self.max_1 - self.min_1), 4)
                        self.sensor[1] = round((np.clip(int(data[1]), self.min_2, self.max_2) - self.min_2) /
                                               (self.max_2 - self.min_2), 4)
                        self.sensor[2] = round((np.clip(int(data[2]), self.min_3, self.max_3) - self.min_3) /
                                               (self.max_3 - self.min_3), 4)
                        self.sensor[3] = round((np.clip(int(data[3]), self.min_4, self.max_4) - self.min_4) /
                                               (self.max_4 - self.min_4), 4)
                        self.sensor[4] = round((np.clip(int(data[4]), self.min_5, self.max_5) - self.min_5) /
                                               (self.max_5 - self.min_5), 4)
                        self.sensor[5] = round((np.clip(int(data[5]), self.min_6, self.max_6) - self.min_6) /
                                               (self.max_6 - self.min_6), 4)
                        # print("self.sensor : ", self.sensor)
                    if self.calibrate_flag == 1:
                        self.sensor1 = np.append(self.sensor1, int(data[0]))
                        self.sensor2 = np.append(self.sensor2, int(data[1]))
                        self.sensor3 = np.append(self.sensor3, int(data[2]))
                        self.sensor4 = np.append(self.sensor4, int(data[3]))
                        self.sensor5 = np.append(self.sensor5, int(data[4]))
                        self.sensor6 = np.append(self.sensor6, int(data[5]))
                        self.grip_force = np.append(self.grip_force, int(data[6]))
                    if self.standard_check == 0:
                        if self.dataIndex < self.dataMaxLength:
                            # 接收到的数据长度小于最大数据缓存长度，直接按索引赋值，索引自增1
                            self.dataX[self.dataIndex] = np.clip(int(data[0]), self.min_1, self.max_1)
                            self.dataY[self.dataIndex] = np.clip(int(data[1]), self.min_2, self.max_2)
                            self.dataZ[self.dataIndex] = np.clip(int(data[2]), self.min_3, self.max_3)
                            self.dataH[self.dataIndex] = np.clip(int(data[3]), self.min_4, self.max_4)
                            self.dataI[self.dataIndex] = np.clip(int(data[4]), self.min_5, self.max_5)
                            self.dataJ[self.dataIndex] = np.clip(int(data[5]), self.min_6, self.max_6)
                            self.dataK[self.dataIndex] = int(data[6])
                            self.dataL[self.dataIndex] = int(data[7])
                            self.dataIndex = self.dataIndex + 1
                        else:
                            # 寄收到的数据长度大于或等于最大数据缓存长度，丢弃最前一个数据新数据添加到数据列尾
                            self.dataX[:-1] = self.dataX[1:]
                            self.dataY[:-1] = self.dataY[1:]
                            self.dataZ[:-1] = self.dataZ[1:]
                            self.dataH[:-1] = self.dataH[1:]
                            self.dataI[:-1] = self.dataI[1:]
                            self.dataJ[:-1] = self.dataJ[1:]
                            self.dataK[:-1] = self.dataK[1:]
                            self.dataL[:-1] = self.dataL[1:]
                            self.dataX[self.dataIndex - 1] = np.clip(int(data[0]), self.min_1, self.max_1)
                            self.dataY[self.dataIndex - 1] = np.clip(int(data[1]), self.min_2, self.max_2)
                            self.dataZ[self.dataIndex - 1] = np.clip(int(data[2]), self.min_3, self.max_3)
                            self.dataH[self.dataIndex - 1] = np.clip(int(data[3]), self.min_4, self.max_4)
                            self.dataI[self.dataIndex - 1] = np.clip(int(data[4]), self.min_5, self.max_5)
                            self.dataJ[self.dataIndex - 1] = np.clip(int(data[5]), self.min_6, self.max_6)
                            self.dataK[self.dataIndex - 1] = int(data[6])
                            self.dataL[self.dataIndex - 1] = int(data[7])
                    if self.standard_check == 1:  # 如果按下归一化按钮
                        if self.dataIndex < self.dataMaxLength:
                            # 接收到的数据长度小于最大数据缓存长度，直接按索引赋值，索引自增1
                            self.dataX[self.dataIndex] = round((np.clip(int(data[0]), self.min_1, self.max_1)
                                                                - self.min_1) / (self.max_1 - self.min_1), 4)
                            self.dataY[self.dataIndex] = round((np.clip(int(data[1]), self.min_2, self.max_2)
                                                                - self.min_2) / (self.max_2 - self.min_2), 4)
                            self.dataZ[self.dataIndex] = round((np.clip(int(data[2]), self.min_3, self.max_3)
                                                                - self.min_3) / (self.max_3 - self.min_3), 4)
                            self.dataH[self.dataIndex] = round((np.clip(int(data[3]), self.min_4, self.max_4)
                                                                - self.min_4) / (self.max_4 - self.min_4), 4)
                            self.dataI[self.dataIndex] = round((np.clip(int(data[4]), self.min_5, self.max_5)
                                                                - self.min_5) / (self.max_5 - self.min_5), 4)
                            self.dataJ[self.dataIndex] = round((np.clip(int(data[5]), self.min_6, self.max_6)
                                                                - self.min_6) / (self.max_6 - self.min_6), 4)
                            self.dataK[self.dataIndex] = int(data[6])
                            self.dataL[self.dataIndex] = int(data[7])
                            self.dataIndex = self.dataIndex + 1
                        else:
                            # 寄收到的数据长度大于或等于最大数据缓存长度，丢弃最前一个数据新数据添加到数据列尾
                            self.dataX[:-1] = self.dataX[1:]
                            self.dataY[:-1] = self.dataY[1:]
                            self.dataZ[:-1] = self.dataZ[1:]
                            self.dataH[:-1] = self.dataH[1:]
                            self.dataI[:-1] = self.dataI[1:]
                            self.dataJ[:-1] = self.dataJ[1:]
                            self.dataK[:-1] = self.dataK[1:]
                            self.dataL[:-1] = self.dataL[1:]
                            self.dataX[self.dataIndex - 1] = round((np.clip(int(data[0]), self.min_1, self.max_1)
                                                                    - self.min_1) / (self.max_1 - self.min_1), 4)
                            self.dataY[self.dataIndex - 1] = round((np.clip(int(data[1]), self.min_2, self.max_2)
                                                                    - self.min_2) / (self.max_2 - self.min_2), 4)
                            self.dataZ[self.dataIndex - 1] = round((np.clip(int(data[2]), self.min_3, self.max_3)
                                                                    - self.min_3) / (self.max_3 - self.min_3), 4)
                            self.dataH[self.dataIndex - 1] = round((np.clip(int(data[3]), self.min_4, self.max_4)
                                                                    - self.min_4) / (self.max_4 - self.min_4), 4)
                            self.dataI[self.dataIndex - 1] = round((np.clip(int(data[4]), self.min_5, self.max_5)
                                                                    - self.min_5) / (self.max_5 - self.min_5), 4)
                            self.dataJ[self.dataIndex - 1] = round((np.clip(int(data[5]), self.min_6, self.max_6)
                                                                    - self.min_6) / (self.max_6 - self.min_6), 4)
                            self.dataK[self.dataIndex - 1] = int(data[6])
                            self.dataL[self.dataIndex - 1] = int(data[7])
                        # 串口打印
                        if self.dataIndex < self.dataMaxLength:
                            str_0 = str(self.dataX[self.dataIndex - 1]) + ',' + str(
                                self.dataY[self.dataIndex - 1]) + ',' + \
                                    str(self.dataZ[self.dataIndex - 1]) + ',' + str(
                                self.dataH[self.dataIndex - 1]) + ',' + \
                                    str(self.dataI[self.dataIndex - 1]) + ',' + str(
                                self.dataJ[self.dataIndex - 1]) + ',' + \
                                    str(self.dataK[self.dataIndex - 1]) + ',' + str(
                                self.dataL[self.dataIndex - 1])
                            self.ui.standard_data.append(str_0.replace('\r\n', ''))
                        else:
                            str_0 = str(self.dataX[-1]) + ',' + str(self.dataY[-1]) + ',' + str(self.dataZ[-1]) + ',' \
                                    + str(self.dataH[-1]) + ',' + str(self.dataI[-1]) + ',' + str(self.dataJ[-1]) + \
                                    ',' + str(self.dataK[-1]) + ',' + str(self.dataL[-1])
                            self.ui.standard_data.append(str_0.replace('\r\n', ''))
                    # 绘图
                    if self.ui.showsensor1.isChecked():
                        self.canvasX.setData(self.dataX)
                    else:
                        self.canvasX.clear()
                    if self.ui.showsensor2.isChecked():
                        self.canvasY.setData(self.dataY)
                    else:
                        self.canvasY.clear()
                    if self.ui.showsensor3.isChecked():
                        self.canvasZ.setData(self.dataZ)
                    else:
                        self.canvasZ.clear()
                    if self.ui.showsensor4.isChecked():
                        self.canvasH.setData(self.dataH)
                    else:
                        self.canvasH.clear()
                    if self.ui.showsensor5.isChecked():
                        self.canvasI.setData(self.dataI)
                    else:
                        self.canvasI.clear()
                    if self.ui.showsensor6.isChecked():
                        self.canvasJ.setData(self.dataJ)
                    else:
                        self.canvasJ.clear()
                    if self.ui.showforce.isChecked():
                        self.canvasK.setData(self.dataK)
                    else:
                        self.canvasK.clear()
                    if self.ui.showforce.isChecked():
                        self.canvasL.setData(self.dataL)
                    else:
                        self.canvasL.clear()
            except:
                pass
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 启动一个应用
    win = MainWindow()  # 实例化主窗口
    win.show()  # 将窗口控件显示在屏幕上
    sys.exit(app.exec_())  # 避免程序执行到这一行后直接退出