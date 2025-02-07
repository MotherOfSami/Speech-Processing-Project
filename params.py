import numpy as np

# PARAMS
ROOM_DIMENSIONS = [6, 5, 4]
FS = 16000
T_60_RANGE = [0.16, 0.32]
N_FFT_LIST = np.array([2048, 4096, 8192, 16384])  # Num of DFT Points
D = 0.05  # [m]
MIC_POSITIONS = [[3 + 3 * (D / 2), 2.5, 2], [3 + (D / 2), 2.5, 2], [3 - (D / 2), 2.5, 2], [3 - 3 * (D / 2), 2.5, 2]]

SOURCE_DEGREE = np.pi / 4  # Radians
SOURCE_RADIUS = 1  # [m]
SOURCE_POSITION = [3 + SOURCE_RADIUS * np.cos(SOURCE_DEGREE), 2.5 + + SOURCE_RADIUS * np.sin(SOURCE_DEGREE), 2]

INTERFERENCE_POSITION = [3 - SOURCE_RADIUS * np.cos(SOURCE_DEGREE), 2.5 + SOURCE_RADIUS * np.sin(SOURCE_DEGREE), 2]

WIN_LENGTH_LIST = N_FFT_LIST // 2
HOP_LENGTH_LIST = WIN_LENGTH_LIST // 2
RECORDING_TIME = 10  # [s]

TARGET_FOLDER_PATH = r'dataset/vcc2018_evaluation/VCC2SF1'
INTERFERENCE_FOLDER_PATH = 'dataset/vcc2018_evaluation/VCC2SM3'
