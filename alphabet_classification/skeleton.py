import math

import cv2 as cv
import numpy as np
import tensorflow as tf

BODY_PARTS = {
    "손목": 0,
    "엄지0": 1, "엄지1": 2, "엄지2": 3, "엄지3": 4,
    "검지0": 5, "검지1": 6, "검지2": 7, "검지3": 8,
    "중지0": 9, "중지1": 10, "중지2": 11, "중지3": 12,
    "약지0": 13, "약지1": 14, "약지2": 15, "약지3": 16,
    "소지0": 17, "소지1": 18, "소지2": 19, "소지3": 20,
}

POSE_PAIRS = [["손목", "엄지0"], ["엄지0", "엄지1"],
              ["엄지1", "엄지2"], ["엄지2", "엄지3"],
              ["손목", "검지0"], ["검지0", "검지1"],
              ["검지1", "검지2"], ["검지2", "검지3"],
              ["손목", "중지0"], ["중지0", "중지1"],
              ["중지1", "중지2"], ["중지2", "중지3"],
              ["손목", "약지0"], ["약지0", "약지1"],
              ["약지1", "약지2"], ["약지2", "약지3"],
              ["손목", "소지0"], ["소지0", "소지1"],
              ["소지1", "소지2"], ["소지2", "소지3"]]

# BODY_PARTS = {
#                 "Wrist": 0,
#                 "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
#                 "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
#                 "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
#                 "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
#                 "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
#             }

# POSE_PAIRS = [["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
#                ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
#                ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
#                ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
#                ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
#                ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
#                ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
#                ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
#                ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
#                ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]]

# 손가락 특정 부분이 검출되었는지 판정하기 위해 사용되는 스레쉬홀드
threshold = 0.1

# 모델을 구성하는 레이어들의 속성이 포함되어 있다.
# 이미지를 입력으로 학습이 완료된 모델을 사용하려면 필요하다
protoFile = "data/pose_deploy.prototxt"
# 학습 완료된 모델이 저장되어 있는 파일
weightsFile = "data/pose_iter_102000.caffemodel"

# 네트워크 불러오기
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
# 백엔드로 쿠다를 사용하여 속도 향상
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# 쿠다 디바이스에 계산을 요청
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# 영상을 가져오기 위해 웹캠과 연결
cap = cv.VideoCapture(0)

# 윈도우 크기
inputHeight = 368
inputWidth = 368
inputScale = 1.0 / 255

# 이미지 분류 모델 불러오기
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# 아무키나 누르면 프로그램 종료
while cv.waitKey(1) < 0:
    # 웹캠으로부터 영상 하나 가져오기
    hasFrame, frame = cap.read()
    # frame = cv.resize(frame, dsize=(320, 240), interpolation=cv.INTER_AREA)

    # 웹캠으로부터 영상을 가져올 수 없으면 프로그램 종료
    if not hasFrame:
        cv.waitKey()
        break

    # detection position
    max_x, max_y, min_x, min_y = 0, 0, 368, 368

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # (inputWidth, inputHeight)로 이미지 크기를 조정하고 4차원 행렬인 blob으로 변환합니다. 넽워크에 이미지를 입력하기 위한 전처리 과정
    # mean subtraction을 수행하여 원본 이미지의 픽셀에서 평균값을 뺌, 여기에선 평균값으로 0으로 주고 있어 빼지 않음
    inp = cv.dnn.blobFromImage(
        frame,  # 입력 이미지
        inputScale,  # 픽셀값 범위를 0 ~ 1 사이 값으로 정규화함 -> 인식 결과가 좋아짐
        (inputWidth, inputHeight),  # 컨볼루션 뉴럴 네트워크에서 요구하는 크기로 이미지 크기를 조정
        (0, 0, 0),  # 입력 영상에서 뺄 mean subtraction 값
        swapRB=False,  # OpenCV의 BGR 순서를 RGB로 바꾸기 위해 사용함 -> False는 바꾸지 않겠다는 뜻
        crop=False)
    # inp.shape -> (1, 3, 368, 368)

    # 네트워크 입력을 지정
    net.setInput(inp)
    # 네트워크의 출력을 얻기 위해 포워드 패스(forward pass)를 수행
    out = net.forward()

    points = []
    # 손가락을 구성하는 모든 부분에 대해
    for i in range(len(BODY_PARTS)):
        # 히트맵을 가지고
        heatMap = out[0, i, :, :]
        # 최대값(conf)과 최대값의 위치(point)를 찾음
        _, conf, _, point = cv.minMaxLoc(heatMap)

        # 원본 영상에 맞게 좌표를 조정 -> out.shape[]로 나누는 이유는 inp 생성 과정에서 (inputWidth, inputHeight)로 이미지 크기를 조정했기 때문임
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])

        '''
            히트맵의 최대값이 스레쉬홀드보다 크다면 손가락을 구성하는 특정 부분을 감지한 것임
            스레쉬 홀드가 작으면 손가락 부분이 잘 검출되지만 주변의 다른 물체를 같이 연결하여 손가락으로 오인식하는 것이 많아짐
            스레쉬홀드가 커지면 손가락 부분이 덜 검출되며 주변의 다른 물체를 같이 연결하여 손가락으로 오인식하는게 줄어드는 듯 함
        '''
        if conf > threshold:
            # 해당 부분에 원과 숫자를 표시한 후, 해당 좌표를 리스트에 삽입
            cv.circle(frame, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
            cv.putText(frame, "{}".format(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, lineType=cv.LINE_AA)
            points.append((x, y))
        else:
            points.append(None)

    # 모든 손가락 연결 관계에 대해
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        # 앞 과정에서 두 부분이 모두 검출되어 두 부분이 모두 존재하면 연결하는 선을 그려줌
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 1)

    # 추론하는데 걸린 시간을 화면에 출력
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # 손가락 검출 결과가 반영된 영상을 보여줌
    cv.imshow('Real-time Sign Language Recognition', frame)
    # cv.imshow('copy image', predict_img)
