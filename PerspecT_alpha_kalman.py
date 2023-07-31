import cv2
import numpy as np
import yaml
import json
import random
from filterpy.kalman import KalmanFilter


def perspective_transform(frame):
    height, width = frame.shape[:2]
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    return transformed_frame


# Function to extract pose keypoints from JSON
def extract_keypoints(json_data):
    keypoints = []
    for person in json_data:
        keypoints.append([person["idx"], person["keypoints"]])
    return keypoints


circles = []  # make trajectory


def visualize_keypoints(image, keypoints):
    # Get filtered keypoints
    filtered_keypoints = kalman_filter(keypoints)

    filtered_coord = []

    for person_keypoints in filtered_keypoints:
        person_idx = person_keypoints[0]
        random.seed(person_idx)
        # Get contact point
        x_c = int(person_keypoints[1][0])
        y_c = int(person_keypoints[1][1])

        point = np.array([[x_c, y_c]]).astype(np.float32)
        point = (
            cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix)
            .reshape(-1, 2)
            .astype(np.int16)
        )
        filtered_coord.append(point)
        cv2.circle(
            image,
            (point[0][0], point[0][1]),
            80,
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
            -1,
        )
        cv2.circle(image, (point[0][0], point[0][1]), 10, (0, 0, 255), -1)
        circles.append((point[0][0], point[0][1]))

    for circle in circles:
        cv2.circle(image, circle, 10, (0, 0, 255), -1)
    return image


# Kalman filter
def kalman_filter(keypoints, process_noise=1, measurement_noise=0.001):
    # return filtered coordinates

    dt = 1 / 29.97  # 1 / fps
    filtered_coord = []
    for point in keypoints:
        kf = KalmanFilter(2, 2)  # 2 state variables, 2 measurement variables

        # Average of both ankles coordinates
        c_x = int((point[1][15 * 3] + point[1][16 * 3]) / 2)
        c_y = int((point[1][15 * 3 + 1] + point[1][16 * 3 + 1]) / 2)

        kf.x = np.array([0, 0])  # state variables
        kf.F = np.array([[1, dt], [0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0], [0, 1]])  # measurement func
        kf.R *= measurement_noise  # measurement noise(uncertainty)

        kf.predict()  # predict step
        kf.update([[c_x], [c_y]])  # update step

        filtered_coord.append([point[0], kf.x])

    return filtered_coord


# YAML 파일 경로
config_file = "config/config_alpha.yaml"

# YAML 파일 로드
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# 변환에 사용할 원본과 목표 좌표
src_points = np.float32(config["src_points"])
dst_points = np.float32(config["dst_points"])
# 매핑 좌표 인자로 전달하면, 변환에 필요한 3x3 변환 행렬 리턴
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 입력 비디오 파일 경로
input_video_file = config["input_file"]

# JSON 파일이 저장된 파일 경로
json_file = config["json_file"]
# JSON 파일 로드
with open(json_file, "r") as f:
    data = json.load(f)

# 결과로 저장될 출력 비디오 파일 경로
output_video_file = config["output_file"]

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_file)
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

# 비디오 작성자 초기화
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

# 비디오 프레임별로 처리
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임에 perspective transform 적용
    transformed_frame = perspective_transform(frame)

    frame = []
    for index in data:
        if index["image_id"] == f"{frame_idx}.jpg":
            frame.append(index)

    keypoints = extract_keypoints(frame)

    transformed_frame = visualize_keypoints(transformed_frame, keypoints)

    # 변환된 프레임을 출력 비디오에 작성
    out.write(transformed_frame)

    # 변환된 프레임 출력
    cv2.imshow("Transformed Frame", transformed_frame)
    if cv2.waitKey(1) == ord("q"):
        break

    frame_idx += 1

# 비디오 파일 닫기
cap.release()
out.release()
cv2.destroyAllWindows()
