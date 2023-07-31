import cv2
import numpy as np
import yaml
import json
import random

def perspective_transform(frame):
    height, width = frame.shape[:2]
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    return transformed_frame

def get_crosspt(x11,y11, x12,y12, x21,y21, x22,y22):
    if x12 == x11:
        cx = x12
        cy = (y21 + y22) / 2
    elif x22 == x21:
        cx = x22
        cy = (y11 + y12) / 2
    else:
        m1 = (y12 - y11) / (x12 - x11)
        m2 = (y22 - y21) / (x22 - x21)
        cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
        cy = m1 * (cx - x11) + y11
    return cx, cy

def get_contactpoint(person_keypoints):
    x_c = int((person_keypoints[1][15 * 3] + person_keypoints[1][16 * 3]) / 2)
    y_c = int((person_keypoints[1][15 * 3 + 1] + person_keypoints[1][16 * 3 + 1]) / 2)
    point = np.array([[x_c, y_c]]).astype(np.float32)
    point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.int16)
    return point

def get_contactpoint_2(person_keypoints):
    chest_x = int((person_keypoints[1][5 * 3] + person_keypoints[1][6 * 3]) / 2)
    chest_y = int((person_keypoints[1][5 * 3 + 1] + person_keypoints[1][6 * 3 + 1]) / 2)
    weist_x = int((person_keypoints[1][11 * 3] + person_keypoints[1][12 * 3]) / 2)
    weist_y = int((person_keypoints[1][11 * 3 + 1] + person_keypoints[1][12 * 3 + 1]) / 2)
    foot1_x = int(person_keypoints[1][15 * 3])
    foot1_y = int(person_keypoints[1][15 * 3 + 1])
    foot2_x = int(person_keypoints[1][16 * 3])
    foot2_y = int(person_keypoints[1][16 * 3 + 1])
    cx, cy = get_crosspt(chest_x, chest_y, weist_x, weist_y, foot1_x, foot1_y, foot2_x, foot2_y)
    point = np.array([[cx, cy]]).astype(np.float32)
    point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.int16)
    return point

# Function to extract pose keypoints from JSON
def extract_keypoints(json_data):
    keypoints = []
    result = []
    for person in json_data:
        keypoints.append([person['idx'], person['keypoints']])
    for person_keypoints in keypoints:
        point = get_contactpoint(person_keypoints)
        result.append([person_keypoints[0], point[0]])
    results.append(result)
    return result

def visualize_keypoints(image, results, frame_idx):
    if Is_track:
        for person in results[frame_idx]:
            random.seed(person[0])
            cv2.circle(image, (person[1][0], person[1][1]), 80, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), -1)
        for result in results:
            for person in result:
                cv2.circle(image, (person[1][0], person[1][1]), 5, (0, 0, 255), -1)
    else:
        for person in results[frame_idx]:
            random.seed(person[0])
            cv2.circle(image, (person[1][0], person[1][1]), 80, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), -1)
            cv2.circle(image, (person[1][0], person[1][1]), 10, (0, 0, 255), -1)
    return image

# YAML 파일 경로
config_file = 'config/config_alpha.yaml'

# YAML 파일 로드
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

Is_track = True

# 변환에 사용할 원본과 목표 좌표
src_points = np.float32(config['src_points'])
dst_points = np.float32(config['dst_points'])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 입력 비디오 파일 경로
input_video_file = config['input_file']

# JSON 파일이 저장된 파일 경로
json_file = config['json_file']
# JSON 파일 로드
with open(json_file, 'r') as f:
    data = json.load(f)

# 결과로 저장될 출력 비디오 파일 경로
output_video_file = config['output_file']

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(input_video_file)
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 비디오 작성자 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

results = []

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
        if index['image_id'] == f'{frame_idx}.jpg':
            frame.append(index)

    result = extract_keypoints(frame)

    transformed_frame = visualize_keypoints(transformed_frame, results, frame_idx)
    
    # 변환된 프레임을 출력 비디오에 작성
    out.write(transformed_frame)

    # 변환된 프레임 출력
    cv2.imshow('Transformed Frame', transformed_frame)
    if cv2.waitKey(1) == ord('q'):
        break

    frame_idx += 1

# 비디오 파일 닫기
cap.release()
out.release()
cv2.destroyAllWindows()