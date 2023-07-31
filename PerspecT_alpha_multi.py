import cv2
import numpy as np
import yaml
import json
import random

def perspective_transform(frame, input_idx):
    height, width = frame.shape[:2]
    transformed_frame = cv2.warpPerspective(frame, matrix[input_idx], (width, height))
    return transformed_frame

# Function to extract pose keypoints from JSON
def extract_keypoints(json_data):
    keypoints = []
    for person in json_data:
        keypoints.append([person['idx'], person['keypoints']])
    return keypoints

def visualize_keypoints(image, keypoints, input_idx):
    for person_keypoints in keypoints:
        person_idx = person_keypoints[0]
        random.seed(person_idx)
        for i in range(0, len(person_keypoints[1]), 3):
            # Get contact point
            x_c = int((person_keypoints[1][15 * 3] + person_keypoints[1][16 * 3]) / 2)
            y_c = int((person_keypoints[1][15 * 3 + 1] + person_keypoints[1][16 * 3 + 1]) / 2)
            point = np.array([[x_c, y_c]]).astype(np.float32)
            point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix[input_idx]).reshape(-1, 2).astype(np.int16)
            cv2.circle(image, (point[0][0], point[0][1]), 10, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), -1)
    return image

# YAML 파일 경로
config_file = 'config/config_alpha_multi.yaml'

# YAML 파일 로드
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

input_num = config['input_num']
# alpha = np.full((2010, 920, 3), 1.0 / input_num)
alpha = np.full((1080, 1920, 3), 1.0 / input_num)

src_points = []
matrix = []
input_video_file = []
json_file = []
data = []
cap = []
dst_points = np.float32(config['dst_points'])

for input_idx in range(input_num):
    src_points.append(np.float32(config[f'src_points_{input_idx}']))
    matrix.append(cv2.getPerspectiveTransform(src_points[input_idx], dst_points))
    input_video_file.append(config[f'input_file_{input_idx}'])
    json_file.append(config[f'json_file_{input_idx}'])
    with open(json_file[input_idx], 'r') as f:
        data.append(json.load(f))
    cap.append(cv2.VideoCapture(input_video_file[input_idx]))

# 결과로 저장될 출력 비디오 파일 경로
output_video_file = config['output_file']

# 비디오 캡처 객체 생성
if not cap[0].isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 속성 가져오기
fps = cap[0].get(cv2.CAP_PROP_FPS)
# frame_size = (int(cap[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 비디오 작성자 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_file, fourcc, fps, (1280, 720))
out = cv2.VideoWriter(output_video_file, fourcc, fps, (2010, 920))

# 비디오 프레임별로 처리
frame_idx = 0
while True:
    ret = []
    image = []
    transformed_frame = []
    keypoints = []
    for input_idx in range(input_num):
        ret_temp, image_temp = cap[input_idx].read()
        image.append(image_temp)
        if not ret_temp:
            break

        # 프레임에 perspective transform 적용
        transformed_frame.append(perspective_transform(image_temp, input_idx))

        frame = []
        for index in data[input_idx]:
            if index['image_id'] == f'{frame_idx}.jpg':
                frame.append(index)

        keypoints.append(extract_keypoints(frame))

    for input_idx in range(input_num):
        transformed_frame[input_idx] = visualize_keypoints(transformed_frame[input_idx], keypoints[input_idx], input_idx)

    # Blend video
    for input_idx in range(input_num):
        transformed_frame[input_idx] = transformed_frame[input_idx].astype(float)
        if input_idx == 0:
            blended_frame = cv2.multiply(alpha, transformed_frame[input_idx])
        else:
            frame_temp = cv2.multiply(alpha, transformed_frame[input_idx])
            blended_frame = cv2.add(blended_frame, frame_temp)
    # blended_frame = cv2.resize(blended_frame, (1280, 720))
    blended_frame = cv2.resize(blended_frame, (2010, 920))
    blended_frame = blended_frame / 255

    # 변환된 프레임을 출력 비디오에 작성
    out.write(blended_frame)

    # 변환된 프레임 출력
    cv2.imshow('Transformed Frame', blended_frame)
    if cv2.waitKey(1) == ord('q'):
        break

    frame_idx += 1

    # if frame_idx == 100:
    #     break

# 비디오 파일 닫기
for input_idx in range(input_num):
    cap[input_idx].release()
out.release()
cv2.destroyAllWindows()