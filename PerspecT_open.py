import cv2
import numpy as np
import yaml
import json
import glob

def perspective_transform(frame):
    height, width = frame.shape[:2]
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    return transformed_frame

# Function to extract pose keypoints from JSON
def extract_keypoints(json_data):
    keypoints = []
    for person in json_data['people']:
        keypoints.append(person['pose_keypoints_2d'])
    return keypoints

def visualize_keypoints(image, keypoints):
    for person_keypoints in keypoints:
        for i in range(0, len(person_keypoints), 3):
            # Get contact point
            x_c = int((person_keypoints[21 * 3] + person_keypoints[24 * 3]) / 2)
            y_c = int((person_keypoints[21 * 3 + 1] + person_keypoints[24 * 3 + 1]) / 2)
            point = np.array([[x_c, y_c]]).astype(np.float32)
            point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.int16)
            cv2.circle(image, (point[0][0], point[0][1]), 5, (0, 0, 255), -1)
    return image

# YAML 파일 경로
config_file = 'config/config_open.yaml'

# YAML 파일 로드
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# 변환에 사용할 원본과 목표 좌표
src_points = np.float32(config['src_points'])
dst_points = np.float32(config['dst_points'])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 입력 비디오 파일 경로
input_video_file = config['input_file']

# JSON 파일이 저장된 디렉토리 경로
json_dir = config['json_dir']

json_files = sorted(glob.glob(json_dir + '*.json'))

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

# 비디오 프레임별로 처리
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임에 perspective transform 적용
    transformed_frame = perspective_transform(frame)

    # 해당 프레임의 JSON 파일 경로
    json_file = json_files[frame_idx]

    # JSON 파일 로드
    with open(json_file, 'r') as f:
        data = json.load(f)

    keypoints = extract_keypoints(data)

    transformed_frame = visualize_keypoints(transformed_frame, keypoints)
    
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