import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
from deepface import DeepFace

st.title('Prince or Not: Deepfake Detection Tool')

st.write('Upload an image or video of the British prince to check if it is a deepfake.')

file_type = st.radio('Select file type:', ['Image', 'Video'])
uploaded_file = st.file_uploader('Upload file', type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

def detect_deepfake_image(image: Image.Image):
    # Convert PIL image to numpy array
    img_np = np.array(image.convert('RGB'))
    try:
        result = DeepFace.analyze(img_np, actions=['deepfake'], enforce_detection=False)
        is_deepfake = result['deepfake']['result']
        confidence = result['deepfake']['score']
        return {'is_deepfake': is_deepfake, 'confidence': confidence}
    except Exception as e:
        return {'is_deepfake': None, 'confidence': 0.0, 'error': str(e)}

def detect_deepfake_video(video_path: str, num_frames: int = 10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    errors = []
    for i in np.linspace(0, frame_count - 1, num=num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detect_deepfake_image(image)
        if res.get('error'):
            errors.append(res['error'])
            continue
        results.append(res)
    cap.release()
    if not results:
        return {'is_deepfake': None, 'confidence': 0.0, 'error': '\n'.join(errors) if errors else 'No frames processed.'}
    # Majority vote for deepfake status
    deepfake_votes = [r['is_deepfake'] for r in results if r['is_deepfake'] is not None]
    avg_confidence = np.mean([r['confidence'] for r in results if r['is_deepfake'] is not None])
    is_deepfake = sum(deepfake_votes) > len(deepfake_votes) / 2
    return {'is_deepfake': is_deepfake, 'confidence': avg_confidence}

if uploaded_file is not None:
    if file_type == 'Image' and uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        result = detect_deepfake_image(image)
        if result.get('error'):
            st.error(f"Detection error: {result['error']}")
        else:
            st.write(f"**Deepfake:** {'Yes' if result['is_deepfake'] else 'No'}")
            st.write(f"**Confidence:** {result['confidence']:.2f}")
    elif file_type == 'Video' and uploaded_file.type.startswith('video'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        result = detect_deepfake_video(tfile.name)
        if result.get('error'):
            st.error(f"Detection error: {result['error']}")
        else:
            st.write(f"**Deepfake:** {'Yes' if result['is_deepfake'] else 'No'}")
            st.write(f"**Confidence:** {result['confidence']:.2f}")
    else:
        st.warning('Please upload a valid file type for your selection.') 