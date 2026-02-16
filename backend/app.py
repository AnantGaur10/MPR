from flask import Flask, render_template, send_file, jsonify, request
from flask_sock import Sock
from flask_cors import CORS
import os
import subprocess
import logging
from pathlib import Path
import base64
import io
import numpy as np
import cv2
from PIL import Image
import pickle
import mediapipe as mp
from collections import deque, Counter
import pyttsx3
from threading import Lock, Thread, Event
import queue
import time
import json
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlparse

load_dotenv()

app = Flask(__name__)
CORS(app)
sock = Sock(app)

from database import SessionLocal
from models import CustomSign, User

from auth_routes import auth_bp
from sign_routes import sign_bp
app.register_blueprint(auth_bp)
app.register_blueprint(sign_bp)

USER_DATA_DIR = Path('user_data')
USER_DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

DOWNLOADS_FOLDER = Path.cwd() / 'downloads'
DOWNLOADS_FOLDER.mkdir(exist_ok=True)

INSTALLER_NAME = 'Sign Language Gestures Setup 1.0.0.exe'
INSTALLER_PATH = DOWNLOADS_FOLDER / INSTALLER_NAME

LABELS_DICT = {i: chr(65 + i) for i in range(26)}
LABELS_DICT[26] = "ENTER"
LABELS_DICT[27] = "SPACE"
LABELS_DICT[28] = "THUMBS_DOWN"

MODEL_PATH = (Path(__file__).parent / 'model.p').resolve()
model = None
mp_hands = None
hands = None
tts_engine = None
tts_lock = Lock()

BUFFER_SIZE = 3
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES_REQUIRED = 12
STABLE_FRAMES_ENTER = 7

hand_position_buffer = deque(maxlen=10)

def load_ml_model():
    global model
    print(f"DEBUG: Loading model from {MODEL_PATH}")
    try:
        if not MODEL_PATH.exists():
            print(f"DEBUG: Model file does not exist at {MODEL_PATH}")
            return False
        
        print(f"DEBUG: File exists. Size: {MODEL_PATH.stat().st_size} bytes")
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
            print("DEBUG: Pickle loaded successfully. Keys:", model_dict.keys() if isinstance(model_dict, dict) else "Not a dict")
            
        model = model_dict['model']
        print(f"DEBUG: Model object loaded: {type(model)}")
        
        if not hasattr(model, 'predict_proba'):
             print("DEBUG: Model missing predict_proba method")
             return False
             
        print("DEBUG: Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error in load_ml_model: {e}")
        import traceback
        traceback.print_exc()
        return False

def init_mediapipe():
    global mp_hands, hands
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.70
        )
        return True
    except Exception as e:
        print(f"Error in init_mediapipe: {e}")
        return False

def init_tts():
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 100)
        return True
    except Exception as e:
        print(f"Error in init_tts: {e}")
        if os.name == 'posix':
            print("Fallback to espeak CLI for Linux/Docker")
            tts_engine = "espeak_cli"
            return True
        return False

def process_landmarks(hand_landmarks):
    try:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        wrist_x, wrist_y = x_coords[0], y_coords[0]
        data_aux = []
        for x, y in zip(x_coords, y_coords):
            data_aux.extend([x - wrist_x, y - wrist_y])
        return data_aux if len(data_aux) == 42 else None
    except Exception as e:
        print(f"Error in process_landmarks: {e}")
        return None

def is_hand_stable(hand_landmarks, threshold=0.02):
    try:
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        hand_position_buffer.append((wrist_x, wrist_y))
        if len(hand_position_buffer) < 5:
            return False
        positions = np.array(list(hand_position_buffer))
        variance = np.var(positions, axis=0).sum()
        return variance < threshold
    except Exception as e:
        print(f"Error in is_hand_stable: {e}")
        return False

def predict_sign(image_data, pred_buffer, frame_count, local_hand_buffer, user_model=None):
    local_model = model
    try:
        if local_model is None or hands is None:
            return None, 0.0, False, False, None, "NONE"
        
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
        else:
            img_bytes = image_data
            
        img = Image.open(io.BytesIO(img_bytes))
        # Convert PIL (RGB) to OpenCV (BGR) and flip
        frame = cv2.flip(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Use wrist-relative normalization (Required by global model.p)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            wrist_x, wrist_y = x_coords[0], y_coords[0]
            data_aux = []
            for x, y in zip(x_coords, y_coords):
                data_aux.extend([x - wrist_x, y - wrist_y])
            
            features = np.array(data_aux).reshape(1, -1)
            
            # 1. Get Global Model Prediction (Baseline)
            global_probs = model.predict_proba(features)[0]
            global_idx = np.argmax(global_probs)
            global_conf = global_probs[global_idx]
            global_gesture = LABELS_DICT.get(global_idx, "UNKNOWN")

            # 2. Get Custom User Model Prediction (If exists)
            single_gesture = global_gesture
            single_conf = global_conf
            single_source = "GLOBAL"
            
            if user_model:
                try:
                    user_probs = user_model.predict_proba(features)[0]
                    user_idx = np.argmax(user_probs)
                    user_conf = user_probs[user_idx]
                    user_gesture = user_model.classes_[user_idx]
                    
                    # Logic: If Global is very confident (>85%), keep it for basic letters.
                    # This prevents the "only one custom sign" bias from taking over everything.
                    if user_gesture != "_NONE_":
                        # If User model is extremely sure (>98%) AND Global is confused (<70%)
                        if user_conf > 0.98 and global_conf < 0.70:
                            single_gesture = user_gesture
                            single_conf = user_conf
                            single_source = "USER"
                        # If User is significantly better than Global
                        elif user_conf > global_conf + 0.30:
                            single_gesture = user_gesture
                            single_conf = user_conf
                            single_source = "USER"
                except Exception as e:
                    print(f"User model prediction error: {e}")

            # 3. Apply Voting Buffer (Matching training/inference_classifier.py logic)
            # Store both the gesture and its individual confidence for smoothing
            pred_buffer.append((single_gesture, single_conf))
            local_hand_buffer.append(True)
            
            # Count frequency for stabilization
            just_gestures = [p[0] for p in pred_buffer]
            counts = Counter(just_gestures)
            most_common_gesture, occurances = counts.most_common(1)[0]
            
            # Weighted Confidence: Average the probabilities of samples that match the most common gesture
            matching_confs = [p[1] for p in pred_buffer if p[0] == most_common_gesture]
            avg_conf = sum(matching_confs) / len(matching_confs)
            
            # Also factor in voting stability (if only 1/3 match, drop effective confidence)
            voting_weight = occurances / len(pred_buffer)
            final_stabilized_conf = avg_conf * voting_weight
            
            stable = is_hand_stable(hand_landmarks)
            
            return most_common_gesture, final_stabilized_conf, True, stable, data_aux, single_source
        else:
            pred_buffer.clear()
            local_hand_buffer.append(False)
            return None, 0, False, False, None, "NONE"
    except Exception as e:
        print(f"Error in predict_sign: {e}")
        return None, 0, False, False, None, "NONE"

def text_to_speech(text):
    try:
        # Docker/Linux Fallback using espeak CLI (bypass audio driver)
        if tts_engine == "espeak_cli" or os.name == 'posix':
            temp_file = "temp_audio.wav"
            try:
                subprocess.run(["espeak", "-s", "150", "-w", temp_file, text], check=True)
                if os.path.exists(temp_file):
                    with open(temp_file, 'rb') as f:
                        audio_bytes = f.read()
                    os.remove(temp_file)
                    return audio_bytes
            except Exception as e:
                 print(f"Espeak error: {e}")
                 pass
            # If espeak failed or not posix, try standard way if engine exists
            
        with tts_lock:
            if tts_engine is None or tts_engine == "espeak_cli":
                return None
            temp_file = "temp_audio.wav"
            tts_engine.save_to_file(text, temp_file)
            tts_engine.runAndWait()
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    audio_bytes = f.read()
                os.remove(temp_file)
                return audio_bytes
            return None
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None


@app.route('/')
def home():
    try:
        installer_exists = INSTALLER_PATH.exists()
        installer_size = None
        if installer_exists:
            size_bytes = INSTALLER_PATH.stat().st_size
            installer_size = f"{size_bytes / (1024 * 1024):.2f} MB"
        return render_template('index.html',
                             installer_exists=installer_exists,
                             installer_size=installer_size,
                             installer_name=INSTALLER_NAME)
    except Exception as e:
        print(f"Error in home route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/overlay')
def overlay():
    return render_template('overlay.html')

@app.route('/download')
def download():
    try:
        if not INSTALLER_PATH.exists():
            return jsonify({'error': 'Installer not found'}), 404
        return send_file(
            INSTALLER_PATH,
            as_attachment=True,
            download_name=INSTALLER_NAME,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        print(f"Error in download route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/installer-info')
def installer_info():
    try:
        if not INSTALLER_PATH.exists():
            return jsonify({'available': False, 'message': 'Installer not available'}), 404
        size_bytes = INSTALLER_PATH.stat().st_size
        return jsonify({
            'available': True,
            'filename': INSTALLER_NAME,
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'version': '1.0.0'
        })
    except Exception as e:
        print(f"Error in installer_info route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-status')
def model_status():
    return jsonify({
        'model_loaded': model is not None,
        'mediapipe_loaded': hands is not None,
        'tts_available': tts_engine is not None,
        'model_type': 'RandomForest + MediaPipe + LLM',
        'gestures': list(LABELS_DICT.values()) if model else []
    })

def frame_receiver(ws, frame_queue, control_queue, stop_event):
    frame_count = 0
    try:
        while not stop_event.is_set():
            data = ws.receive()
            if data is None:
                break
            if isinstance(data, str):
                if data.startswith('data:image'):
                    try:
                        frame_queue.put_nowait(data)
                        frame_count += 1
                    except queue.Full:
                        pass
                else:
                    try:
                        control_msg = json.loads(data)
                        if 'type' in control_msg:
                            control_queue.put_nowait(control_msg)
                    except:
                        pass
            elif isinstance(data, bytes):
                try:
                    frame_queue.put_nowait(data)
                    frame_count += 1
                except queue.Full:
                    pass
    except Exception as e:
        print(f"Error in frame_receiver: {e}")
        pass
    finally:
        stop_event.set()

class BufferedLogger:
    def __init__(self, filename="predictions.log", buffer_size=15):
        self.filename = filename # Keep for legacy but we won't write to it
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = Lock()
    
    def log(self, message):
        with self.lock:
            log_line = f"{time.strftime('%H:%M:%S')} | {message}"
            self.buffer.append(log_line)
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        try:
            # Batch print to terminal instead of writing to disk
            for line in self.buffer:
                print(line)
            self.buffer = []
        except Exception as e:
            print(f"Error flushing buffered logger: {e}")

prediction_logger = BufferedLogger(buffer_size=15)

def send_state_update(ws, current_word, sentence):
    try:
        state = {
            'type': 'STATE_UPDATE',
            'word': current_word,
            'sentence': sentence.strip(),
            'timestamp': time.time()
        }
        ws.send(json.dumps(state))
    except Exception as e:
        print(f"Error in send_state_update: {e}")
        pass

def frame_processor(ws, frame_queue, control_queue, stop_event, user_id=None):
    def load_user_model_from_disk(uid):
        if not uid: return None
        path = USER_DATA_DIR / str(uid) / 'user_model.p'
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f).get('model')
            except Exception as e:
                print(f"Error loading user model: {e}")
        return None

    user_model = load_user_model_from_disk(user_id)
    current_word = ""
    sentence = ""
    last_gesture = ""
    same_gesture_count = 0
    conversation_history = []
    double_letter_frames_required = 15
    letter_hold_count = 0
    letter_already_doubled = False
    no_hand_frames = 0
    last_confirmed_gesture = None
    last_confirmed_time = 0
    gesture_cooldown = 1.8
    is_paused = False
    total_predictions = 0
    successful_predictions = 0
    pred_buffer = deque(maxlen=BUFFER_SIZE)
    local_hand_buffer = deque(maxlen=10)
    frame_count = 0
    hand_detected_frames = 0
    is_recording = False
    record_sign_name = ""
    recorded_landmarks = []
    RECORD_LIMIT = 60
    send_state_update(ws, current_word, sentence)
    try:
        while not stop_event.is_set():
            try:
                control = control_queue.get_nowait()
                if control['type'] == 'PAUSE':
                    is_paused = True
                elif control['type'] == 'RESUME':
                    is_paused = False
                elif control['type'] == 'BACKSPACE':
                    if current_word:
                        current_word = current_word[:-1]
                        send_state_update(ws, current_word, sentence)
                        last_gesture = ""
                        same_gesture_count = 0
                        letter_hold_count = 0
                        letter_already_doubled = False
                elif control['type'] == 'SPACE':
                    if current_word:
                        sentence += current_word + " "
                        current_word = ""
                        send_state_update(ws, current_word, sentence)
                    last_gesture = ""
                    same_gesture_count = 0
                    letter_hold_count = 0
                    letter_already_doubled = False
                    no_hand_frames = 0
                elif control['type'] == 'SUBMIT_SENTENCE':
                    if current_word:
                        sentence += current_word + " "
                    final_sentence = sentence.strip()
                    if final_sentence:
                        conversation_history.append(final_sentence)
                        audio = text_to_speech(final_sentence)
                        if audio:
                            ws.send(audio)
                        sentence = ""
                        current_word = ""
                        last_gesture = ""
                        same_gesture_count = 0
                        letter_hold_count = 0
                        letter_already_doubled = False
                        no_hand_frames = 0
                        send_state_update(ws, current_word, sentence)
                elif control['type'] == 'GET_STATE':
                    send_state_update(ws, current_word, sentence)
                elif control['type'] == 'RECORD_START':
                    sign_name = control.get('sign_name', 'UNKNOWN').upper()
                    is_recording = True
                    record_sign_name = sign_name
                    recorded_landmarks = []
                    prediction_logger.log(f"User: {user_id} | STARTED RECORDING: {sign_name}")
                    ws.send(json.dumps({'type': 'RECORD_STATUS', 'status': 'STARTING', 'sign': sign_name}))
                elif control['type'] == 'RESET':
                    current_word = ""
                    sentence = ""
                    last_gesture = ""
                    same_gesture_count = 0
                    letter_hold_count = 0
                    letter_already_doubled = False
                    no_hand_frames = 0
                    send_state_update(ws, current_word, sentence)
                elif control['type'] == 'RELOAD_MODEL':
                    user_model = load_user_model_from_disk(user_id)
                    prediction_logger.log(f"User: {user_id} | MODEL RELOADED | Found: {user_model is not None}")
            except queue.Empty:
                pass
            if is_paused:
                time.sleep(0.1)
                continue
            try:
                frame_data = frame_queue.get(block=True, timeout=0.1)
                frame_count += 1
                gesture, confidence, hand_detected, hand_stable, features, source = predict_sign(
                    frame_data, pred_buffer, frame_count, local_hand_buffer, user_model=user_model
                )
                if gesture:
                    if is_recording:
                        prediction_logger.log(f"User: {user_id} | RECORDING: {record_sign_name} | AI Thinks: {gesture} ({confidence:.2f}) | Frame: {len(recorded_landmarks) + 1}/{RECORD_LIMIT}")
                    else:
                        prediction_logger.log(f"User: {user_id} | Holding: {gesture} | Confidence: {confidence:.2f} | Stable: {hand_stable} | Source: {source}")
                
                if is_recording and hand_detected and features:
                    recorded_landmarks.append(features)
                    if len(recorded_landmarks) % 10 == 0:
                        ws.send(json.dumps({
                            'type': 'RECORD_STATUS', 
                            'status': 'RECORDING', 
                            'count': len(recorded_landmarks),
                            'limit': RECORD_LIMIT
                        }))
                    if len(recorded_landmarks) >= RECORD_LIMIT:
                        is_recording = False
                        from utils import record_sign_to_file
                        success = record_sign_to_file(user_id, record_sign_name, recorded_landmarks)
                        
                        # Immediately update database so it appears in the list
                        if success:
                            db_session = SessionLocal()
                            try:
                                existing = db_session.query(CustomSign).filter(
                                    CustomSign.user_id == user_id,
                                    CustomSign.sign_name == record_sign_name
                                ).first()
                                
                                # Count total samples for this sign from the pickle file
                                user_dir = USER_DATA_DIR / str(user_id)
                                pickle_path = user_dir / 'custom_landmarks.pickle'
                                total_samples = RECORD_LIMIT
                                if pickle_path.exists():
                                    try:
                                        with open(pickle_path, 'rb') as f:
                                            storage = pickle.load(f)
                                        total_samples = len([l for l in storage['labels'] if l == record_sign_name])
                                    except: pass
                                
                                if not existing:
                                    new_sign = CustomSign(
                                        user_id=user_id,
                                        sign_name=record_sign_name,
                                        model_path="", # Not trained yet
                                        sample_count=str(total_samples)
                                    )
                                    db_session.add(new_sign)
                                else:
                                    existing.sample_count = str(total_samples)
                                db_session.commit()
                            except Exception as e:
                                print(f"Error updating DB after recording: {e}")
                                db_session.rollback()
                            finally:
                                db_session.close()

                        prediction_logger.log(f"User: {user_id} | RECORDING COMPLETE: {record_sign_name} | Samples: {len(recorded_landmarks)} | Success: {success}")
                        ws.send(json.dumps({
                            'type': 'RECORD_STATUS', 
                            'status': 'COMPLETE', 
                            'sign': record_sign_name,
                            'success': success
                        }))
                        recorded_landmarks = []
                if hand_detected:
                    hand_detected_frames += 1
                    no_hand_frames = 0
                else:
                    no_hand_frames += 1
                if frame_count % 100 == 0:
                    hand_detected_frames = 0
                if gesture and confidence >= CONFIDENCE_THRESHOLD and hand_stable:
                    total_predictions += 1
                    if gesture == last_gesture:
                        same_gesture_count += 1
                        if gesture not in ["SPACE", "ENTER", "THUMBS_DOWN"] and current_word and gesture == current_word[-1]:
                            letter_hold_count += 1
                            if letter_hold_count >= double_letter_frames_required and not letter_already_doubled:
                                current_time = time.time()
                                if (last_confirmed_gesture != gesture or 
                                    current_time - last_confirmed_time >= gesture_cooldown):
                                    current_word += gesture
                                    send_state_update(ws, current_word, sentence)
                                    letter_already_doubled = True
                                    letter_hold_count = 0
                                    last_confirmed_gesture = gesture
                                    last_confirmed_time = current_time
                    else:
                        last_gesture = gesture
                        same_gesture_count = 1
                        letter_hold_count = 0
                        letter_already_doubled = False
                    if same_gesture_count >= STABLE_FRAMES_ENTER:
                        current_time = time.time()
                        if (last_confirmed_gesture != gesture or 
                            current_time - last_confirmed_time >= gesture_cooldown):
                            if gesture == "THUMBS_DOWN":
                                if current_word:
                                    current_word = current_word[:-1]
                                    send_state_update(ws, current_word, sentence)
                                    successful_predictions += 1
                            elif gesture == "ENTER":
                                if current_word:
                                    sentence += current_word + " "
                                final_sentence = sentence.strip()
                                if final_sentence:
                                    conversation_history.append(final_sentence)
                                    audio = text_to_speech(final_sentence)
                                    if audio:
                                        ws.send(audio)
                                    sentence = ""
                                    current_word = ""
                                    send_state_update(ws, current_word, sentence)
                                successful_predictions += 1
                            elif gesture == "SPACE":
                                if current_word:
                                    sentence += current_word + " "
                                    current_word = ""
                                    send_state_update(ws, current_word, sentence)
                                successful_predictions += 1
                            elif not current_word or gesture != current_word[-1]:
                                current_word += gesture
                                successful_predictions += 1
                                send_state_update(ws, current_word, sentence)
                            last_confirmed_gesture = gesture
                            last_confirmed_time = current_time
                            same_gesture_count = 0
                            letter_hold_count = 0
                            letter_already_doubled = False
                elif gesture and confidence >= CONFIDENCE_THRESHOLD and not hand_stable:
                    pass
                else:
                    if not hand_detected:
                        if last_gesture:
                            last_gesture = ""
                            same_gesture_count = 0
                            letter_hold_count = 0
                            letter_already_doubled = False
                            local_hand_buffer.clear()
            except queue.Empty:
                if stop_event.is_set():
                    break
            except Exception as e:
                print(f"Error in frame_processor (loop): {e}")
                pass
    except Exception as e:
        print(f"Error in frame_processor: {e}")
    finally:
        prediction_logger.flush()

@sock.route('/ws/ml')
def ml_websocket(ws):
    try:
        token = request.args.get('token')
        from auth import get_user_from_ws_token
        user_info = get_user_from_ws_token(token)
        if not user_info:
            ws.send(json.dumps({'error': 'Authentication required'}))
            ws.close()
            return
        if model is None or hands is None:
            path = "."  # current directory

            for name in os.listdir(path):
                full_path = os.path.join(path, name)
                
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    print(f"{name} - {size} bytes")
            print(os.listdir("."))
            print(hands)
            print("model is")
            print(model)
            ws.send(json.dumps({'error': 'Model not loaded'}))
            return
        frame_queue = queue.LifoQueue(maxsize=5)
        control_queue = queue.Queue()
        stop_event = Event()
        receiver = Thread(target=frame_receiver, args=(ws, frame_queue, control_queue, stop_event))
        processor = Thread(target=frame_processor, args=(ws, frame_queue, control_queue, stop_event, user_info.get('user_id')))
        receiver.start()
        processor.start()
        try:
            receiver.join()
        except Exception as e:
            print(f"Error in ml_websocket (receiver join): {e}")
            pass
        stop_event.set()
        processor.join(timeout=5)
    except Exception as e:
        print(f"Error in ml_websocket: {e}")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        from database import init_db
        init_db()
    except Exception as e:
        print(f"Error in main (init_db): {e}")
        pass
    load_ml_model()
    init_mediapipe()
    init_tts()
    app.run(host='0.0.0.0', port=8080, debug=False)