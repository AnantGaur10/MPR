import pickle
from pathlib import Path
import os

USER_DATA_DIR = Path('user_data')
try:
    if not USER_DATA_DIR.exists():
        USER_DATA_DIR.mkdir(exist_ok=True)
except Exception as e:
    print(f"Error initializing USER_DATA_DIR: {e}")

def record_sign_to_file(user_id, sign_name, landmarks_list):
    try:
        user_dir = USER_DATA_DIR / str(user_id)
        if not user_dir.exists():
            user_dir.mkdir(exist_ok=True)
        pickle_path = user_dir / 'custom_landmarks.pickle'
        storage = {'data': [], 'labels': []} 
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    storage = pickle.load(f)
            except EOFError:
                pass
            except Exception as e:
                print(f"Error reading custom_landmarks.pickle: {e}")
                pass
        for lm in landmarks_list:
            if len(lm) == 42:
                storage['data'].append(lm)
                storage['labels'].append(sign_name)
        with open(pickle_path, 'wb') as f:
            pickle.dump(storage, f)
        return True
    except Exception as e:
        print(f"Error in record_sign_to_file: {e}")
        return False
