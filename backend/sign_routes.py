import os
import pickle
import numpy as np
from flask import Blueprint, request, jsonify, g
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
from database import SessionLocal
from models import CustomSign, User
from auth import token_required
from utils import record_sign_to_file, USER_DATA_DIR

sign_bp = Blueprint('signs', __name__, url_prefix='/api/signs')

@sign_bp.route('/record', methods=['POST'])
@token_required
def record_sign():
    try:
        data = request.get_json()
        if not data or 'sign_name' not in data or 'landmarks' not in data:
            return jsonify({'error': 'Missing sign_name or landmarks'}), 400
        sign_name = data['sign_name'].strip().upper()
        landmarks_list = data['landmarks']
        if not isinstance(landmarks_list, list) or len(landmarks_list) == 0:
            return jsonify({'error': 'Landmarks must be a non-empty list'}), 400
        user_id = g.user_id
        success = record_sign_to_file(user_id, sign_name, landmarks_list)
        if not success:
            return jsonify({'error': 'Failed to save recording'}), 500
        return jsonify({
            'message': f'Recorded {len(landmarks_list)} samples for "{sign_name}"'
        }), 200
    except Exception as e:
        print(f"Error in record_sign route: {e}")
        return jsonify({'error': str(e)}), 500

@sign_bp.route('/train', methods=['POST'])
@token_required
def train_user_model():
    try:
        user_id = g.user_id
        user_dir = USER_DATA_DIR / str(user_id)
        pickle_path = user_dir / 'custom_landmarks.pickle'
        model_path = user_dir / 'user_model.p'
        if not pickle_path.exists():
            return jsonify({'error': 'No recorded data found for this user'}), 404
        try:
            with open(pickle_path, 'rb') as f:
                data_dict = pickle.load(f)
            X = np.array(data_dict['data'])
            y = np.array(data_dict['labels'])
            unique_labels = np.unique(y)
            
            if len(unique_labels) == 0:
                return jsonify({'error': 'No data to train'}), 400

            # --- NEGATIVE SAMPLING (Crucial for avoiding bias) ---
            # Try to load global samples to teach the model "what my sign is NOT"
            root_data_path = Path(__file__).parent.parent / 'data.pickle'
            negative_samples = []
            if root_data_path.exists():
                try:
                    with open(root_data_path, 'rb') as f:
                        global_data = pickle.load(f)
                        g_X = np.array(global_data['data'])
                        # Take 100 random samples from global dataset
                        idx = np.random.choice(len(g_X), min(100, len(g_X)), replace=False)
                        negative_samples = g_X[idx]
                except: pass
            
            if len(negative_samples) == 0:
                # Fallback: Generate jittered versions or some "neutral" noise
                negative_samples = np.random.normal(0, 0.05, (50, 42))

            X = np.vstack([X, negative_samples])
            y = np.concatenate([y, ["_NONE_"] * len(negative_samples)])
            unique_labels = np.unique(y)

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X, y)
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'labels': unique_labels.tolist()}, f)
            db = SessionLocal()
            try:
                for lbl in unique_labels:
                    existing = db.query(CustomSign).filter(
                        CustomSign.user_id == user_id,
                        CustomSign.sign_name == lbl
                    ).first()
                    if not existing:
                        new_sign = CustomSign(
                            user_id=user_id,
                            sign_name=lbl,
                            model_path=str(model_path),
                            sample_count=str(len([l for l in y if l == lbl]))
                        )
                        db.add(new_sign)
                    else:
                        existing.sample_count = str(len([l for l in y if l == lbl]))
                        existing.model_path = str(model_path)
                db.commit()
            except Exception as e:
                print(f"Error in train_user_model db operation: {e}")
                db.rollback()
                raise e
            finally:
                db.close()
            return jsonify({
                'message': 'Model trained successfully',
                'signs': unique_labels.tolist(),
                'total_samples': len(X)
            }), 200
        except Exception as e:
            print(f"Error in train_user_model training operation: {e}")
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"Error in train_user_model route: {e}")
        return jsonify({'error': str(e)}), 500

@sign_bp.route('/list', methods=['GET'])
@token_required
def list_signs():
    try:
        db = SessionLocal()
        try:
            signs = db.query(CustomSign).filter(CustomSign.user_id == g.user_id).all()
            return jsonify({'signs': [s.to_dict() for s in signs]}), 200
        except Exception as e:
            print(f"Error in list_signs db operation: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in list_signs route: {e}")
        return jsonify({'error': str(e)}), 500

@sign_bp.route('/delete/<sign_id>', methods=['DELETE'])
@token_required
def delete_sign(sign_id):
    try:
        db = SessionLocal()
        try:
            import uuid
            try:
                if isinstance(sign_id, str):
                    query_id = uuid.UUID(sign_id)
                else:
                    query_id = sign_id
            except ValueError:
                return jsonify({'error': 'Invalid sign ID format'}), 400
            sign = db.query(CustomSign).filter(
                CustomSign.id == query_id,
                CustomSign.user_id == g.user_id
            ).first()
            if not sign:
                return jsonify({'error': 'Sign not found'}), 404
            sign_name = sign.sign_name
            db.delete(sign)
            db.commit()
            user_dir = USER_DATA_DIR / str(g.user_id)
            pickle_path = user_dir / 'custom_landmarks.pickle'
            model_path = user_dir / 'user_model.p'
            if model_path.exists():
                try:
                    os.remove(model_path)
                except Exception as e:
                    print(f"Error removing model file: {e}")
            if pickle_path.exists():
                try:
                    with open(pickle_path, 'rb') as f:
                        storage = pickle.load(f)
                    new_data = []
                    new_labels = []
                    for d, l in zip(storage['data'], storage['labels']):
                        if l != sign_name:
                            new_data.append(d)
                            new_labels.append(l)
                    with open(pickle_path, 'wb') as f:
                        pickle.dump({'data': new_data, 'labels': new_labels}, f)
                except Exception as e:
                    print(f"Error in delete_sign pickle operation: {e}")
                    pass
            return jsonify({'message': f'Sign "{sign_name}" deleted and model reset. Please retrain to update recognition.'}), 200
        except Exception as e:
            print(f"Error in delete_sign db operation: {e}")
            db.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in delete_sign route: {e}")
        return jsonify({'error': str(e)}), 500
