from flask import Blueprint, request, jsonify, g
from sqlalchemy.exc import IntegrityError
from database import SessionLocal
from models import User
from auth import hash_password, verify_password, create_access_token, token_required

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        if not email or '@' not in email:
            return jsonify({'error': 'Valid email is required'}), 400
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        if not name or len(name) < 2:
            return jsonify({'error': 'Name must be at least 2 characters'}), 400
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.email == email).first()
            if existing:
                return jsonify({'error': 'Email already registered'}), 409
            user = User(
                email=email,
                password_hash=hash_password(password),
                name=name
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            token = create_access_token(str(user.id), user.email)
            return jsonify({
                'message': 'User created successfully',
                'token': token,
                'user': user.to_dict()
            }), 201
        except IntegrityError as e:
            print(f"IntegrityError in signup: {e}")
            db.rollback()
            return jsonify({'error': 'Email already registered'}), 409
        except Exception as e:
            print(f"Error in signup db operation: {e}")
            db.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in signup route: {e}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/signin', methods=['POST'])
def signin():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user or not verify_password(password, user.password_hash):
                return jsonify({'error': 'Invalid email or password'}), 401
            token = create_access_token(str(user.id), user.email)
            return jsonify({
                'message': 'Signed in successfully',
                'token': token,
                'user': user.to_dict()
            }), 200
        except Exception as e:
            print(f"Error in signin db operation: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in signin route: {e}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/me', methods=['GET'])
@token_required
def get_current_user():
    try:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == g.user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            return jsonify({'user': user.to_dict()}), 200
        except Exception as e:
            print(f"Error in get_current_user db operation: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in get_current_user route: {e}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/update', methods=['PUT'])
@token_required
def update_profile():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == g.user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            if 'name' in data and len(data['name'].strip()) >= 2:
                user.name = data['name'].strip()
            if 'password' in data and len(data['password']) >= 6:
                user.password_hash = hash_password(data['password'])
            db.commit()
            db.refresh(user)
            return jsonify({
                'message': 'Profile updated successfully',
                'user': user.to_dict()
            }), 200
        except Exception as e:
            print(f"Error in update_profile db operation: {e}")
            db.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in update_profile route: {e}")
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout():
    try:
        jti = g.get('token_jti')
        if not jti:
            return jsonify({'message': 'Logged out successfully'}), 200
        db = SessionLocal()
        try:
            from models import TokenBlocklist
            blocklist_entry = TokenBlocklist(jti=jti)
            db.add(blocklist_entry)
            db.commit()
            return jsonify({'message': 'Logged out successfully'}), 200
        except Exception as e:
            print(f"Error in logout db operation: {e}")
            db.rollback()
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    except Exception as e:
        print(f"Error in logout route: {e}")
        return jsonify({'error': str(e)}), 500
