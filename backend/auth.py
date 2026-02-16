import os
import jwt
import bcrypt
import uuid
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
from dotenv import load_dotenv
from models import TokenBlocklist
from database import SessionLocal

load_dotenv()

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-secret-key-change-in-production')
JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 86400))

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        'user_id': user_id,
        'email': email,
        'jti': str(uuid.uuid4()),
        'exp': datetime.utcnow() + timedelta(seconds=JWT_ACCESS_TOKEN_EXPIRES),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception as e:
        print(f"Error in decode_token: {e}")
        return None

def is_token_revoked(jti: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(TokenBlocklist).filter(TokenBlocklist.jti == jti).first() is not None
    except Exception as e:
        print(f"Error in is_token_revoked: {e}")
        return False
    finally:
        db.close()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            token = None
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
            if not token:
                token = request.args.get('token')
            if not token:
                return jsonify({'error': 'Token is missing'}), 401
            payload = decode_token(token)
            if not payload:
                return jsonify({'error': 'Token is invalid or expired'}), 401
            jti = payload.get('jti')
            if jti and is_token_revoked(jti):
                 return jsonify({'error': 'Token has been revoked'}), 401
            g.user_id = payload.get('user_id')
            g.user_email = payload.get('email')
            g.token_jti = jti
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Error in token_required wrapper: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
    return decorated

def get_user_from_ws_token(token: str) -> dict | None:
    try:
        if not token:
            return None
        payload = decode_token(token)
        if not payload:
            return None
        jti = payload.get('jti')
        if jti and is_token_revoked(jti):
            return None
        return payload
    except Exception as e:
        print(f"Error in get_user_from_ws_token: {e}")
        return None
