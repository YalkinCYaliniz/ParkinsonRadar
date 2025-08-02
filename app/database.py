"""
Database setup and models for Parkinson Voice Analysis Application
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'pg-26612b69-alpaykeskin1411-891e.i.aivencloud.com',
    'port': 17569,
    'database': 'defaultdb',
    'user': 'avnadmin',
    'password': 'AVNS_H-2ZpSph-veKtUIPfTr'
}

class Database:
    def __init__(self):
        self.connection = None
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**DB_CONFIG)
            return self.connection
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def execute_query(self, query, params=None, fetch=False):
        """Execute a query with optional parameters"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
        except Exception as e:
            print(f"Query execution error: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def fetch_one(self, query, params=None):
        """Fetch one row from query"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            result = cursor.fetchone()
            cursor.close()
            return result
        except Exception as e:
            print(f"Fetch one error: {e}")
            return None

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, full_name=None, created_at=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.full_name = full_name
        self.created_at = created_at
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    @staticmethod
    def create_user(username, email, password, full_name=None):
        """Create a new user"""
        db = Database()
        db.connect()
        
        # Check if username or email already exists
        existing_user = db.fetch_one(
            "SELECT id FROM users WHERE username = %s OR email = %s",
            (username, email)
        )
        
        if existing_user:
            db.close()
            return None, "Kullanıcı adı veya email zaten kullanılıyor"
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Insert new user
        query = """
            INSERT INTO users (username, email, password_hash, full_name, created_at)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """
        
        try:
            cursor = db.connection.cursor()
            cursor.execute(query, (username, email, password_hash, full_name, datetime.now()))
            user_id = cursor.fetchone()[0]
            db.connection.commit()
            cursor.close()
            db.close()
            return user_id, "Kullanıcı başarıyla oluşturuldu"
        except Exception as e:
            db.close()
            return None, f"Kullanıcı oluşturma hatası: {e}"
    
    @staticmethod
    def get_user_by_id(user_id):
        """Get user by ID"""
        db = Database()
        db.connect()
        
        user_data = db.fetch_one(
            "SELECT * FROM users WHERE id = %s",
            (user_id,)
        )
        
        db.close()
        
        if user_data:
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                full_name=user_data['full_name'],
                created_at=user_data['created_at']
            )
        return None
    
    @staticmethod
    def get_user_by_username(username):
        """Get user by username"""
        db = Database()
        db.connect()
        
        user_data = db.fetch_one(
            "SELECT * FROM users WHERE username = %s",
            (username,)
        )
        
        db.close()
        
        if user_data:
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                full_name=user_data['full_name'],
                created_at=user_data['created_at']
            )
        return None
    
    def update_profile(self, full_name=None, email=None):
        """Update user profile"""
        db = Database()
        db.connect()
        
        updates = []
        params = []
        
        if full_name is not None:
            updates.append("full_name = %s")
            params.append(full_name)
            
        if email is not None:
            updates.append("email = %s")
            params.append(email)
        
        if updates:
            params.append(self.id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
            
            result = db.execute_query(query, params)
            db.close()
            
            if result:
                # Update instance variables
                if full_name is not None:
                    self.full_name = full_name
                if email is not None:
                    self.email = email
                return True, "Profil başarıyla güncellendi"
            else:
                return False, "Profil güncelleme hatası"
        
        db.close()
        return False, "Güncellenecek alan bulunamadı"

def init_database():
    """Initialize database tables"""
    db = Database()
    db.connect()
    
    # Create users table
    users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(200),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    
    # Create analysis_history table for storing user analysis results
    analysis_table = """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            filename VARCHAR(255),
            prediction_result JSONB,
            features JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    
    try:
        db.execute_query(users_table)
        db.execute_query(analysis_table)
        print("Database tables initialized successfully!")
        db.close()
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        db.close()
        return False