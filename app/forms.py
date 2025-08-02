"""
Forms for user authentication and profile management
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, EmailField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from app.database import User


class LoginForm(FlaskForm):
    username = StringField('Kullanıcı Adı', validators=[
        DataRequired(message='Kullanıcı adı gereklidir'),
        Length(min=3, max=80, message='Kullanıcı adı 3-80 karakter arasında olmalıdır')
    ])
    password = PasswordField('Şifre', validators=[
        DataRequired(message='Şifre gereklidir')
    ])
    submit = SubmitField('Giriş Yap')


class RegisterForm(FlaskForm):
    username = StringField('Kullanıcı Adı', validators=[
        DataRequired(message='Kullanıcı adı gereklidir'),
        Length(min=3, max=80, message='Kullanıcı adı 3-80 karakter arasında olmalıdır')
    ])
    email = EmailField('Email', validators=[
        DataRequired(message='Email gereklidir'),
        Email(message='Geçerli bir email adresi giriniz')
    ])
    full_name = StringField('Ad Soyad', validators=[
        Length(max=200, message='Ad soyad en fazla 200 karakter olabilir')
    ])
    password = PasswordField('Şifre', validators=[
        DataRequired(message='Şifre gereklidir'),
        Length(min=6, message='Şifre en az 6 karakter olmalıdır')
    ])
    password_confirm = PasswordField('Şifre Tekrar', validators=[
        DataRequired(message='Şifre tekrarı gereklidir'),
        EqualTo('password', message='Şifreler eşleşmiyor')
    ])
    submit = SubmitField('Kayıt Ol')

    def validate_username(self, username):
        user = User.get_user_by_username(username.data)
        if user:
            raise ValidationError('Bu kullanıcı adı zaten kullanılıyor. Lütfen farklı bir kullanıcı adı seçiniz.')

    def validate_email(self, email):
        # Check if email exists in database
        from app.database import Database
        db = Database()
        db.connect()
        existing_user = db.fetch_one("SELECT id FROM users WHERE email = %s", (email.data,))
        db.close()
        if existing_user:
            raise ValidationError('Bu email adresi zaten kullanılıyor. Lütfen farklı bir email adresi kullanınız.')


class ProfileForm(FlaskForm):
    username = StringField('Kullanıcı Adı', render_kw={'readonly': True})
    email = EmailField('Email', validators=[
        DataRequired(message='Email gereklidir'),
        Email(message='Geçerli bir email adresi giriniz')
    ])
    full_name = StringField('Ad Soyad', validators=[
        Length(max=200, message='Ad soyad en fazla 200 karakter olabilir')
    ])
    submit = SubmitField('Profili Güncelle')

    def __init__(self, original_email, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.original_email = original_email

    def validate_email(self, email):
        if email.data != self.original_email:
            # Check if new email exists in database
            from app.database import Database
            db = Database()
            db.connect()
            existing_user = db.fetch_one("SELECT id FROM users WHERE email = %s", (email.data,))
            db.close()
            if existing_user:
                raise ValidationError('Bu email adresi zaten kullanılıyor. Lütfen farklı bir email adresi kullanınız.')