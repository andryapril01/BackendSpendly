# check_users.py - Script untuk mengecek user yang terdaftar

import os
import sys
from dotenv import load_dotenv
from flask import Flask
from models import db, User, Category, Transaction
from config import config
from datetime import datetime

# Load environment variables
load_dotenv()

def create_app(config_name='development'):
    """Create Flask application with database setup"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    
    return app

def check_all_users():
    """Tampilkan semua user yang terdaftar"""
    app = create_app()
    
    with app.app_context():
        try:
            print("=" * 60)
            print("DAFTAR SEMUA USER YANG TERDAFTAR")
            print("=" * 60)
            
            users = User.query.all()
            
            if not users:
                print("❌ Tidak ada user yang terdaftar dalam database")
                return
            
            print(f"✅ Total user terdaftar: {len(users)}")
            print("\nDetail User:")
            print("-" * 60)
            
            for i, user in enumerate(users, 1):
                print(f"\n{i}. USER ID: {user.id}")
                print(f"   Email: {user.email}")
                print(f"   Nama: {user.first_name} {user.last_name}")
                print(f"   Telepon: {user.phone or 'Tidak ada'}")
                print(f"   Status: {'✅ Aktif' if user.is_active else '❌ Tidak Aktif'}")
                print(f"   Verifikasi: {'✅ Terverifikasi' if user.is_verified else '❌ Belum Verifikasi'}")
                print(f"   Terdaftar: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Cek kategori user
                categories = Category.query.filter_by(user_id=user.id).count()
                print(f"   Kategori: {categories} kategori")
                
                # Cek transaksi user
                transactions = Transaction.query.filter_by(user_id=user.id).count()
                total_expense = db.session.query(db.func.sum(Transaction.amount)).filter(
                    Transaction.user_id == user.id,
                    Transaction.amount > 0
                ).scalar() or 0
                print(f"   Transaksi: {transactions} transaksi (Total: Rp {total_expense:,.0f})")
                
                print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def check_recent_registrations():
    """Tampilkan pendaftaran terbaru (24 jam terakhir)"""
    app = create_app()
    
    with app.app_context():
        try:
            print("\n" + "=" * 60)
            print("PENDAFTARAN TERBARU (24 JAM TERAKHIR)")
            print("=" * 60)
            
            # Ambil user yang daftar dalam 24 jam terakhir
            yesterday = datetime.now() - timedelta(days=1)
            recent_users = User.query.filter(User.created_at >= yesterday).order_by(User.created_at.desc()).all()
            
            if not recent_users:
                print("❌ Tidak ada pendaftaran baru dalam 24 jam terakhir")
                return
            
            print(f"✅ {len(recent_users)} pendaftaran baru:")
            
            for user in recent_users:
                time_ago = datetime.now() - user.created_at
                hours_ago = int(time_ago.total_seconds() / 3600)
                minutes_ago = int((time_ago.total_seconds() % 3600) / 60)
                
                print(f"\n📝 {user.first_name} {user.last_name}")
                print(f"   Email: {user.email}")
                print(f"   Waktu: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   ({hours_ago} jam {minutes_ago} menit yang lalu)")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def search_user_by_email(email):
    """Cari user berdasarkan email"""
    app = create_app()
    
    with app.app_context():
        try:
            user = User.query.filter_by(email=email).first()
            
            if not user:
                print(f"❌ User dengan email '{email}' tidak ditemukan")
                return
            
            print(f"\n✅ User ditemukan:")
            print(f"ID: {user.id}")
            print(f"Email: {user.email}")
            print(f"Nama: {user.first_name} {user.last_name}")
            print(f"Telepon: {user.phone or 'Tidak ada'}")
            print(f"Status: {'Aktif' if user.is_active else 'Tidak Aktif'}")
            print(f"Verifikasi: {'Terverifikasi' if user.is_verified else 'Belum Verifikasi'}")
            print(f"Terdaftar: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Cek data terkait
            categories = Category.query.filter_by(user_id=user.id).count()
            transactions = Transaction.query.filter_by(user_id=user.id).count()
            print(f"Kategori: {categories}")
            print(f"Transaksi: {transactions}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    import argparse
    from datetime import timedelta
    
    parser = argparse.ArgumentParser(description='Check users in Spendly database')
    parser.add_argument('--all', action='store_true', help='Show all registered users')
    parser.add_argument('--recent', action='store_true', help='Show recent registrations (last 24h)')
    parser.add_argument('--email', type=str, help='Search user by email')
    
    args = parser.parse_args()
    
    if args.email:
        search_user_by_email(args.email)
    elif args.recent:
        check_recent_registrations()
    elif args.all:
        check_all_users()
    else:
        # Default: tampilkan semua
        check_all_users()
        check_recent_registrations()