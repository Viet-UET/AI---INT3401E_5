#!/usr/bin/env python3
"""
Script để huấn luyện mô hình NNUE từ dữ liệu PGN.
Có thể chạy như sau:
    python train_nnue.py --pgn path/to/games.pgn --epochs 5 --batch-size 32
"""

import os
import argparse
from engine.NNUE import NNUE
from engine.PGNProcessor import PGNProcessor

# Thư mục mặc định để lưu các file PGN và mô hình
DEFAULT_DATA_DIR = "./data"
DEFAULT_MODELS_DIR = "./data/models"
DEFAULT_PGN_DIR = "./data/pgn"
DEFAULT_MODEL_PATH = "./data/models/nnue_model.pkl"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
os.makedirs(DEFAULT_PGN_DIR, exist_ok=True)

def train_from_pgn(pgn_path, model_path, epochs=5, batch_size=32, learning_rate=0.001, max_games=100):
    """Huấn luyện mô hình NNUE từ file PGN"""
    # Kiểm tra xem file PGN tồn tại chưa
    if not os.path.exists(pgn_path):
        print(f"File PGN {pgn_path} không tồn tại!")
        return False
    
    # Tạo model NNUE mới hoặc tải model có sẵn
    if os.path.exists(model_path):
        print(f"Tải model đã có từ {model_path}...")
        nnue = NNUE(model_path)
    else:
        print(f"Tạo model mới...")
        nnue = NNUE()
    
    # Tạo PGN processor với model NNUE
    pgn_processor = PGNProcessor(nnue)
    
    # Huấn luyện từ file PGN
    success = pgn_processor.train_nnue_from_pgn(
        pgn_path, model_path, 
        epochs=epochs, 
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_games=max_games
    )
    
    if success:
        print(f"Đã huấn luyện và lưu model thành công vào {model_path}")
    else:
        print(f"Huấn luyện model thất bại")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình NNUE từ dữ liệu PGN")
    
    # Tham số cho đường dẫn file PGN
    parser.add_argument("--pgn", type=str, required=True, help="Đường dẫn tới file PGN")
    
    # Tham số cho việc huấn luyện
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, 
                        help=f"Đường dẫn lưu model (mặc định: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Số epochs để huấn luyện (mặc định: 5)")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Kích thước batch (mặc định: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate (mặc định: 0.001)")
    parser.add_argument("--max-games", type=int, default=100, 
                        help="Số lượng ván cờ tối đa để xử lý (mặc định: 100)")
    
    args = parser.parse_args()
    
    # Huấn luyện từ file PGN
    train_from_pgn(
        pgn_path=args.pgn,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_games=args.max_games
    )

if __name__ == "__main__":
    main() 