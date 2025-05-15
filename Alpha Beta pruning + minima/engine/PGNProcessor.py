import re
import chess
import chess.pgn
import numpy as np
import io
import os
from engine.ChessEngine import GameState, Move
from engine.NNUE import NNUE

class PGNProcessor:
    def __init__(self, nnue_model=None):
        """Khởi tạo PGNProcessor với model NNUE để tạo dataset"""
        self.nnue = nnue_model if nnue_model else NNUE()
        
    def read_pgn_file(self, file_path):
        """Đọc tất cả các ván cờ từ file PGN"""
        games = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as pgn_file:
                pgn_text = pgn_file.read()
                
            # Tách các ván cờ
            pgn_parts = re.split(r'\n\n\[', pgn_text)
            for i, part in enumerate(pgn_parts):
                if i > 0:
                    part = '[' + part
                
                # Đọc ván cờ
                pgn = io.StringIO(part)
                try:
                    game = chess.pgn.read_game(pgn)
                    if game:
                        games.append(game)
                except Exception as e:
                    print(f"Lỗi khi đọc ván cờ: {e}")
                    continue
            
            print(f"Đã đọc {len(games)} ván cờ từ {file_path}")
            return games
        
        except Exception as e:
            print(f"Lỗi khi đọc file PGN {file_path}: {e}")
            return []
    
    def _get_game_result_score(self, game):
        """Lấy điểm của ván cờ (1 cho trắng thắng, 0 cho hòa, -1 cho đen thắng)"""
        result = game.headers.get("Result", "*")
        
        if result == "1-0":  # Trắng thắng
            return 1.0
        elif result == "0-1":  # Đen thắng
            return -1.0
        elif result == "1/2-1/2":  # Hòa
            return 0.0
        else:
            return None  # Kết quả không rõ ràng
    
    def _convert_position_and_result(self, game, max_positions=50):
        """Chuyển đổi các vị trí trong ván cờ thành features và nhãn"""
        result_score = self._get_game_result_score(game)
        if result_score is None:
            return [], []
        
        features = []
        labels = []
        
        # Khởi tạo GameState 
        gs = GameState()
        
        # Duyệt qua các nước đi
        board = chess.Board()
        for i, move in enumerate(game.mainline_moves()):
            # Lấy chọn mẫu ngẫu nhiên để tránh quá nhiều trạng thái
            if i % 3 != 0 and i < 15:
                board.push(move)
                continue
                
            # Lấy trạng thái hiện tại của bàn cờ
            fen = board.fen()
            current_board = self._fen_to_board(fen)
            
            # Tạo feature từ bàn cờ
            feature = self.nnue.board_to_features(current_board)
            
            # Tính label (kết quả ván cờ với một chút ngẫu nhiên để cải thiện khả năng học)
            # Dần dần giảm ảnh hưởng của kết quả cuối khi xa với kết quả
            move_number = i // 2 + 1  # Số nước đi tính từ 1
            total_moves = board.fullmove_number
            progress = move_number / max(total_moves, 20)  # Tỉ lệ tiến triển của ván cờ
            
            # Trọng số của stockfish eval và kết quả cuối
            weight_result = min(0.2 + 0.8 * progress, 1.0)  # Tăng dần từ 0.2 đến 1.0
            
            # Mặc định: sử dụng kết quả cuối
            label = result_score
            
            # Chuẩn hóa nhãn về khoảng [-1, 1]
            features.append(feature)
            labels.append(label)
            
            # Thực hiện nước đi trên bàn cờ
            board.push(move)
            
            # Giới hạn số lượng vị trí lấy từ mỗi ván
            if len(features) >= max_positions:
                break
                
        return features, labels
    
    def _fen_to_board(self, fen):
        """Chuyển đổi FEN thành bàn cờ dạng mảng 2D theo định dạng của ChessEngine"""
        board = []
        fen_parts = fen.split(' ')
        rows = fen_parts[0].split('/')
        
        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['--'] * int(char))
                else:
                    color = 'w' if char.isupper() else 'b'
                    piece_type = char.lower()
                    
                    # Chuyển đổi quy ước ký hiệu
                    if piece_type == 'p':
                        board_row.append(color + 'p')
                    elif piece_type == 'r':
                        board_row.append(color + 'R')
                    elif piece_type == 'n':
                        board_row.append(color + 'N')
                    elif piece_type == 'b':
                        board_row.append(color + 'B')
                    elif piece_type == 'q':
                        board_row.append(color + 'Q')
                    elif piece_type == 'k':
                        board_row.append(color + 'K')
            
            board.append(board_row)
        
        return board
    
    def prepare_training_data(self, pgn_file_path, max_games=100, max_positions=50):
        """Chuẩn bị dữ liệu huấn luyện từ file PGN"""
        all_features = []
        all_labels = []
        
        # Đọc các ván cờ từ file PGN
        games = self.read_pgn_file(pgn_file_path)
        
        # Giới hạn số lượng ván cờ để xử lý
        games = games[:max_games]
        
        print(f"Bắt đầu xử lý {len(games)} ván cờ...")
        for i, game in enumerate(games):
            if i % 10 == 0:
                print(f"Đang xử lý ván cờ {i+1}/{len(games)}...")
            
            features, labels = self._convert_position_and_result(game, max_positions)
            all_features.extend(features)
            all_labels.extend(labels)
        
        print(f"Đã chuẩn bị xong {len(all_features)} mẫu dữ liệu.")
        return all_features, all_labels
    
    def train_nnue_from_pgn(self, pgn_file_path, output_model_path, epochs=5, 
                           batch_size=32, learning_rate=0.001, max_games=100):
        """Huấn luyện mô hình NNUE từ file PGN và lưu xuống file"""
        # Chuẩn bị dữ liệu
        features, labels = self.prepare_training_data(pgn_file_path, max_games)
        
        if not features:
            print("Không có dữ liệu huấn luyện!")
            return False
            
        # Chuyển đổi thành mảng numpy
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Bắt đầu huấn luyện với {len(features)} mẫu, {epochs} epochs...")
        
        # Huấn luyện mô hình
        for epoch in range(epochs):
            # Trộn dữ liệu
            indices = np.random.permutation(len(features))
            
            # Chia thành các batch nhỏ
            total_loss = 0
            batches = 0
            
            for start_idx in range(0, len(features), batch_size):
                end_idx = min(start_idx + batch_size, len(features))
                batch_indices = indices[start_idx:end_idx]
                
                batch_features = [features[i] for i in batch_indices]
                batch_labels = [labels[i] for i in batch_indices]
                
                # Huấn luyện trên batch
                loss = self.nnue.train_step(batch_features, batch_labels, learning_rate)
                total_loss += loss
                batches += 1
                
                # In tiến trình
                if batches % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batches}, Loss: {loss:.6f}")
            
            avg_loss = total_loss / batches
            print(f"Epoch {epoch+1}/{epochs} hoàn thành, Loss trung bình: {avg_loss:.6f}")
            
            # Giảm learning rate sau mỗi epoch
            learning_rate *= 0.9
            
        # Lưu mô hình
        success = self.nnue.save_model(output_model_path)
        
        return success 