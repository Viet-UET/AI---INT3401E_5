import numpy as np
import pickle
import os
from engine.ChessEngine import GameState

# Kích thước feature vector và các tham số model
INPUT_SIZE = 768  # 12 loại quân * 64 ô
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1

class NNUE:
    def __init__(self, model_path=None):
        # Khởi tạo weights hoặc load từ file
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Khởi tạo weights với giá trị nhỏ ngẫu nhiên
            self.fc1_weights = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01
            self.fc1_bias = np.zeros(HIDDEN_SIZE)
            self.fc2_weights = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01
            self.fc2_bias = np.zeros(OUTPUT_SIZE)
            
        # Cache cho feature vector và layer trung gian
        self.input_features = None
        self.hidden_output = None
        
    def load_model(self, path):
        """Load model weights từ file"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                self.fc1_weights = model_data['fc1_weights']
                self.fc1_bias = model_data['fc1_bias']
                self.fc2_weights = model_data['fc2_weights']
                self.fc2_bias = model_data['fc2_bias']
            print(f"Đã tải model NNUE từ {path}")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            # Khởi tạo weights mặc định
            self.fc1_weights = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.01
            self.fc1_bias = np.zeros(HIDDEN_SIZE)
            self.fc2_weights = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.01
            self.fc2_bias = np.zeros(OUTPUT_SIZE)
    
    def save_model(self, path):
        """Lưu model weights xuống file"""
        model_data = {
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Đã lưu model NNUE vào {path}")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")
            return False
    
    def board_to_features(self, board):
        """Chuyển đổi bàn cờ thành vector đặc trưng
        Mỗi loại quân tạo một one-hot vector cho 64 ô"""
        features = np.zeros(INPUT_SIZE)
        
        # Các loại quân
        piece_types = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 
                        'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
        
        # Duyệt qua từng ô trên bàn cờ
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != '--':
                    # Tìm index của loại quân trong danh sách
                    piece_idx = piece_types.index(piece)
                    # Tính vị trí trong vector đặc trưng
                    feature_idx = piece_idx * 64 + row * 8 + col
                    features[feature_idx] = 1
        
        return features
    
    def relu(self, x):
        """Hàm kích hoạt ReLU"""
        return np.maximum(0, x)
    
    def forward(self, features):
        """Forward pass qua neural network"""
        # Layer 1
        hidden = np.dot(features, self.fc1_weights) + self.fc1_bias
        hidden_output = self.relu(hidden)
        
        # Layer 2 (output)
        output = np.dot(hidden_output, self.fc2_weights) + self.fc2_bias
        
        # Cache lại để dùng trong backward pass
        self.input_features = features
        self.hidden_output = hidden_output
        
        return output[0]  # Trả về giá trị đánh giá (scalar)
    
    def evaluate(self, gamestate, perspective='w'):
        """Đánh giá vị trí hiện tại của bàn cờ qua mạng NNUE"""
        features = self.board_to_features(gamestate.board)
        
        # Lưu ý: Output của mạng là từ góc nhìn trắng
        # Nếu đang đánh giá cho đen, cần đảo dấu
        value = self.forward(features)
        if perspective == 'b':
            value = -value
            
        return value
    
    def train_step(self, inputs, targets, learning_rate=0.01):
        """Thực hiện một bước huấn luyện với backpropagation"""
        batch_size = len(inputs)
        total_loss = 0
        
        for i in range(batch_size):
            try:
                # Forward pass
                features = inputs[i]
                target = targets[i]
                prediction = self.forward(features)
                
                # Tính loss (MSE)
                loss = (prediction - target) ** 2
                total_loss += loss
                
                # Backward pass
                # Gradient cho output layer
                d_pred = 2 * (prediction - target)
                
                # Gradient cho fc2
                d_fc2_weights = np.outer(self.hidden_output, d_pred)
                d_fc2_bias = d_pred
                
                # Gradient cho hidden layer
                d_hidden = np.dot(d_pred, self.fc2_weights.T)
                
                # Áp dụng ReLU gradient một cách an toàn (sửa lỗi kích thước)
                relu_mask = (self.hidden_output > 0).flatten()
                d_hidden = d_hidden.flatten() * relu_mask
                
                # Gradient cho fc1
                d_fc1_weights = np.outer(self.input_features, d_hidden)
                d_fc1_bias = d_hidden
                
                # Cập nhật weights (gradient descent)
                self.fc2_weights -= learning_rate * d_fc2_weights
                self.fc2_bias -= learning_rate * d_fc2_bias
                self.fc1_weights -= learning_rate * d_fc1_weights.reshape(self.fc1_weights.shape)
                self.fc1_bias -= learning_rate * d_fc1_bias
            except Exception as e:
                print(f"Lỗi khi huấn luyện mẫu thứ {i}: {e}")
                continue
            
        return total_loss / batch_size 