# chess/pytorch/NNet.py
# -*- coding: utf-8 -*-

import os
import sys
import time

import numpy as np
# Thêm thư mục gốc của dự án vào sys.path để import NeuralNet
# Đi lên 2 cấp từ chess/pytorch/ -> chess/ -> alpha-zero-general/
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from NeuralNet import NeuralNet # Import lớp NeuralNet cơ sở

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Cấu hình cơ bản ---
# TODO: Nên đưa các tham số này vào file config hoặc argument khi chạy
args = {
    'lr': 0.001,         # Tốc độ học (Learning rate)
    'dropout': 0.3,      # Tỷ lệ dropout
    'epochs': 10,        # Số epochs huấn luyện mỗi lần lặp
    'batch_size': 64,    # Kích thước batch
    'cuda': torch.cuda.is_available(), # Tự động kiểm tra có GPU CUDA không
    'num_channels': 128, # Số lượng kênh (filters) trong các lớp convolutional (có thể tăng lên 256 nếu tài nguyên mạnh)
    'num_res_blocks': 10 # Số block Residual (có thể tăng lên 19-20 như AlphaZero gốc nếu tài nguyên mạnh)
}


# Lớp định nghĩa kiến trúc mạng nơ-ron thực tế (kế thừa nn.Module của PyTorch)
class ChessNNet(nn.Module):
    def __init__(self, game, args):
        super(ChessNNet, self).__init__()
        # Lấy thông tin từ đối tượng game
        self.board_x, self.board_y = game.getBoardSize() # 8x8
        self.action_size = game.getActionSize() # ~4544
        self.args = args
        # Số lượng input planes phải khớp với output của getBoardRep
        self.num_input_planes = 18 # *** Đảm bảo con số này khớp với getBoardRep ***

        # --- Định nghĩa các lớp của mạng ---

        # Lớp Convolutional đầu tiên
        self.conv1 = nn.Conv2d(self.num_input_planes, args['num_channels'], kernel_size=3, stride=1, padding=1, bias=False) # Thêm bias=False nếu dùng BN
        self.bn1 = nn.BatchNorm2d(args['num_channels'])

        # Chuỗi các Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(args['num_channels'], args['dropout']) for _ in range(args['num_res_blocks'])])

        # Đầu ra Policy Head
        self.pi_conv = nn.Conv2d(args['num_channels'], 32, kernel_size=1, stride=1, bias=False) # Giảm số kênh trước FC layer
        self.pi_bn = nn.BatchNorm2d(32)
        # Kích thước input cho lớp FC = số kênh * chiều rộng * chiều cao
        self.pi_fc = nn.Linear(32 * self.board_x * self.board_y, self.action_size)

        # Đầu ra Value Head
        self.v_conv = nn.Conv2d(args['num_channels'], 1, kernel_size=1, stride=1, bias=False) # Giảm về 1 kênh
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * self.board_x * self.board_y, 256) # Lớp FC trung gian
        self.v_fc2 = nn.Linear(256, 1) # Lớp FC cuối cùng cho ra 1 giá trị

    def forward(self, s):
        # Input s có shape: batch_size x num_planes x board_x x board_y (Channels-Height-Width)

        # Lớp conv đầu tiên
        s = F.relu(self.bn1(self.conv1(s)))

        # Qua các ResBlocks
        for block in self.res_blocks:
            s = block(s)

        # --- Policy Head ---
        pi = F.relu(self.pi_bn(self.pi_conv(s)))
        # Flatten output của lớp conv cuối cùng của policy head
        pi = pi.reshape(pi.size(0), -1) # Giữ nguyên batch_size, flatten các chiều còn lại
        pi = self.pi_fc(pi)
        # Trả về log probabilities (log softmax) - phù hợp với loss function NLLLoss hoặc CrossEntropy
        pi_out = F.log_softmax(pi, dim=1)

        # --- Value Head ---
        v = F.relu(self.v_bn(self.v_conv(s)))
        # Flatten output của lớp conv cuối cùng của value head
        v = v.reshape(v.size(0), -1) # Giữ nguyên batch_size, flatten các chiều còn lại
        v = F.relu(self.v_fc1(v))
        # Qua lớp FC cuối cùng và hàm kích hoạt Tanh để giá trị nằm trong khoảng [-1, 1]
        v_out = torch.tanh(self.v_fc2(v))

        return pi_out, v_out


# Lớp Residual Block (khối dư) - kiến trúc phổ biến giúp huấn luyện mạng sâu
class ResBlock(nn.Module):
    def __init__(self, num_channels, dropout):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out) # Áp dụng dropout
        out = self.bn2(self.conv2(out))
        out += residual # Skip connection (cộng với input ban đầu)
        out = F.relu(out) # Áp dụng ReLU sau khi cộng skip connection
        return out


# Lớp Wrapper - đóng gói mô hình PyTorch để tương thích với interface NeuralNet của framework
class NNetWrapper(NeuralNet):
    def __init__(self, game, args_from_main):
        super().__init__(game)
        # Khởi tạo mô hình PyTorch thực tế
        self.args = args_from_main
        self.nnet = ChessNNet(game, self.args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        # Lấy lại số input planes từ game hoặc đảm bảo nó khớp với ChessNNet
        self.num_input_planes = self.nnet.num_input_planes

        # Chuyển mô hình lên GPU nếu có CUDA
        if self.args['cuda']: 
            self.nnet.cuda()

    def train(self, examples):
        """
        Huấn luyện mạng nơ-ron với tập dữ liệu examples.
        examples: list các tuple (board_rep, policy_target, value_target)
                  board_rep là output của game.getBoardRep (shape 8,8,18)
                  policy_target là vector output của MCTS (size action_size)
                  value_target là kết quả cuối cùng của ván đấu (-1, 0, 1)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr']) # Dùng Adam optimizer

        for epoch in range(self.args['epochs']):
            print(f'EPOCH ::: {epoch+1}')
            self.nnet.train() # Đặt mạng ở chế độ huấn luyện (quan trọng cho Dropout, BN)
            pi_losses = [] # Lưu policy loss của từng batch
            v_losses = []  # Lưu value loss của từng batch

            # Xáo trộn dữ liệu huấn luyện
            np.random.shuffle(examples)

            # Chia dữ liệu thành các batch
            num_batches = int(len(examples) / self.args['batch_size'])

            for i in range(num_batches):
                start_idx = i * self.args['batch_size']
                end_idx = start_idx + self.args['batch_size']
                batch = examples[start_idx:end_idx]

                # Tách dữ liệu từ batch
                boards, target_pis, target_vs = list(zip(*batch))

                # --- Chuẩn bị dữ liệu cho PyTorch ---
                # Chuyển board representation từ (Batch, H, W, Channels) sang (Batch, Channels, H, W)
                boards = np.array(boards)
                boards = boards.transpose(0, 3, 1, 2)
                boards = torch.FloatTensor(boards)

                target_pis = torch.FloatTensor(np.array(target_pis))
                # Đảm bảo target_vs là FloatTensor và có shape đúng (batch_size,)
                target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float64))

                # Chuyển dữ liệu lên GPU nếu dùng CUDA
                if self.args['cuda']:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # --- Tính toán ---
                # Đưa input qua mạng
                out_pi, out_v = self.nnet(boards)

                # Tính loss
                l_pi = self.loss_pi(target_pis, out_pi) # Policy loss
                l_v = self.loss_v(target_vs, out_v)     # Value loss
                total_loss = l_pi + l_v                  # Tổng loss

                # Lưu loss
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())

                # --- Cập nhật trọng số ---
                optimizer.zero_grad()   # Reset gradient
                total_loss.backward() # Lan truyền ngược lỗi
                optimizer.step()        # Cập nhật trọng số

            # In loss trung bình của epoch
            print(f"Epoch {epoch+1} finished - Avg Policy Loss: {np.mean(pi_losses):.4f}, Avg Value Loss: {np.mean(v_losses):.4f}")


    def predict(self, board_rep):
        """
        Dự đoán policy và value cho một trạng thái bàn cờ.
        Input: board_rep (numpy array shape 8,8,18 - output của getBoardRep)
        Output: (policy, value)
                 policy: numpy array (size action_size) chứa xác suất các nước đi
                 value: float scalar đánh giá trạng thái bàn cờ
        """
        # --- Chuẩn bị input cho PyTorch ---
        # Chuyển từ (H, W, C) sang (C, H, W)
        board_rep = board_rep.transpose(2, 0, 1)
        # Thêm chiều batch (batch_size=1)
        board_rep = torch.FloatTensor(board_rep).unsqueeze(0)

        # Chuyển lên GPU nếu dùng CUDA
        if self.args['cuda']:
            board_rep = board_rep.contiguous().cuda()

        # Đặt mạng ở chế độ đánh giá (evaluation mode - quan trọng cho Dropout, BN)
        self.nnet.eval()
        # Tắt tính toán gradient vì đang dự đoán
        with torch.no_grad():
            pi, v = self.nnet(board_rep)

        # Chuyển kết quả về CPU và dạng numpy
        # Lấy exponent của log_softmax để ra xác suất policy
        policy = torch.exp(pi).data.cpu().numpy()[0]
        value = v.data.cpu().numpy()[0][0] # Lấy giá trị scalar từ tensor shape [1,1]

        return policy, value

    def loss_pi(self, targets, outputs):
        """
        Hàm tính Policy loss.
        targets: policy vector từ MCTS (dạng probabilities)
        outputs: log probabilities từ mạng (output của log_softmax)
        Sử dụng Negative Log Likelihood Loss (tương đương CrossEntropy khi output là log_softmax)
        """
        # Mean over the batch
        return -torch.sum(targets * outputs) / targets.size(0)

    def loss_v(self, targets, outputs):
        """
        Hàm tính Value loss.
        targets: kết quả thực tế của ván đấu (-1, 0 hoặc 1)
        outputs: giá trị dự đoán từ mạng (output của tanh, trong khoảng -1 đến 1)
        Sử dụng Mean Squared Error loss.
        """
        # Mean Squared Error loss, đảm bảo outputs có shape (batch_size,) giống targets
        return torch.mean((targets - outputs.view(-1))**2)


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """ Lưu trạng thái mạng (trọng số) vào file checkpoint. """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.makedirs(folder) # Dùng makedirs để tạo cả thư mục cha nếu cần
        else:
            print(f"Checkpoint Directory exists: {folder}")
        # Lưu state_dict của mô hình PyTorch
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)
        print(f"Checkpoint saved to '{filepath}'")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """ Tải trọng số mạng từ file checkpoint. """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
        # Xác định load lên CPU hay GPU
        map_location = torch.device('cuda') if self.args['cuda'] else torch.device('cpu')
        checkpoint = torch.load(filepath, map_location=map_location)
        # Tải trọng số vào mô hình hiện tại
        self.nnet.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded from '{filepath}' (on {map_location})")

# Kết thúc file NNet.py