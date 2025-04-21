# main.py (Đã chỉnh sửa)
# -*- coding: utf-8 -*-
import sys
import os
import logging
import coloredlogs
import argparse  # <<< THÊM IMPORT NÀY

from Coach import Coach
# Không import Game và NNet cố định ở đây nữa
# from othello.OthelloGame import OthelloGame as Game
# from othello.pytorch.NNet import NNetWrapper as nn
from utils import * # Giả sử dotdict được định nghĩa trong utils

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Đổi thành DEBUG để xem nhiều thông tin hơn.

# --- Các tham số mặc định ---
# Sẽ được cập nhật/ghi đè bởi các tham số dòng lệnh qua argparse
DEFAULT_ARGS = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder': '/dev/models/8x100x50', # Đường dẫn thư mục mẫu
    'load_file': 'best.pth.tar',          # Tên file mẫu

    'numItersForTrainExamplesHistory': 20,

    # Thêm các tham số có thể muốn điều chỉnh từ dòng lệnh
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available() if 'torch' in sys.modules else False, # Kiểm tra nếu torch đã import
    'num_channels': 128, # Nên khớp với NNet.py hoặc cho phép override
    'num_res_blocks': 10 # Nên khớp với NNet.py hoặc cho phép override
})

def main():
    # --- Xử lý tham số dòng lệnh ---
    parser = argparse.ArgumentParser(description='AlphaZero Main Trainer')
    # Thêm các argument, giá trị default lấy từ DEFAULT_ARGS
    parser.add_argument('--game', '-g', type=str, default=DEFAULT_ARGS.game if 'game' in DEFAULT_ARGS else 'othello', help='Tên game (othello, tictactoe, chess, ...)')
    parser.add_argument('--numIters', '-i', type=int, default=DEFAULT_ARGS.numIters, help='Số vòng lặp huấn luyện (iterations)')
    parser.add_argument('--numEps', '-e', type=int, default=DEFAULT_ARGS.numEps, help='Số ván self-play mỗi vòng lặp')
    parser.add_argument('--tempThreshold', '-t', type=int, default=DEFAULT_ARGS.tempThreshold, help='Ngưỡng temp=0 cho MCTS')
    parser.add_argument('--updateThreshold', '-u', type=float, default=DEFAULT_ARGS.updateThreshold, help='Ngưỡng chấp nhận model mới qua Arena')
    parser.add_argument('--maxlenOfQueue', '-q', type=int, default=DEFAULT_ARGS.maxlenOfQueue, help='Độ dài tối đa hàng đợi dữ liệu huấn luyện')
    parser.add_argument('--numMCTSSims', '-m', type=int, default=DEFAULT_ARGS.numMCTSSims, help='Số mô phỏng MCTS mỗi nước đi')
    parser.add_argument('--arenaCompare', '-a', type=int, default=DEFAULT_ARGS.arenaCompare, help='Số ván đấu Arena')
    parser.add_argument('--cpuct', '-c', type=float, default=DEFAULT_ARGS.cpuct, help='Hằng số εξερεύνηση MCTS')
    parser.add_argument('--checkpoint', '-p', type=str, default=DEFAULT_ARGS.checkpoint, help='Thư mục lưu checkpoint')
    parser.add_argument('--load_model', action='store_true', help='Có load model đã lưu không?')
    parser.add_argument('--load_folder', type=str, default=DEFAULT_ARGS.load_folder, help='Thư mục chứa model để load')
    parser.add_argument('--load_file', type=str, default=DEFAULT_ARGS.load_file, help='Tên file model để load')
    parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=DEFAULT_ARGS.numItersForTrainExamplesHistory, help='Số vòng lặp giữ lại dữ liệu huấn luyện')
    # Thêm các args khác nếu cần (ví dụ: lr, dropout, epochs, batch_size...)
    parser.add_argument('--lr', type=float, default=DEFAULT_ARGS.lr, help='Tốc độ học')
    parser.add_argument('--dropout', type=float, default=DEFAULT_ARGS.dropout, help='Tỷ lệ dropout')
    parser.add_argument('--epochs', type=int, default=DEFAULT_ARGS.epochs, help='Số epochs huấn luyện NNet mỗi vòng lặp')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_ARGS.batch_size, help='Kích thước batch huấn luyện NNet')
    parser.add_argument('--num_channels', type=int, default=DEFAULT_ARGS.num_channels, help='Số kênh Conv trong NNet')
    parser.add_argument('--num_res_blocks', type=int, default=DEFAULT_ARGS.num_res_blocks, help='Số ResBlocks trong NNet')

    # Parse các tham số từ dòng lệnh
    args_parsed = parser.parse_args()

    # Tạo đối tượng args cuối cùng (kết hợp default và parsed)
    # Dùng dotdict để truy cập bằng dấu chấm (args.numIters)
    args = dotdict({**DEFAULT_ARGS, **vars(args_parsed)})
    # Cập nhật lại tuple load_folder_file từ các args đã parse
    args.load_folder_file = (args.load_folder, args.load_file)
    # Cập nhật cuda status sau khi args có thể đã thay đổi
    args.cuda = torch.cuda.is_available() if 'torch' in sys.modules else False

    # --- Import và Khởi tạo Game, NNet động ---
    log.info(f"Loading game '{args.game}'...")
    if args.game == 'othello':
        from othello.OthelloGame import OthelloGame as Game
        # Giả sử dùng pytorch backend cho Othello
        from othello.pytorch.NNet import NNetWrapper as nn
        g = Game(6) # Othello cần kích thước bàn cờ
    elif args.game == 'chess':
        # <<< Phần dành cho Chess >>>
        from chess_game.ChessGame import ChessGame as Game         # Import lớp Game của bạn
        from chess_game.pytorch.NNet import NNetWrapper as nn # Import lớp NNetWrapper của bạn
        g = Game()                                           # Khởi tạo ChessGame (không cần tham số)
    else:
         raise ValueError(f"Invalid game specified: {args.game}")

    log.info('Loading %s...', nn.__name__)
    # Khởi tạo NNetWrapper, truyền vào game 'g' và các tham số 'args'
    # *** Đảm bảo NNetWrapper.__init__ trong NNet.py nhận tham số args ***
    nnet = nn(g, args) # <<< TRUYỀN args VÀO NNetWrapper

    # --- Phần còn lại giữ nguyên ---
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        try:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        except FileNotFoundError:
            log.error(f"Checkpoint không tìm thấy tại '{os.path.join(args.load_folder_file[0], args.load_folder_file[1])}'. Bắt đầu mà không load model.")
            args.load_model = False # Không thử load examples nếu model không load được
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    # Truyền các tham số đã cập nhật vào Coach
    c = Coach(g, nnet, args)

    if args.load_model: # Chỉ load examples nếu model đã được load
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    # Import torch ở đây để kiểm tra cuda sớm hơn nếu cần
    # Hoặc để kiểm tra cuda trong hàm main
    try:
        import torch
    except ImportError:
        print("PyTorch not found. Please install PyTorch.")
        sys.exit(1)

    main()