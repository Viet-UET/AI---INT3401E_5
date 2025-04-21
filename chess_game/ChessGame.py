# chess/ChessGame.py
# -*- coding: utf-8 -*-

import sys
import os

# --- Bắt đầu sửa lỗi import ---
# Lấy đường dẫn tuyệt đối đến thư mục chứa file script này (ví dụ: c:\alpha-zero-general\chess)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn tuyệt đối đến thư mục cha (ví dụ: c:\alpha-zero-general)
parent_dir = os.path.dirname(script_dir)
# Thêm đường dẫn tuyệt đối của thư mục cha vào sys.path
# để Python có thể tìm thấy module Game khi import
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Kết thúc sửa lỗi import ---

from Game import Game # Import lớp Game cơ sở từ thư mục cha
import chess         # Import thư viện python-chess (đã sửa từ 'python_chess')
import numpy as np



class ChessGame(Game):
    """
    Lớp cài đặt cụ thể cho trò chơi Cờ Vua, kế thừa từ lớp Game cơ sở
    trong khung sườn AlphaZero General.
    """
    def __init__(self):
        """
        Khởi tạo trò chơi, bao gồm việc tạo action map.
        """
        self.action_map = self._create_action_map()
        # Tạo map ngược từ index -> uci string để dùng trong getNextState
        self.inv_action_map = {v: k for k, v in self.action_map.items()}
        # In ra kích thước không gian hành động để kiểm tra
        print(f"Initialized ChessGame with action space size: {len(self.action_map)}")

    def _create_action_map(self):
        """
        Tạo map từ chuỗi UCI của nước đi sang action index (0 -> N-1).
        Bao gồm nước đi thường và tất cả các khả năng phong cấp
        cho các nước đi Tốt đến hàng cuối.
        Nhập thành được xử lý như nước đi Vua thông thường ('e1g1', 'e1c1', etc.).
        """
        action_map = {}
        idx = 0
        squares = range(64) # List of 64 square indices (0-63)
        promotions = [5, 4, 3, 2] # Sử dụng trực tiếp giá trị số nguyên của quân cờ (Q=5, R=4, B=3, N=2)
        # Duyệt qua tất cả các cặp ô đi/đến có thể
        for from_sq in squares:
            for to_sq in squares:
                if from_sq == to_sq: continue # Bỏ qua đi tại chỗ

                # 1. Map nước đi không phong cấp
                move = chess.Move(from_sq, to_sq)
                action_map[move.uci()] = idx
                idx += 1

                # 2. Map các nước đi có thể phong cấp
                # Chỉ tạo map nếu nước đi là đến hàng cuối (rank 7 cho Trắng, rank 0 cho Đen)
                from_rank = chess.square_rank(from_sq)
                to_rank = chess.square_rank(to_sq)
                if (from_rank == 6 and to_rank == 7) or \
                   (from_rank == 1 and to_rank == 0):
                    for promo_piece in promotions:
                        promo_move = chess.Move(from_sq, to_sq, promotion=promo_piece)
                        action_map[promo_move.uci()] = idx
                        idx += 1

        # Kích thước action space dự kiến: 64*63 + (8*8*4)*2 = 4032 + 512 = 4544
        return action_map

    def getInitBoard(self):
        """
        Trả về trạng thái bàn cờ ban đầu dưới dạng đối tượng chess.Board.
        """
        board = chess.Board()
        return board

    def getBoardSize(self):
        """
        Trả về kích thước bàn cờ dạng tuple (rows, cols).
        """
        return (8, 8)

    def getActionSize(self):
        """
        Trả về tổng số lượng hành động (nước đi) có thể có đã được map.
        Con số này phải khớp với kích thước output của Policy Head trong NNet.
        """
        return len(self.action_map)

    def getNextState(self, board, player, action_index):
        """
        Thực hiện hành động (dựa trên action_index) lên bàn cờ.
        Input:
            board: đối tượng chess.Board hiện tại
            player: người chơi hiện tại (1=White, -1=Black)
            action_index: chỉ số của hành động (từ 0 đến getActionSize()-1)
        Output:
            (next_board, next_player): tuple chứa bàn cờ mới và người chơi kế tiếp (-player)
        """
        try:
            # Lấy chuỗi UCI tương ứng với action_index từ map ngược
            uci_move = self.inv_action_map[action_index]
        except KeyError:
            # Trường hợp cực kỳ hiếm nếu MCTS/NN đưa ra index không hợp lệ
            print(f"!!! Lỗi getNextState: action_index {action_index} không tồn tại trong inv_action_map.")
            # Có thể raise lỗi hoặc trả về trạng thái cũ để xử lý ở tầng trên
            # Tạm thời trả về trạng thái cũ và đổi lượt
            return (board.copy(), -player)

        try:
            # Chuyển chuỗi UCI thành đối tượng Move trong ngữ cảnh bàn cờ hiện tại
            # board.parse_uci sẽ kiểm tra tính hợp lệ cơ bản và xử lý nhập thành
            move = board.parse_uci(uci_move)
        except ValueError:
            # Lỗi nếu uci_move không hợp lệ về cú pháp hoặc không thể thực hiện
            # (VD: cố nhập thành khi không được phép dù index đúng)
            # Điều này không nên xảy ra nếu MCTS chọn đúng từ valid_moves
            print(f"!!! Lỗi getNextState: Không thể parse hoặc thực hiện UCI '{uci_move}' (index {action_index}) cho bàn cờ:\n{board}")
            # Trả về trạng thái cũ và đổi lượt
            return (board.copy(), -player)

        # Tạo bản copy của bàn cờ để không thay đổi trạng thái gốc
        next_board = board.copy()
        # Thực hiện nước đi
        next_board.push(move)
        # Đổi lượt chơi
        next_player = -player
        return (next_board, next_player)

    def getValidMoves(self, board, player):
        """
        Trả về một vector numpy nhị phân (0 hoặc 1) biểu thị các nước đi hợp lệ.
        Vector này có kích thước bằng getActionSize().
        Giá trị 1.0 tại index i nghĩa là nước đi tương ứng với index i là hợp lệ.
        """
        # Khởi tạo vector toàn số 0
        valids = np.zeros(self.getActionSize(), dtype=np.float32)
        # Lấy danh sách các nước đi hợp lệ từ trạng thái bàn cờ hiện tại
        legal_moves = board.legal_moves
        # Duyệt qua từng nước đi hợp lệ
        for move in legal_moves:
            try:
                # Lấy index tương ứng với nước đi hợp lệ từ action_map
                action_index = self.action_map[move.uci()]
                # Đặt giá trị 1.0 tại index đó trong vector valids
                valids[action_index] = 1.0
            except KeyError:
                # Cảnh báo nếu một nước đi hợp lệ không có trong map đã tạo
                # Điều này chỉ ra hàm _create_action_map có thể chưa đủ hoặc sai
                # print(f"Cảnh báo: Nước đi hợp lệ {move.uci()} không tìm thấy trong action_map!")
                pass # Tạm thời bỏ qua nếu không tìm thấy
        return valids

    def getGameEnded(self, board, player):
        """
        Kiểm tra xem ván cờ đã kết thúc chưa và trả về kết quả theo góc nhìn của 'player'.
        Output:
            1 nếu 'player' thắng
           -1 nếu 'player' thua
           1e-4 (số rất nhỏ, dùng để phân biệt với 0) nếu hòa
            0 nếu ván cờ chưa kết thúc
        """
        if board.is_checkmate(): # Kiểm tra chiếu bí
            # Nếu lượt đi hiện tại *không phải* của player, nghĩa là đối phương bị chiếu bí -> player thắng
            if board.turn != (player == 1): # player=1 là Trắng (turn=True), player=-1 là Đen (turn=False)
                return 1
            else: # Ngược lại, player bị chiếu bí -> player thua
                return -1
        elif board.is_stalemate() or \
             board.is_insufficient_material() or \
             board.is_seventyfive_moves() or \
             board.can_claim_draw(): # Kiểm tra các luật hòa cờ khác
            return 1e-4 # Trả về giá trị nhỏ cho trường hợp hòa
        else:
            return 0 # Ván cờ chưa kết thúc

    def getCanonicalForm(self, board, player):
        """
        Trả về dạng "chuẩn tắc" của bàn cờ theo góc nhìn của player.
        Trong nhiều trường hợp, việc xoay bàn cờ trong getBoardRep đã đủ.
        Hàm này có thể chỉ cần trả về board gốc.
        """
        # Trả về board gốc, việc xử lý góc nhìn đã làm trong getBoardRep
        return board

    def stringRepresentation(self, board):
        """
        Trả về một biểu diễn dạng chuỗi *duy nhất* cho trạng thái bàn cờ.
        Dùng để làm key trong MCTS nhằm phát hiện các trạng thái đã gặp.
        FEN là lựa chọn tốt nhất.
        """
        return board.fen()

    def getScore(self, board, player):
        """
        (Tùy chọn) Trả về một điểm số heuristic cho trạng thái bàn cờ.
        Hàm này thường không cần thiết trong AlphaZero vì Value Head của NNet
        sẽ dự đoán giá trị (outcome) của ván đấu.
        Nhưng có thể hữu ích cho việc debug hoặc các thuật toán khác.
        """
        # Ví dụ đơn giản (chỉ mang tính minh họa):
        # score = 0
        # for sq in chess.SQUARES:
        #     piece = board.piece_at(sq)
        #     if piece:
        #         # Định nghĩa giá trị cho từng loại quân
        #         # ... (code tính điểm) ...
        # return score * player # Trả về điểm theo góc nhìn player
        pass # Bỏ qua trong cài đặt AlphaZero chuẩn

    def getSymmetries(self, board, pi):
        """
        (Tùy chọn) Tạo ra các trạng thái đối xứng của bàn cờ và policy tương ứng.
        Dùng để tăng cường dữ liệu huấn luyện (Data Augmentation).
        Với cờ vua, đối xứng duy nhất thường dùng là lật bàn cờ theo chiều dọc.
        Input:
            board: numpy array biểu diễn bàn cờ (output của getBoardRep)
            pi: policy vector (output của Policy Head)
        Output:
            List các tuple (symmetric_board, symmetric_pi)
        """
        # Cài đặt này yêu cầu hiểu rõ cách action map tương ứng với việc lật bàn cờ
        # Ví dụ: nước e2e4 sau khi lật sẽ thành e7e5... cần map lại index của policy vector.
        # Đây là phần nâng cao, có thể bỏ qua ban đầu.
        assert(len(pi) == self.getActionSize()) # Chính sách phải có kích thước đúng
        # Tạm thời chỉ trả về trạng thái gốc\
        # --- THAY ĐỔI QUAN TRỌNG ---
        # Chuyển đổi board object thành dạng biểu diễn số NGAY TẠI ĐÂY
        # Dùng player=1 làm góc nhìn chuẩn khi lưu ví dụ (khớp với cách MCTS tính pi)
        board_rep = self.getBoardRep(board, 1)
        return [(board_rep, pi)]

    def getBoardRep(self, board, player):
        """
        Chuyển đổi chess.Board thành numpy array (8, 8, num_planes) cho Neural Network.
        Đã được hoàn thiện ở bước trước.
        """
        num_planes = 18 # Phải khớp với input của mạng NNet sau này
        representation = np.zeros((8, 8, num_planes), dtype=np.float32)

        my_color = chess.WHITE if player == 1 else chess.BLACK
        opponent_color = not my_color

        piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        piece_map = {piece_type: i for i, piece_type in enumerate(piece_order)}

        for r in range(8):
            for c in range(8):
                sq = chess.square(file_index=c, rank_index=7 - r)
                piece = board.piece_at(sq)
                if piece:
                    plane_idx = piece_map[piece.piece_type]
                    if piece.color == my_color:
                        representation[r, c, plane_idx] = 1.0
                    else:
                        representation[r, c, plane_idx + 6] = 1.0

        if board.turn == my_color:
            representation[:, :, 12] = 1.0

        if board.has_kingside_castling_rights(chess.WHITE): representation[:, :, 13] = 1.0 # WK
        if board.has_queenside_castling_rights(chess.WHITE): representation[:, :, 14] = 1.0 # WQ
        if board.has_kingside_castling_rights(chess.BLACK): representation[:, :, 15] = 1.0 # BK
        if board.has_queenside_castling_rights(chess.BLACK): representation[:, :, 16] = 1.0 # BQ

        if board.ep_square:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            representation[7 - ep_rank, ep_file, 17] = 1.0

        if player == -1:
            representation = np.rot90(representation, k=2, axes=(0, 1))

        return representation

# Kết thúc lớp ChessGame