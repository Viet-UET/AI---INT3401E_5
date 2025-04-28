# Nội dung cho file: src/PlayAIMode.py

import pygame
import pygame_gui # Đảm bảo import
import math
import time
import random
from multiprocessing import Process, Queue # Hoặc threading

from src.GameInit import GameInit
from engine.AIEngine import AIEngine
# Import tất cả hằng số, bao gồm cả các hằng số mới
from src.config import *
from engine.ChessEngine import GameState, Move

# Định nghĩa thời gian giới hạn (tính bằng giây) - Có thể lấy từ config nếu muốn
AI_TIME_LIMIT = 10.0

class PlayAIMode(GameInit):

    def __init__(self, playerTurn='w'): # Cho phép chọn lượt cho người chơi
        # Khởi tạo GameInit
        super().__init__()

        # Giả định không dùng panel AI riêng, đặt kích thước màn hình cơ bản
        self.screen = pygame.display.set_mode((WIDTH_WINDOW, HEIGHT_WINDOW))
        # Tạo UIManager mới với kích thước đúng
        self.manager = pygame_gui.UIManager((WIDTH_WINDOW, HEIGHT_WINDOW), theme_path="./data/theme_custom.json")
        # Gọi lại hàm load GUI (đã đổi tên thành _loadGUI trong GameInit)
        self._loadGUI()
        # Cập nhật thông tin panel ban đầu
        self.editChessPanel()

        # Xác định lượt chơi
        self.playerTurn = playerTurn
        self.aiTurn = 'b' if playerTurn == 'w' else 'w'

        # Khởi tạo AI Engine cho lượt của máy
        self.aiEngine = AIEngine(self.aiTurn)

        # Biến trạng thái cho AI
        self.aiThinking = False
        self.aiMoveFinderProcess = None
        self.aiStartTime = 0
        self.return_queue = Queue() # Dùng chung cho các lần gọi process

        # Cập nhật validMoves ban đầu
        self.validMoves = self.gs.getValidMoves()


    def mainLoop(self):
        """Vòng lặp chính cho chế độ Người vs Máy."""
        while self.running:
            isPlayerTurn = (self.gs.turn == self.playerTurn)
            self.time_delta = self.clock.tick(MAX_FPS) / 1000.0
            aiMove = None           # Nước đi cuối cùng AI chọn
            made_ai_move = False    # Cờ báo AI đã xác định xong nước đi

            # --- Xử lý sự kiện (Input người chơi và hệ thống) ---
            self.__eventHandler(isPlayerTurn)

            # ---- Logic AI (Chỉ chạy khi đến lượt AI) ----
            if not self.gameOver and not isPlayerTurn:
                # Bắt đầu tìm kiếm chính nếu chưa chạy
                if not self.aiThinking and self.aiMoveFinderProcess is None:
                    print(f"--- Turn: AI ({self.aiTurn.upper()}) ---")
                    print(f"AI ({self.aiTurn}) is thinking (max {AI_TIME_LIMIT}s, depth {DEPTH})...")
                    self.aiThinking = True
                    self.aiStartTime = time.time()
                    self.return_queue = Queue() # Tạo queue mới
                    # Cập nhật validMoves ngay trước khi AI suy nghĩ
                    self.validMoves = self.gs.getValidMoves()
                    if not self.validMoves:
                        self.gameOver = True
                        print("No valid moves found before starting AI.")
                    else:
                        # Khởi chạy process tìm kiếm chính
                        self.aiMoveFinderProcess = Process(target=self.aiEngine.AlphaBetaPruning,
                                                       args=(self.gs, DEPTH, -math.inf, math.inf, self.return_queue))
                        self.aiMoveFinderProcess.start()

                # Nếu AI đang suy nghĩ (tiến trình chính đang chạy)
                elif self.aiThinking:
                    elapsedTime = time.time() - self.aiStartTime
                    process_alive = self.aiMoveFinderProcess.is_alive()

                    # 1. Kiểm tra xem tìm kiếm chính đã xong chưa
                    if not process_alive:
                        print(f"AI ({self.aiTurn}) main process finished. Getting result...")
                        try:
                            ai_result = self.return_queue.get_nowait()
                            aiMove = ai_result.bestMove
                            print(f"AI ({self.aiTurn}) finished thinking in {elapsedTime:.2f}s.")
                            made_ai_move = True # Đã có nước đi
                        except Exception as e:
                            print(f"DEBUG: Error getting AI ({self.aiTurn}) result from main queue: {e}")
                            aiMove = None
                            made_ai_move = True # Đánh dấu đã xử lý xong, sẽ fallback
                        # Dọn dẹp trạng thái
                        self.aiThinking = False
                        self.aiMoveFinderProcess = None

                    # 2. Kiểm tra xem tìm kiếm chính có bị timeout không
                    elif elapsedTime > AI_TIME_LIMIT:
                        print(f"AI ({self.aiTurn}) main search timed out ({elapsedTime:.2f}s)! Terminating.")
                        self.aiMoveFinderProcess.terminate()
                        self.aiMoveFinderProcess.join()
                        self.aiThinking = False
                        self.aiMoveFinderProcess = None

                        # ---- Bắt đầu tìm kiếm nông ----
                        print(f"--- Starting shallow search (depth {REDUCED_DEPTH}, max {SHORT_TIMEOUT}s) ---")
                        shallow_queue = Queue()
                        # Cập nhật lại validMoves
                        self.validMoves = self.gs.getValidMoves()
                        if not self.validMoves:
                             self.gameOver = True
                             print("No valid moves for shallow search.")
                             made_ai_move = True
                             aiMove = None
                        else:
                            shallow_process = Process(target=self.aiEngine.AlphaBetaPruning,
                                                      args=(self.gs, REDUCED_DEPTH, -math.inf, math.inf, shallow_queue))
                            shallow_process.start()
                            shallow_process.join(timeout=SHORT_TIMEOUT)

                            if shallow_process.is_alive():
                                print(f"AI ({self.aiTurn}) shallow search also timed out! Terminating.")
                                shallow_process.terminate()
                                shallow_process.join()
                                aiMove = None
                            else:
                                try:
                                    shallow_result = shallow_queue.get_nowait()
                                    aiMove = shallow_result.bestMove
                                    if aiMove:
                                        print(f"AI ({self.aiTurn}) using move from shallow search: {aiMove.getChessNotation()}")
                                    else:
                                        print(f"AI ({self.aiTurn}) shallow search finished but returned no move.")
                                except Exception as e:
                                    print(f"DEBUG: Error getting AI ({self.aiTurn}) result from shallow queue: {e}")
                                    aiMove = None

                            # ---- Fallback cuối cùng: Chọn Random ----
                            if aiMove is None and not self.gameOver:
                                print(f"AI ({self.aiTurn}) falling back to random move.")
                                if self.validMoves:
                                    aiMove = random.choice(self.validMoves)
                                    print(f"AI ({self.aiTurn}) chooses random move: {aiMove.getChessNotation()}")
                                else:
                                    print(f"AI ({self.aiTurn}) timed out, shallow failed, and no valid moves available.")
                                    self.gameOver = True

                        made_ai_move = True # Đánh dấu đã xác định xong nước đi (nông hoặc random)

            # ---- Thực hiện nước đi của AI (nếu đã xác định) ----
            if made_ai_move and aiMove is not None and not self.gameOver:
                self.gs.makeMove(aiMove)
                self.moveMade = True # Cờ để cập nhật UI và trạng thái game
                # Phát âm thanh
                if aiMove.capturedPiece != '--':
                    pygame.mixer.Sound.play(self.sound_capture)
                else:
                    pygame.mixer.Sound.play(self.sound_move)
            elif made_ai_move and aiMove is None and not self.gameOver:
                 # Lỗi logic nếu không có nước đi mà game chưa over
                 print("Error: No AI move determined, but game not over. Setting game over.")
                 self.gameOver = True

            # ---- Cập nhật trạng thái và vẽ (Sau cả lượt người và lượt AI) ----
            if self.moveMade:
                self.validMoves = self.gs.getValidMoves() # Cập nhật sau mỗi nước đi hợp lệ
                # print(f"Valid moves after move: {len(self.validMoves)}")
                if len(self.validMoves) == 0 and not self.gameOver: # Kiểm tra hết nước
                    self.gameOver = True
                    if self.gs.inCheck:
                        winner = "Black" if self.gs.turn == 'w' else "White" # Người thắng là người không bị chiếu hết
                        print(f"Checkmate! {winner} wins.")
                    elif self.gs.is_threefold_repetition():
                        print("Draw by Threefold Repetition!")
                    else:
                        print("Stalemate!") # Hoặc Draw!
                self.editChessPanel() # Cập nhật UI panel
                self.moveMade = False # Reset cờ

            # Cập nhật Pygame GUI Manager
            self.manager.update(self.time_delta)
            # Vẽ lại màn hình game
            self.drawGameScreen()
            # Cập nhật hiển thị Pygame
            pygame.display.update()

        # ---- Dọn dẹp khi thoát vòng lặp ----
        if self.aiMoveFinderProcess is not None and self.aiMoveFinderProcess.is_alive():
            print("Terminating AI process on exit.")
            self.aiMoveFinderProcess.terminate()
            self.aiMoveFinderProcess.join()


    def __eventHandler(self, isPlayerTurn):
        """Xử lý sự kiện cho chế độ PvC."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game Quit")
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                 # Chỉ cho phép nhấn chuột khi đến lượt người chơi và AI không đang tính toán
                if not self.gameOver and isPlayerTurn and not self.aiThinking:
                    location = pygame.mouse.get_pos()
                    # Kiểm tra xem có click vào bàn cờ không
                    if location[0] < WIDTH and location[1] < HEIGHT:
                         self.clickUserHandler() # Gọi hàm xử lý click của người chơi từ GameInit
                    # else: # Click ra ngoài bàn cờ (có thể là UI)
                    #     pass # manager.process_events sẽ xử lý

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    self.__reset()
                elif event.key == pygame.K_z: # Undo
                    # Chỉ cho phép undo khi đến lượt người chơi và AI không đang nghĩ
                    if isPlayerTurn and not self.aiThinking:
                         if len(self.gs.moveLog) >= 2: # Nếu có đủ nước đi để undo cả 2
                            print("Undoing last two moves (Player + AI)")
                            self.gs.undoMove() # Undo nước của AI
                            self.gs.undoMove() # Undo nước của người chơi
                            self.moveMade = True # Cần cập nhật lại
                            self.gameOver = False
                         elif len(self.gs.moveLog) == 1: # Chỉ có 1 nước đi (của người chơi)
                             print("Undoing player's first move")
                             self.gs.undoMove()
                             self.moveMade = True
                             self.gameOver = False
                         else:
                             print("No moves to undo.")
                    elif not isPlayerTurn:
                        print("Cannot undo while AI is thinking or during AI's turn.")

            # Xử lý sự kiện cho pygame_gui (quan trọng cho các nút/textbox)
            self.manager.process_events(event)


    def __reset(self):
        """Reset lại trạng thái trò chơi PvC."""
        print("Resetting Player vs AI game...")
        # Dừng process AI nếu đang chạy
        if self.aiMoveFinderProcess is not None and self.aiMoveFinderProcess.is_alive():
             print("Terminating active AI process during reset.")
             self.aiMoveFinderProcess.terminate()
             self.aiMoveFinderProcess.join()

        # Gọi lại __init__ của lớp này để reset đúng cách
        # Giữ nguyên lượt người chơi đã chọn ban đầu hoặc đặt lại mặc định
        # Ví dụ: đặt lại mặc định người chơi cầm quân Trắng
        self.__init__(playerTurn='w') # Hoặc self.__init__(self.playerTurn) để giữ nguyên

        print("Reset complete.")

# Kết thúc class PlayAIMode