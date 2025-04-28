# Nội dung cho file: src/CompVsCompMode.py

import pygame
import pygame_gui  # Đảm bảo import này tồn tại
import math
import time
import random
from multiprocessing import Process, Queue

from src.GameInit import GameInit
from engine.AIEngine import AIEngine
# Import tất cả hằng số, bao gồm cả các hằng số mới
from src.config import *
from engine.ChessEngine import GameState, Move

# Định nghĩa thời gian giới hạn (tính bằng giây) - Có thể lấy từ config nếu muốn
AI_TIME_LIMIT = 10.0

class CompVsCompMode(GameInit):

    def __init__(self):
        # Khởi tạo GameInit
        super().__init__()

        # Đặt lại kích thước màn hình cho chế độ CvC (không có panel AI)
        self.screen = pygame.display.set_mode((WIDTH_WINDOW, HEIGHT_WINDOW))
        # Tạo UIManager mới với kích thước đúng
        self.manager = pygame_gui.UIManager((WIDTH_WINDOW, HEIGHT_WINDOW), theme_path="./data/theme_custom.json")
        # Gọi lại hàm load GUI (đã đổi tên thành _loadGUI trong GameInit)
        # để tạo các widget với manager mới và đúng kích thước
        self._loadGUI()
        # Cập nhật thông tin panel ban đầu
        self.editChessPanel()

        # Khởi tạo hai AI Engine
        self.ai_white = AIEngine('w')
        self.ai_black = AIEngine('b')

        # Biến quản lý trạng thái AI hiện tại
        self.currentAiThinking = False
        self.currentAiProcess = None
        self.currentAiStartTime = 0
        self.currentAiQueue = Queue() # Dùng chung cho các lần gọi process

        # Độ trễ giữa các nước đi (tùy chọn)
        self.move_delay_timer = 0
        self.MOVE_DELAY = 0.5 # Giây

        # Cập nhật validMoves ban đầu
        self.validMoves = self.gs.getValidMoves()


    def mainLoop(self):
        """Vòng lặp chính cho chế độ Máy vs Máy."""
        while self.running:
            self.time_delta = self.clock.tick(MAX_FPS) / 1000.0

            # Xử lý sự kiện (thoát, reset)
            self.__eventHandler()

            # Xác định AI hiện tại
            current_ai_engine = self.ai_white if self.gs.turn == 'w' else self.ai_black
            aiMove = None           # Nước đi cuối cùng được chọn
            made_ai_move = False    # Cờ báo đã xác định được nước đi hay chưa

            # ---- Logic AI ----
            # Bắt đầu tìm kiếm chính nếu đến lượt AI, chưa chạy, và không đang delay
            if not self.gameOver and not self.currentAiThinking and self.currentAiProcess is None and self.move_delay_timer <= 0:
                print(f"--- Turn: {self.gs.turn.upper()} ---")
                print(f"AI ({self.gs.turn}) is thinking (max {AI_TIME_LIMIT}s, depth {DEPTH})...")
                self.currentAiThinking = True
                self.currentAiStartTime = time.time()
                self.currentAiQueue = Queue() # Tạo queue mới cho lần chạy này
                # Cập nhật validMoves ngay trước khi AI suy nghĩ
                self.validMoves = self.gs.getValidMoves()
                if not self.validMoves: # Kiểm tra lại nếu không còn nước đi
                    self.gameOver = True
                    print("No valid moves found before starting AI.")
                else:
                    # Khởi chạy process tìm kiếm chính
                    self.currentAiProcess = Process(target=current_ai_engine.AlphaBetaPruning,
                                                    args=(self.gs, DEPTH, -math.inf, math.inf, self.currentAiQueue))
                    self.currentAiProcess.start()

            # Nếu AI đang suy nghĩ (tiến trình chính đang chạy)
            elif self.currentAiThinking:
                elapsedTime = time.time() - self.currentAiStartTime
                process_alive = self.currentAiProcess.is_alive()

                # 1. Kiểm tra xem tìm kiếm chính đã xong chưa
                if not process_alive:
                    print(f"AI ({self.gs.turn}) main process finished. Getting result...")
                    try:
                        ai_result = self.currentAiQueue.get_nowait()
                        aiMove = ai_result.bestMove
                        print(f"AI ({self.gs.turn}) finished thinking in {elapsedTime:.2f}s.")
                        made_ai_move = True # Đã có nước đi
                    except Exception as e:
                        print(f"DEBUG: Error getting AI ({self.gs.turn}) result from main queue: {e}")
                        aiMove = None # Lỗi -> sẽ fallback random
                        made_ai_move = True # Đánh dấu đã xử lý xong
                    # Dọn dẹp trạng thái tìm kiếm chính
                    self.currentAiThinking = False
                    self.currentAiProcess = None

                # 2. Kiểm tra xem tìm kiếm chính có bị timeout không
                elif elapsedTime > AI_TIME_LIMIT:
                    print(f"AI ({self.gs.turn}) main search timed out ({elapsedTime:.2f}s)! Terminating.")
                    self.currentAiProcess.terminate()
                    self.currentAiProcess.join()
                    self.currentAiThinking = False
                    self.currentAiProcess = None

                    # ---- Bắt đầu tìm kiếm nông ----
                    print(f"--- Starting shallow search (depth {REDUCED_DEPTH}, max {SHORT_TIMEOUT}s) ---")
                    shallow_queue = Queue()
                    # Cập nhật lại validMoves phòng trường hợp có thay đổi (dù không nên)
                    self.validMoves = self.gs.getValidMoves()
                    if not self.validMoves:
                         self.gameOver = True
                         print("No valid moves for shallow search.")
                         made_ai_move = True # Không còn nước đi
                         aiMove = None
                    else:
                        shallow_process = Process(target=current_ai_engine.AlphaBetaPruning,
                                                  args=(self.gs, REDUCED_DEPTH, -math.inf, math.inf, shallow_queue))
                        shallow_process.start()
                        shallow_process.join(timeout=SHORT_TIMEOUT) # Chờ tối đa SHORT_TIMEOUT

                        if shallow_process.is_alive():
                            # Tìm kiếm nông cũng timeout
                            print(f"AI ({self.gs.turn}) shallow search also timed out! Terminating.")
                            shallow_process.terminate()
                            shallow_process.join()
                            aiMove = None # Không có kết quả
                        else:
                            # Tìm kiếm nông đã kết thúc
                            try:
                                shallow_result = shallow_queue.get_nowait()
                                aiMove = shallow_result.bestMove
                                if aiMove:
                                    print(f"AI ({self.gs.turn}) using move from shallow search: {aiMove.getChessNotation()}")
                                else:
                                    print(f"AI ({self.gs.turn}) shallow search finished but returned no move.")
                            except Exception as e:
                                print(f"DEBUG: Error getting AI ({self.gs.turn}) result from shallow queue: {e}")
                                aiMove = None

                        # ---- Fallback cuối cùng: Chọn Random ----
                        if aiMove is None and not self.gameOver:
                            print(f"AI ({self.gs.turn}) falling back to random move.")
                            if self.validMoves:
                                aiMove = random.choice(self.validMoves)
                                print(f"AI ({self.gs.turn}) chooses random move: {aiMove.getChessNotation()}")
                            else:
                                # Trường hợp rất hiếm: timeout, shallow fail, và hết nước
                                print(f"AI ({self.gs.turn}) timed out, shallow failed, and no valid moves available.")
                                self.gameOver = True

                    made_ai_move = True # Đánh dấu đã xác định xong nước đi (nông hoặc random)

            # ---- Thực hiện nước đi đã chọn ----
            if made_ai_move and aiMove is not None and not self.gameOver:
                self.gs.makeMove(aiMove)
                self.moveMade = True # Cờ để cập nhật UI và trạng thái game
                self.move_delay_timer = self.MOVE_DELAY # Bắt đầu delay cho nước tiếp theo
                # Phát âm thanh
                if aiMove.capturedPiece != '--':
                    pygame.mixer.Sound.play(self.sound_capture)
                else:
                    pygame.mixer.Sound.play(self.sound_move)
            elif made_ai_move and aiMove is None and not self.gameOver:
                 # Không tìm được nước đi nào (kể cả random) dù game chưa over? -> Lỗi logic -> kết thúc game
                 print("Error: No move determined, but game not over. Setting game over.")
                 self.gameOver = True

            # ---- Cập nhật trạng thái và vẽ ----
            if self.moveMade:
                self.validMoves = self.gs.getValidMoves() # Cập nhật lại sau nước đi
                # print(f"Valid moves after move: {len(self.validMoves)}")
                if len(self.validMoves) == 0 and not self.gameOver: # Kiểm tra hết nước
                    self.gameOver = True
                    if self.gs.inCheck:
                        winner = "Black" if self.gs.turn == 'w' else "White"
                        print(f"Checkmate! {winner} wins.")
                    elif self.gs.is_threefold_repetition(): # Kiểm tra hòa lặp lại
                        print("Draw by Threefold Repetition!")
                    else: # Các trường hợp hòa khác (50 nước,...) nếu có
                        print("Stalemate!") # Hoặc Draw!
                self.editChessPanel() # Cập nhật UI panel
                self.moveMade = False # Reset cờ

            # Cập nhật Pygame GUI Manager
            self.manager.update(self.time_delta)
            # Vẽ lại toàn bộ màn hình game
            self.drawGameScreen()
            # Cập nhật hiển thị Pygame
            pygame.display.update()

            # Giảm thời gian chờ nếu đang delay
            if self.move_delay_timer > 0:
                self.move_delay_timer -= self.time_delta

        # ---- Dọn dẹp khi thoát vòng lặp ----
        if self.currentAiProcess is not None and self.currentAiProcess.is_alive():
            print("Terminating AI process on exit.")
            self.currentAiProcess.terminate()
            self.currentAiProcess.join()


    def __eventHandler(self):
        """Xử lý sự kiện cho chế độ CvC (chủ yếu Thoát và Reset)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game Quit")
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    self.__reset()
                elif event.key == pygame.K_z: # Undo (chỉ undo 1 nước AI gần nhất)
                    # Chỉ cho phép undo khi AI không đang nghĩ
                    if not self.currentAiThinking and len(self.gs.moveLog) > 0:
                        print("Undoing last AI move")
                        self.gs.undoMove()
                        self.moveMade = True # Cần cập nhật lại validMoves và UI
                        self.gameOver = False
                        self.move_delay_timer = 0 # Dừng delay nếu đang chạy
            # Xử lý sự kiện cho pygame_gui (nếu có nút bấm trên panel)
            self.manager.process_events(event)


    def __reset(self):
        """Reset lại trạng thái trò chơi CvC."""
        print("Resetting Comp vs Comp game...")
        # Dừng process AI nếu đang chạy
        if self.currentAiProcess is not None and self.currentAiProcess.is_alive():
             print("Terminating active AI process during reset.")
             self.currentAiProcess.terminate()
             self.currentAiProcess.join()

        # Gọi lại __init__ của chính lớp này để đảm bảo mọi thứ
        # (bao gồm màn hình, manager, AI engines, biến trạng thái)
        # được khởi tạo lại đúng cách cho chế độ CvC.
        self.__init__()

        print("Reset complete.")
