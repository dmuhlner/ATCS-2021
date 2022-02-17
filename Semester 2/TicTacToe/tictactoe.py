import random
import time


class TicTacToe:
    random_game = 1
    user_game = 0
    minimax_game = 2
    ai_only_game = 3
    fair_game = 4
    alpha_beta = 5

    def __init__(self, type = None):
        self.board = "\t1\t2\t3\nA\t-\t-\t-\nB\t-\t-\t-\nC\t-\t-\t-\n"

        # self.user_game = 0
        # self.random_game = 1

        if type is None:
            playerIn = ""
            while playerIn != "minimax" and playerIn != "random" and playerIn != "user" and playerIn != "ai" and playerIn != "fair" and playerIn != "ab":
                print("What type of game would you like to play? Options:\nMinimax\nRandom\nUser\nAI\nFair\nAB")
                playerIn = input().lower()
            if playerIn == "minimax":
                type = TicTacToe.minimax_game
            elif playerIn == "random":
                type = TicTacToe.random_game
            elif playerIn == "user":
                type = TicTacToe.user_game
            elif playerIn == "ai":
                type = TicTacToe.ai_only_game
            elif playerIn == "fair":
                type = TicTacToe.fair_game
            elif playerIn == "ab":
                type = TicTacToe.alpha_beta
        elif isinstance(type, int) and (type == 0 or type == 1 or type ==2 or type == 3 or type == 4 or type == 5):
            type = type
        else:
            return

        self.type = type

        self.A1 = 9
        self.A2 = 11
        self.A3 = 13
        self.B1 = 17
        self.B2 = 19
        self.B3 = 21
        self.C1 = 25
        self.C2 = 27
        self.C3 = 29
        self.indices = [self.A1,self.A2,self.A3,self.B1,self.B2,self.B3,self.C1,self.C2,self.C3]

        self.won = False



    def print_instructions(self):
        print("Each player gets a letter, either X or O.")
        print("When you are prompted, place your letter in one of the spots on the board that is occupied by a dash.")
        print("To do this, type in the location you want to play it byt writing the letter, then number.")
        print("For instance, typing A1 would put your letter in the top left spot.")
        print()
        return

    def print_board(self):
        print(self.board)
        return

    def is_valid_move(self, index):
        if index is None or len(index) != 2:
            return False
        first = index[0]
        second = index[1]
        if first != "A" and first != "B" and first != "C":
            return False
        if second != "1" and second != "2" and second != "3":
            return False
        loc = self.getIndex(index)
        return self.board[loc] == "-"

    def is_valid_AI_move(self, index, board):
        return board[index] == "-"

    def place_player(self, player, index):
        self.board = self.board[0:index] + player + self.board[index + 1:]

    def take_manual_turn(self, player):
        index = None
        while True:
            print("Enter a valid move as a letter then a number not separated by a space. For example, 'A1'")
            move = input()
            if self.is_valid_move(move):
                break

        self.place_player(player, self.getIndex(move))
        return

    def getIndex(self, index):
       first = index[0]
       second = index[1]
       if first == "A":
            if second == "1":
                return self.A1
            if second == "2":
                return self.A2
            if second == "3":
                return self.A3
       if first == "B":
           if second == "1":
               return self.B1
           if second == "2":
               return self.B2
           if second == "3":
               return self.B3
       if first == "C":
               if second == "1":
                   return self.C1
               if second == "2":
                   return self.C2
               if second == "3":
                   return self.C3

    def take_turn(self, player):
        if self.type == self.ai_only_game:
            self.take_minimax_turn(player)
        elif self.type == self.user_game or player == "X":
            print(player + ": it is your turn to move")
            self.take_manual_turn(player)
        elif self.type == self.random_game:
            self.take_random_turn(player)
        elif self.type == self.minimax_game:
            self.take_minimax_turn(player, 0)
        elif self.type == self.fair_game:
            self.take_minimax_turn(player, 1)
        elif self.type == self.alpha_beta:
            self.take_ab_turn(player)
        return

    def check_col_win(self, player, board):
        if board[self.A1] == player and board[self.A2] == player and board[self.A3] == player:
            return True
        if board[self.B1] == player and board[self.B2] == player and board[self.B3] == player:
            return True
        if board[self.C1] == player and board[self.C2] == player and board[self.C3] == player:
            return True
        return False

    def check_row_win(self, player, board):
        if board[self.A1] == player and board[self.B1] == player and board[self.C1] == player:
            return True
        if board[self.A2] == player and board[self.B2] == player and board[self.C2] == player:
            return True
        if board[self.A3] == player and board[self.B3] == player and board[self.C3] == player:
            return True
        return False

    def check_diag_win(self, player, board):
        if board[self.A1] == player and board[self.B2] == player and board[self.C3] == player:
            return True
        if board[self.C1] == player and board[self.B2] == player and board[self.A3] == player:
            return True
        return False

    def check_win(self, player, board):
        return self.check_col_win(player, board) or self.check_row_win(player, board) or self.check_diag_win(player, board)

    def check_tie(self, board):
        if board.count("-") == 0:
            return True
        return False

    def take_random_turn(self, player):
        valid = False
        while not valid:
            move = random.randrange(8)

            if self.is_valid_AI_move(self.indices[move], self.board):
                self.place_player(player,self.indices[move])
                return

        print(move)

    def reset(self):
        self.board = "\t1\t2\t3\nA\t-\t-\t-\nB\t-\t-\t-\nC\t-\t-\t-\n"
        self.won = False
        self.print_board()

    def opposite_player(self, player):
        if player == "X":
            return "O"
        return "X"

    def minimax(self, player, max, depth):
        if self.check_win("O", self.board):
            return 10
        elif self.check_win(self.opposite_player("O"), self.board):
            return -10
        if self.check_tie(self.board):
            return 0

        if depth != 0:
            present_board = self.board
            keep = 0
            if max:
                keep_success = -11
                for move in self.indices:
                    self.board = present_board
                    if self.is_valid_AI_move(move, self.board):
                        self.place_player(player, move)
                        success = self.minimax(self.opposite_player(player), False, depth - 1)
                        if success > keep_success:
                            keep = move
                            keep_success = success
            else:
                keep_success = 11
                for move in self.indices:
                    self.board = present_board
                    if self.is_valid_AI_move(move, self.board):
                        self.place_player(player, move)
                        success = self.minimax(self.opposite_player(player), True, depth - 1)
                        if success < keep_success:
                            keep = move
                            keep_success = success
            self.board = present_board
            return keep_success
        return 0

    def ab_minimax(self, player, max, depth, alpha, beta):
        if self.check_win("O", self.board):
            return 10
        elif self.check_win(self.opposite_player("O"), self.board):
            return -10
        if self.check_tie(self.board):
            return 0

        if depth != 0:
            present_board = self.board
            keep = 0
            if max:
                for move in self.indices:
                    self.board = present_board
                    if self.is_valid_AI_move(move, self.board):
                        self.place_player(player, move)
                        success = self.ab_minimax(self.opposite_player(player), False, depth - 1, alpha, beta)
                        if success > alpha:
                            keep = move
                            alpha = success
                    if alpha >= beta:
                        break
                self.board = present_board
                return alpha
            else:
                for move in self.indices:
                    self.board = present_board
                    if self.is_valid_AI_move(move, self.board):
                        self.place_player(player, move)
                        success = self.ab_minimax(self.opposite_player(player), True, depth - 1, alpha, beta)
                        if success < beta:
                            keep = move
                            beta = success
                    if beta <= alpha:
                        break
                self.board = present_board
                return beta
        return 0

    def take_minimax_turn(self, player, version):
        if version == 1:
            depth = 3
        else:
            depth = 100

        best = [0, -11]
        present_board = self.board
        for move in self.indices:
            self.board = present_board
            if self.is_valid_AI_move(move, self.board):
                self.place_player(player, move)
                current = [move, self.minimax(self.opposite_player(player), False, depth)]
                if current[1] > best[1]:
                    best = current
        self.board = present_board
        self.place_player(player, best[0])

    def take_ab_turn(self, player):
        best = [0, -110]
        present_board = self.board
        for move in self.indices:
            self.board = present_board
            if self.is_valid_AI_move(move, self.board):
                self.place_player(player, move)
                current = [move, self.ab_minimax(self.opposite_player(player), False, 3, best[1], 110)]
                if current[1] > best[1]:
                    best = current
        self.board = present_board
        self.place_player(player, best[0])

    def play_game(self):
        self.reset()
        self.print_instructions()
        player = "O"
        while not self.won:
            if player == "X":
                player = "O"
            else:
                player = "X"
            start = time.time()
            self.take_turn(player)
            end = time.time()
            print("This turn took: ", end-start, " seconds")
            self.won = self.check_win(player, self.board)
            self.print_board()
            if self.check_tie(self.board):
                break

        if self.won:
            print(player + " wins!\n")
        else:
            print("Tie")

        again = input("Would you like to play again?\n")
        if again == "Yes" or again == "yes" or again == "y" or again == "Y":
            self.play_game()
        return
