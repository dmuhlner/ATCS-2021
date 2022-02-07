import random


class TicTacToe:
    random_game = 1
    user_game = 0

    def __init__(self, type):
        self.board = "\t1\t2\t3\nA\t-\t-\t-\nB\t-\t-\t-\nC\t-\t-\t-\n"

        # self.user_game = 0
        # self.random_game = 1

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
        self.indices = [9,11,13,17,19,21,25,27,29]

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

    def is_valid_AI_move(self, index):
        return self.board[index] == "-"

    def place_player(self, player, index):
        self.board = self.board[0:index] + player + self.board[index + 1:]
        return

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
        if self.type == self.user_game or player == "X":
            print(player + ": it is your turn to move")
            self.take_manual_turn(player)
        elif self.type == self.random_game:
            self.take_random_turn(player)
        return

    def check_col_win(self, player):
        if self.board[self.A1] == player and self.board[self.A2] == player and self.board[self.A3] == player:
            return True
        if self.board[self.B1] == player and self.board[self.B2] == player and self.board[self.B3] == player:
            return True
        if self.board[self.C1] == player and self.board[self.C2] == player and self.board[self.C3] == player:
            return True
        return False

    def check_row_win(self, player):
        if self.board[self.A1] == player and self.board[self.B1] == player and self.board[self.C1] == player:
            return True
        if self.board[self.A2] == player and self.board[self.B2] == player and self.board[self.C2] == player:
            return True
        if self.board[self.A3] == player and self.board[self.B3] == player and self.board[self.C3] == player:
            return True
        return False

    def check_diag_win(self, player):
        if self.board[self.A1] == player and self.board[self.B2] == player and self.board[self.C3] == player:
            return True
        if self.board[self.C1] ==player and self.board[self.B2] == player and self.board[self.A3] == player:
            return True
        return False

    def check_win(self, player):
        return self.check_col_win(player) or self.check_row_win(player) or self.check_diag_win(player)

    def check_tie(self):
        if self.board.count("-") == 0:
            return True
        return False

    def take_random_turn(self, player):
        valid = False
        while not valid:
            move = random.randrange(8)

            if self.is_valid_AI_move(self.indices[move]):
                self.place_player(player,self.indices[move])
                return

        print(move)

    def reset(self):
        self.board = "\t1\t2\t3\nA\t-\t-\t-\nB\t-\t-\t-\nC\t-\t-\t-\n"
        self.won = False
        self.print_board()

    def play_game(self):

        self.reset()
        self.print_instructions()
        player = "O"
        while not self.won:
            if player == "X":
                player = "O"
            else:
                player = "X"

            self.take_turn(player)
            self.won = self.check_win(player)
            self.print_board()
            if self.check_tie():
                break

        if self.won:
            print(player + " wins!\n")
        else:
            print("Tie")

        again = input("Would you like to play again?\n")
        if again == "Yes" or again == "yes":
            self.play_game()
        return

