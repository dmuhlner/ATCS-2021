games = ["Minecraft", "Battlefield", "Warzone", "CSGO"]

print("I like to p[lay these games: ")
for game in games:
    print(game)

new_game = input("What's a game you like to play? \n")

games.append(new_game)

for game in games:
    print(game)