games = ["Minecraft", "Battlefield", "Warzone", "CSGO"]

print("I like to p[lay these games: ")
for game in games:
    print(game)

new_game = ""
while new_game != "done":
    new_game = input("What's a game you like to play? \nType done once you are finished \n")
    if new_game != "done":
        games.append(new_game)

for game in games:
    print(game)