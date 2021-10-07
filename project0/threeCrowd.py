names = ["Dakin", "Aidan", "Alex", "Daniel"]

def checkCrowd():
    if len(names) > 3:
        print("the room is crowded")
    else:
        print("the room is not crowded")

checkCrowd()
del names[0]
del names[1]
checkCrowd()
