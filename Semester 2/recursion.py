def factorial(n):
    if n == 0:
        return 1

    return n * factorial(n - 1)


def countdown(n):
    if n == 0:
        return "Blastoff!"
    return str(n) + " " + countdown(n-1)



def reverse(input):
    if len(input) == 0:
        return ""

    return reverse(input[1:]) + input[0]

def numBacteriaAlive(n):
    if n == 0:
        return 10

    return 3*numBacteriaAlive(n-1)
