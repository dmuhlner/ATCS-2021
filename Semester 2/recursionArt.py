import math
import turtle as tr

wn = tr.Screen()
wn.title("Recursion Art")
draw = tr.Turtle()

def spiral (length, angle, multiplier):
    if length > 200:
        return
    if length < 1:
        return
    draw.forward(length)
    draw.right(angle)
    spiral(length * multiplier, angle, multiplier)

def tree (trunk, height, angle = 0, xpos=0, ypos=-200):
    if height == 0:
        return
    draw.penup()
    draw.setx(xpos)
    draw.sety(ypos)
    draw.pendown()
    draw.setheading(0)
    draw.left(90-angle)
    draw.forward(trunk)
    x = draw.xcor()
    y = draw.ycor()
    tree(trunk/1.5, height-1, angle+45, x, y)
    tree(trunk / 1.5, height - 1, angle - 45, x, y)


def snowflake (side, levels, xpos=0, ypos=0, angle = 0):
    draw.setheading(angle)
    draw.penup()
    draw.setx(xpos)
    draw.sety(ypos)
    draw.pendown()

    if levels==0:
        draw.forward(side)
        return

    for i in range(3):
        draw.forward(side/3)
        draw.penup()
        draw.forward(side/3)
        draw.pendown()
        draw.forward(side/3)
        draw.left(120)

    snowflake(side/3, levels-1, xpos+side/3, ypos, angle - 60)
    snowflake(side/3, levels-1, xpos+side-side/6, ypos+side/6*math.sqrt(3), 120 + angle - 60)
    snowflake(side/3, levels-1, xpos+side/3, ypos+side/3*math.sqrt(3), 240 + angle - 60)

# snowflake(300,2)
# tree(200, 5)
#spiral(1, -45, 1.1)