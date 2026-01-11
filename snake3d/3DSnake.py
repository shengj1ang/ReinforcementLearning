import random
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


edges = (
    (0, 1),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
    (0, 3),
    (0, 4),
)

surfaces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6),
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
    (0, 1, 1),
)


def cube_solid(x, y, z, shade):
    verticies = (
        (x + 1, y - 1, z - 1),
        (x + 1, y + 1, z - 1),
        (x - 1, y + 1, z - 1),
        (x - 1, y - 1, z - 1),
        (x + 1, y - 1, z + 1),
        (x + 1, y + 1, z + 1),
        (x - 1, y - 1, z + 1),
        (x - 1, y + 1, z + 1),
    )

    glBegin(GL_QUADS)
    for surface in surfaces:
        for vertex in surface:
            glColor3fv(colors[shade])
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(colors[10])
            glVertex3fv(verticies[vertex])
    glEnd()


def cube_wire(x, y, z):
    verticies = (
        (x + 1, y - 1, z - 1),
        (x + 1, y + 1, z - 1),
        (x - 1, y + 1, z - 1),
        (x - 1, y - 1, z - 1),
        (x + 1, y - 1, z + 1),
        (x + 1, y + 1, z + 1),
        (x - 1, y - 1, z + 1),
        (x - 1, y + 1, z + 1),
    )

    glBegin(GL_LINES)
    glColor3f(0.8, 0.8, 0.8)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()




pygame.init()

GridSize = 8

fruit_x = random.randint(0, GridSize - 1) * 2
fruit_y = random.randint(0, GridSize - 1) * 2
fruit_z = random.randint(0, GridSize - 1) * 2

direction = (1, 0, 0)
TailPos = [[0, 0, 0], [2, 0, 0], [4, 0, 0]]

SPEED_MS = 200        # 蛇移动速度（越大越慢）
GROW_PER_FRUIT = 1    # 每吃一个果子长几节
growth = 0            # 剩余“增长步数”
score = 0             # 得分（吃到果子次数）
game_over = False     # 游戏结束标志

MOVEEVENT = pygame.USEREVENT + 1
pygame.time.set_timer(MOVEEVENT, SPEED_MS)

display = (800, 800)
screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

gluPerspective(60, display[0] / display[1], 0.1, 100.0)


# 摄像机自动根据网格大小后退
camera_distance = GridSize * 4 + 10       # 根据网格大小自动调整镜头距离
glTranslatef(-GridSize, -GridSize, -camera_distance)


glEnable(GL_DEPTH_TEST)

clock = pygame.time.Clock()
running = True

dragging = False
last_mouse_pos = (0, 0)


def spawn_fruit():
    global fruit_x, fruit_y, fruit_z
    while True:
        fx = random.randint(0, GridSize - 1) * 2
        fy = random.randint(0, GridSize - 1) * 2
        fz = random.randint(0, GridSize - 1) * 2
        if [fx, fy, fz] not in TailPos:
            fruit_x, fruit_y, fruit_z = fx, fy, fz
            break


def update_caption():
    if game_over:
        pygame.display.set_caption(
            f"3D Snake - Game Over | Length: {len(TailPos)}  Score: {score}"
        )
    else:
        pygame.display.set_caption(
            f"3D Snake - Length: {len(TailPos)}  Score: {score}"
        )


update_caption()

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            dragging = True
            last_mouse_pos = event.pos

        if event.type == MOUSEBUTTONUP and event.button == 1:
            dragging = False

        if event.type == MOUSEMOTION and dragging:
            dx = event.pos[0] - last_mouse_pos[0]
            dy = event.pos[1] - last_mouse_pos[1]
            last_mouse_pos = event.pos

            glMatrixMode(GL_MODELVIEW)
            glRotatef(dx * 0.3, 0, 1, 0)
            glRotatef(dy * 0.3, 1, 0, 0)

        if event.type == MOVEEVENT and not game_over:
            x, y, z = TailPos[-1]
            dx, dy, dz = direction
            nx = x + dx * 2
            ny = y + dy * 2
            nz = z + dz * 2

            if nx >= GridSize * 2:
                nx = 0
            elif nx < 0:
                nx = (GridSize - 1) * 2
            if ny >= GridSize * 2:
                ny = 0
            elif ny < 0:
                ny = (GridSize - 1) * 2
            if nz >= GridSize * 2:
                nz = 0
            elif nz < 0:
                nz = (GridSize - 1) * 2

            new_head = [nx, ny, nz]
            eating = (nx == fruit_x and ny == fruit_y and nz == fruit_z)

            will_pop_tail = (growth == 0 and not eating)

            if will_pop_tail:
                body_to_check = TailPos[1:]
            else:
                body_to_check = TailPos

            # 撞到蛇身 → 游戏结束
            if new_head in body_to_check:
                game_over = True
                update_caption()
            else:
                TailPos.append(new_head)

                if eating:
                    growth += GROW_PER_FRUIT
                    score += 1
                    spawn_fruit()
                else:
                    if growth > 0:
                        growth -= 1
                    else:
                        TailPos.pop(0)

                update_caption()

    # 方向控制：禁止直接 180° 掉头
    if not game_over:
        keys = pygame.key.get_pressed()
        new_dir = list(direction)

        if keys[pygame.K_w]:
            new_dir = [0, -1, 0]
        elif keys[pygame.K_s]:
            new_dir = [0, 1, 0]
        elif keys[pygame.K_a]:
            new_dir = [-1, 0, 0]
        elif keys[pygame.K_d]:
            new_dir = [1, 0, 0]
        elif keys[pygame.K_q]:
            new_dir = [0, 0, 1]
        elif keys[pygame.K_e]:
            new_dir = [0, 0, -1]

        if not (
            new_dir[0] == -direction[0]
            and new_dir[1] == -direction[1]
            and new_dir[2] == -direction[2]
        ):
            direction = tuple(new_dir)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    for gx in range(GridSize):
        for gy in range(GridSize):
            for gz in range(GridSize):
                cube_wire(2 * gx, 2 * gy, 2 * gz)

    cube_solid(fruit_x, fruit_y, fruit_z, 6)

    for seg in TailPos:
        cube_solid(seg[0], seg[1], seg[2], 1)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()