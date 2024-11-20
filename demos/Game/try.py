import pygame
import sys


def game_demo():
    # 初始化 Pygame
    pygame.init()

    # 设置屏幕大小
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Pygame Example")

    # 设置颜色
    WHITE = (255, 255, 255)

    # 设置时钟
    clock = pygame.time.Clock()

    # 加载图像
    player_image = pygame.image.load("tmp/mine.jpg")
    player_rect = player_image.get_rect()
    player_rect.topleft = (400, 300)

    # 游戏主循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 按键检测
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player_rect.x -= 5
        if keys[pygame.K_RIGHT]:
            player_rect.x += 5
        if keys[pygame.K_UP]:
            player_rect.y -= 5
        if keys[pygame.K_DOWN]:
            player_rect.y += 5

        # 绘制更新
        screen.fill(WHITE)
        screen.blit(player_image, player_rect)
        pygame.display.flip()

        # 控制帧率
        clock.tick(60)

    # 退出 Pygame
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    game_demo()
