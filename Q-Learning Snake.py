import numpy as np
import pygame
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

# define game parameters
BLOCK_SIZE = 20
AI_SPEED = 100  # AI training speed
HUMAN_SPEED = 15  # human model speed
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)


class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, screen, font):
        # draw buttons
        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, current_color, self.rect)
        pygame.draw.rect(screen, GRAY, self.rect, 2)

        # draw text
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def is_clicked(self, mouse_pos, click):
        return self.rect.collidepoint(mouse_pos) and click


class Menu:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Game Menu')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.title_font = pygame.font.SysFont('arial', 40, bold=True)

        # creat buttons
        button_width, button_height = 200, 60
        center_x = width // 2 - button_width // 2

        self.ai_button = Button(
            center_x,
            height // 2 - 40,
            button_width,
            button_height,
            "AI",
            LIGHT_BLUE,
            GREEN
        )

        self.human_button = Button(
            center_x,
            height // 2 + 40,
            button_width,
            button_height,
            "HUMAN",
            LIGHT_BLUE,
            GREEN
        )

    def run(self):
        running = True

        while running:
            click = False
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        click = True

            self.ai_button.check_hover(mouse_pos)
            self.human_button.check_hover(mouse_pos)

            if self.ai_button.is_clicked(mouse_pos, click):
                return "AI"

            if self.human_button.is_clicked(mouse_pos, click):
                return "HUMAN"

            # draw the menu
            self.display.fill(BLACK)

            # draw the title
            title_surface = self.title_font.render("SNAKE", True, WHITE)
            title_rect = title_surface.get_rect(center=(self.width // 2, self.height // 4))
            self.display.blit(title_surface, title_rect)

            # draw buttons
            self.ai_button.draw(self.display, self.font)
            self.human_button.draw(self.display, self.font)

            pygame.display.flip()
            self.clock.tick(30)

        return None


class SnakeGame:
    def __init__(self, width=640, height=480, human_player=False):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Game AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.human_player = human_player
        self.speed = HUMAN_SPEED if human_player else AI_SPEED
        self.reset()

    # initializes or resets the game state
    def reset(self):
        self.direction = 'RIGHT'
        self.head = [self.width / 2, self.height / 2]
        self.snake = [
            self.head,
            [self.head[0] - BLOCK_SIZE, self.head[1]],
            [self.head[0] - 2 * BLOCK_SIZE, self.head[1]]
        ]
        self.score = 0
        self.food = self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    # randomly positions new food
    def _place_food(self):
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food = [x, y]
            if food not in self.snake:
                return food

    #  handles one step of gameplay
    def play_step(self, action, game_number=None):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move snake
        self._move(action)
        self.snake.insert(0, list(self.head))

        # check if the game is ending
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # check if the snake eat the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        # update UI
        self._update_ui(game_number)
        self.clock.tick(self.speed)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hit a wall
        if pt[0] >= self.width or pt[0] < 0 or pt[1] >= self.height or pt[1] < 0:
            return True
        # hit oneself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self, game_number=None):
        self.display.fill(BLACK)

        # draw the snake
        for i, pt in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(self.display, BLUE, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE), 1)

        # draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        # Show scores and game rounds
        if game_number is not None:
            text = self.font.render(f"Score: {self.score} | Game: {game_number}", True, WHITE)
        else:
            text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [10, 10])

        pygame.display.flip()

    #  translates action inputs into directional movement
    def _move(self, action):
        # action convert to direction
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # turn right
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # turn left

        self.direction = new_dir

        x = self.head[0]
        y = self.head[1]
        if self.direction == 'RIGHT':
            x += BLOCK_SIZE
        elif self.direction == 'LEFT':
            x -= BLOCK_SIZE
        elif self.direction == 'DOWN':
            y += BLOCK_SIZE
        elif self.direction == 'UP':
            y -= BLOCK_SIZE

        self.head = [x, y]

    def _get_state(self):
        head = self.snake[0]

        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        # current direction
        dir_l = self.direction == 'LEFT'
        dir_r = self.direction == 'RIGHT'
        dir_u = self.direction == 'UP'
        dir_d = self.direction == 'DOWN'

        state = [
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # move direction
            dir_l, dir_r, dir_u, dir_d,

            # relative position of food
            self.food[0] < self.head[0],  # food in the left
            self.food[0] > self.head[0],  # food in the right
            self.food[1] < self.head[1],  # food on the top
            self.food[1] > self.head[1]  # food on the down
        ]
        return np.array(state, dtype=int)

    def get_state(self):
        return self._get_state()


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Controlling randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 256, 3)  # The input size is 11, the hidden layer 256, and the output is 3 actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 32

        # Record scores and average scores
        self.scores = []
        self.mean_scores = []
        self.total_score = 0

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        action_idx = np.argmax(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return

        # random sampling from memory
        mini_sample = random.sample(self.memory, self.batch_size)

        states = torch.tensor(np.array([state for state, _, _, _, _ in mini_sample]), dtype=torch.float)
        actions = torch.tensor([action for _, action, _, _, _ in mini_sample], dtype=torch.long)
        rewards = torch.tensor([reward for _, _, reward, _, _ in mini_sample], dtype=torch.float)
        next_states = torch.tensor(np.array([next_state for _, _, _, next_state, _ in mini_sample]), dtype=torch.float)
        dones = torch.tensor([done for _, _, _, _, done in mini_sample], dtype=torch.float)

        # get current Q value
        current_q_values = self.model(states)
        current_q_value = current_q_values.gather(1, actions.unsqueeze(-1))

        # calculate the target Q
        with torch.no_grad():
            next_q_values = self.model(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value

        loss = nn.MSELoss()(current_q_value.squeeze(), target_q_value.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = np.argmax(action)

        # prepare data
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor([action_idx], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.float)

        # get current Q value
        current_q_values = self.model(state.unsqueeze(0))
        current_q_value = current_q_values.gather(1, action.unsqueeze(-1))

        # calculate the target Q
        with torch.no_grad():
            next_q_values = self.model(next_state.unsqueeze(0))
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value

        loss = nn.MSELoss()(current_q_value.squeeze(), target_q_value.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    record = 0
    agent = Agent()
    game = SnakeGame()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            # get current state
            state_old = agent.get_state(game)

            # get action
            final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(final_move, agent.n_games + 1)
            state_new = agent.get_state(game)

            # train short-term memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long-term memory and reset the game
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    model_folder_path = './model'
                    if not os.path.exists(model_folder_path):
                        os.makedirs(model_folder_path)
                    torch.save(agent.model.state_dict(), f'./model/snake_model_{record}.pth')

                # Update score records
                agent.scores.append(score)
                agent.total_score += score
                if agent.n_games > 0:
                    agent.mean_scores.append(agent.total_score / agent.n_games)

                print(f'Game {agent.n_games}, Score {score}, Record {record}, Epsilon {agent.epsilon}')

    except KeyboardInterrupt:
        print("Game over!")

        # save the final model
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(agent.model.state_dict(), './model/final_model.pth')
        print("Saved final model.")


def play_human():
    game = SnakeGame(human_player=True)
    game_number = 1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if game.direction != 'RIGHT':
                        game.direction = 'LEFT'
                elif event.key == pygame.K_RIGHT:
                    if game.direction != 'LEFT':
                        game.direction = 'RIGHT'
                elif event.key == pygame.K_UP:
                    if game.direction != 'DOWN':
                        game.direction = 'UP'
                elif event.key == pygame.K_DOWN:
                    if game.direction != 'UP':
                        game.direction = 'DOWN'
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        # move snake
        x, y = game.head
        if game.direction == 'RIGHT':
            x += BLOCK_SIZE
        elif game.direction == 'LEFT':
            x -= BLOCK_SIZE
        elif game.direction == 'DOWN':
            y += BLOCK_SIZE
        elif game.direction == 'UP':
            y -= BLOCK_SIZE
        game.head = [x, y]

        game.snake.insert(0, list(game.head))

        # check if the snake eat the food
        if game.head == game.food:
            game.score += 1
            game.food = game._place_food()
        else:
            game.snake.pop()

        # check if the game is ending
        game_over = False
        if game._is_collision():
            game_over = True
            print(f"Game {game_number} Score: {game.score}")
            game_number += 1
            game.reset()

        # update UI
        game._update_ui(game_number)
        game.clock.tick(game.speed)


if __name__ == '__main__':
    pygame.init()

    menu = Menu()
    choice = menu.run()

    if choice == "AI":
        train()
    elif choice == "HUMAN":
        play_human()