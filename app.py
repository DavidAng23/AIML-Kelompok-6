import eel
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import threading
import json
import os
from collections import defaultdict

# --- Konfigurasi Eel & Flask ---
WEB_FOLDER = 'web'
eel.init(WEB_FOLDER)
FLASK_PORT = 5001 # Port untuk API backend
app = Flask(__name__)
CORS(app)

# --- Implementasi AI Reinforcement Learning (Q-Learning) ---
class RLWaveAI:
    def __init__(self, q_table_file='q_table.json'):
        # Parameter Q-Learning
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.9
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.05

        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()

        self.actions = ["balanced_mix", "fast_overwhelm", "tough_wall", "prioritizer_rush", "healer_support"]
        
        self.last_state = None
        self.last_action = None

        self.enemy_types = {
            "basic": {"hp": 10, "speed": 1.0, "reward": 5, "color": "red"},
            "fast": {"hp": 8, "speed": 1.5, "reward": 7, "color": "orange"},
            "tough": {"hp": 25, "speed": 0.7, "reward": 10, "color": "purple"},
            "prioritizer": {"hp": 15, "speed": 0.8, "reward": 8, "color": "darkgreen", "special_behavior": "attack_towers", "attack_range": 150, "attack_damage": 2, "attack_speed": 60},
            "healer": {"hp": 12, "speed": 0.9, "reward": 6, "color": "lightblue", "special_behavior": "heal_allies", "heal_range": 100, "heal_amount": 5, "heal_speed": 90}
        }
        
        # Definisi Menara di Sisi AI ---
        self.tower_definitions = {
            'basic': { 'cost': 50, 'type': 'basic' },
            'fast': { 'cost': 75, 'type': 'fast' },
            'heavy': { 'cost': 120, 'type': 'heavy' }
        }

        self.strategy_configs = {
            "balanced_mix": {"basic": 0.4, "fast": 0.2, "tough": 0.15, "prioritizer": 0.1, "healer": 0.15},
            "fast_overwhelm": {"basic": 0.2, "fast": 0.6, "tough": 0.1, "prioritizer": 0.1},
            "tough_wall": {"basic": 0.2, "tough": 0.6, "fast": 0.1, "healer": 0.1},
            "prioritizer_rush": {"basic": 0.4, "prioritizer": 0.4, "fast": 0.2},
            "healer_support": {"basic": 0.3, "healer": 0.3, "tough": 0.2, "prioritizer": 0.2},
        }
        
        print("AI: Initialized Reinforcement Learning (Q-Learning) Agent.")
        print(f"AI: Loaded {len(self.q_table)} states from Q-Table.")

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            try:
                with open(self.q_table_file, 'r') as f:
                    q_data = json.load(f)
                    return defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in q_data.items()})
            except (json.JSONDecodeError, IOError):
                 print(f"Warning: Could not read {self.q_table_file}. Starting with a new Q-Table.")
                 return defaultdict(lambda: defaultdict(float))
        return defaultdict(lambda: defaultdict(float))

    def save_q_table(self):
        try:
            with open(self.q_table_file, 'w') as f:
                json.dump(self.q_table, f, indent=4)
            print(f"AI: Q-Table saved. Total states known: {len(self.q_table)}")
        except IOError:
            print(f"Error: Could not save Q-Table to {self.q_table_file}.")

    # Fungsi menganalisis komposisi menara pemain
    def get_state(self, lives, coins, towers_data):
        # Keadaan Nyawa 
        if lives > 7: lives_state = "sehat"
        elif lives > 3: lives_state = "terluka"
        else: lives_state = "kritis"
        
        # Keadaan Ekonomi 
        if coins < 150: eco_state = "miskin"
        elif coins < 300: eco_state = "cukup"
        else: eco_state = "kaya"
            
        # Analisis Komposisi Pertahanan 
        tower_investment = defaultdict(int)
        for tower in towers_data:
            tower_type = tower.get('type', 'basic')
            cost = self.tower_definitions.get(tower_type, {}).get('cost', 50)
            tower_investment[tower_type] += cost
        
        # Kategori investasi pada setiap tipe menara
        # Thresholds (ambang batas) bisa disesuaikan untuk mengubah sensitivitas AI
        def get_investment_category(cost):
            if cost == 0:
                return "nihil"
            elif cost < 150: # (cukup untuk 1-2 menara murah)
                return "sedikit"
            elif cost < 350: # (beberapa menara atau 1-2 menara mahal)
                return "sedang"
            else: # (investasi besar)
                return "banyak"

        basic_state = get_investment_category(tower_investment['basic'])
        fast_state = get_investment_category(tower_investment['fast'])
        heavy_state = get_investment_category(tower_investment['heavy'])
        
        state_string = (
            f"nyawa:{lives_state}_ekonomi:{eco_state}_"
            f"basic:{basic_state}_fast:{fast_state}_heavy:{heavy_state}"
        )
        return state_string

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            action_values = self.q_table[state]
            if not action_values:
                return random.choice(self.actions)
            return max(action_values, key=action_values.get)

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value
        print(f"AI LEARN: State='{state}', Action='{action}', Reward={reward:.1f}, New Q-Value={new_value:.2f}")

    def learn(self, last_wave_stats, current_lives, current_coins, current_towers):
        if not self.last_state or not self.last_action:
            print("AI SKIP LEARN: No last state/action.")
            return

        # Reward didasarkan pada performa pemain
        lives_lost = last_wave_stats['initial_lives'] - last_wave_stats['lives_remaining']
        enemies_defeated = last_wave_stats['enemies_defeated']
        
        reward = 0
        # Hadiah besar jika pemain tidak kehilangan nyawa
        if lives_lost == 0:
            reward += 50 
        # Penalti kecil jika kehilangan sedikit nyawa
        elif 1 <= lives_lost <= 2:
            reward -= 15
        # Penalti besar jika kehilangan banyak nyawa
        else:
            reward -= 40

        # Penalti sangat besar jika pemain kalah total
        if last_wave_stats['lives_remaining'] <= 0:
            reward = -100

        # Sedikit hadiah tambahan berdasarkan musuh yang dikalahkan
        reward += (enemies_defeated * 0.5)

        next_state = self.get_state(current_lives, current_coins, current_towers)
        self.update_q_table(self.last_state, self.last_action, reward, next_state)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        return reward

    def generate_wave(self, action, current_wave_num, current_stage):
        difficulty_multiplier = 1 + (current_wave_num * 0.08) + (current_stage * 0.15)
        total_enemies = int((5 + current_wave_num * 0.8 + current_stage * 2) * random.uniform(0.9, 1.1))
        
        hp_modifier = 1.0
        speed_modifier = 1.0
        insight_modifier_text = ""

        if action == "fast_overwhelm":
            speed_modifier = 1.15
            insight_modifier_text = " (Speed Boost)"
        elif action == "tough_wall":
            hp_modifier = 1.20
            insight_modifier_text = " (HP Boost)"
        elif action == "healer_support":
             hp_modifier = 1.10
             insight_modifier_text = " (Tougher Composition)"
        
        current_strategy_probs = self.strategy_configs.get(action, self.strategy_configs["balanced_mix"])

        wave_composition = []
        for _ in range(total_enemies):
            r = random.random()
            cumulative_prob = 0
            selected_enemy_type = "basic"
            for enemy_type, prob in current_strategy_probs.items():
                cumulative_prob += prob
                if r < cumulative_prob:
                    selected_enemy_type = enemy_type
                    break
            wave_composition.append(selected_enemy_type)
        
        random.shuffle(wave_composition)
        wave_details = []
        for enemy_type_name in wave_composition:
            details = self.enemy_types[enemy_type_name].copy()
            details["type"] = enemy_type_name
            details["hp"] = int(details["hp"] * difficulty_multiplier * hp_modifier)
            details["speed"] = details["speed"] * speed_modifier
            details["path_id"] = random.choice([1, 2])
            wave_details.append(details)

        return {
            "wave_number": current_wave_num + 1,
            "enemies": wave_details,
            "interval_ms": max(200, 1000 - (current_wave_num * 25)),
            "ai_insight": f"RL Agent chose: {action.replace('_', ' ').title()}{insight_modifier_text}",
            "difficulty_factor": self.epsilon,
            "time_until_next_wave_ms": 10000 
        }

wave_ai = RLWaveAI()

@app.route('/api/get_next_wave', methods=['POST'])
def get_next_wave_api():
    data = request.json
    player_lives = data.get('lives', 10)
    player_coins = data.get('coins', 100)
    player_towers = data.get('towers', [])
    current_wave = data.get('current_wave', 0)
    current_stage = data.get('current_stage', 1)
    last_wave_stats = data.get('last_wave_stats', None)
    is_full_reset = data.get('is_full_reset', False)

    last_wave_reward = None
    if last_wave_stats and not is_full_reset:
       last_wave_reward = wave_ai.learn(last_wave_stats, player_lives, player_coins, player_towers)

    if is_full_reset:
        wave_ai.epsilon = 0.9
        print("AI: Game has been fully reset. Epsilon restored to initial value.")

    current_state = wave_ai.get_state(player_lives, player_coins, player_towers)
    chosen_action = wave_ai.choose_action(current_state)
    
    wave_ai.last_state = current_state
    wave_ai.last_action = chosen_action
    
    next_wave_data = wave_ai.generate_wave(chosen_action, current_wave, current_stage)
    
    next_wave_data['stage_number'] = current_stage

    next_wave_data['last_wave_reward'] = last_wave_reward
    
    return jsonify(next_wave_data)

def run_flask_app():
    print(f"Starting Flask API on http://localhost:{FLASK_PORT}...")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    import time
    time.sleep(1) 

    eel_port = 8080 

    try:
        print(f"Starting Eel application on port {eel_port}...")
        eel.start('index.html', mode='edge', size=(850, 750), port=eel_port)
    except (OSError, IOError) as e:
        print(f"Failed to start Eel in Edge mode: {e}. Trying Chrome...")
        try:
            eel.start('index.html', mode='chrome', size=(850, 750), port=eel_port)
        except (OSError, IOError) as e2:
            print(f"Failed to start Eel in Chrome mode: {e2}. Trying default browser...")
            eel.start('index.html', size=(850, 750), port=eel_port)

    print("Eel application closed. Saving AI progress...")
    wave_ai.save_q_table()
