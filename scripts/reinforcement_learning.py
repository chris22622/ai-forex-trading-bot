"""
Reinforcement Learning Module for Deriv Trading Bot
Implements Q-Learning and Deep Q-Network (DQN) for strategy optimization
"""

import json
import pickle
import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    # When TensorFlow is not available, we'll only use Q-Learning
    print("⚠️ TensorFlow not available. Using basic Q-learning only.")

class RLTradeEnvironment:
    """Trading environment for reinforcement learning"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.state_history: deque = deque(maxlen=max_history)
        self.action_history: deque = deque(maxlen=max_history)
        self.reward_history: deque = deque(maxlen=max_history)  # Fixed: removed extra parameter
        self.experience_replay: deque = deque(maxlen=10000)

        # State space: [price_change, rsi, ema_signal, volume, time_of_day]
        self.state_size = 5
        self.action_size = 3  # 0=HOLD, 1=BUY, 2=SELL

        self.current_state = np.zeros(self.state_size)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger('RL_Environment')
        logger.setLevel(logging.INFO)
        return logger

    def normalize_state(self, price_change: float, rsi: float, ema_signal: float,
                       volume: float = 0.5, time_of_day: float = 0.5) -> np.ndarray:
        """Normalize state variables to [-1, 1] range"""
        state = np.array([
            np.tanh(price_change * 100),  # Price change
            (rsi - 50) / 50,  # RSI normalized
            np.tanh(ema_signal),  # EMA signal
            (volume - 0.5) * 2,  # Volume
            (time_of_day - 0.5) * 2  # Time of day
        ])
        return np.clip(state, -1, 1)

    def get_state(self, indicators: Dict[str, Any], price_history: List[float]) -> np.ndarray:
        """Get current state from market data"""
        try:
            # Calculate price change
            if len(price_history) >= 2:
                price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
            else:
                price_change = 0

            # Get RSI
            rsi = indicators.get('rsi', 50) or 50

            # Get EMA signal
            ema_fast = indicators.get('ema_fast', 0) or 0
            ema_slow = indicators.get('ema_slow', 0) or 0
            ema_signal = (ema_fast - ema_slow) / max(ema_slow, 1) if ema_slow > 0 else 0

            # Time of day (0-1)
            hour = datetime.now().hour
            time_of_day = hour / 24

            # Volume proxy (using price volatility)
            if len(price_history) >= 10:
                recent_std = np.std(price_history[-10:])
                volume = min(recent_std * 1000, 1.0)  # Normalize
            else:
                volume = 0.5

            state = self.normalize_state(price_change, rsi, ema_signal, volume, time_of_day)
            self.current_state = state
            return state

        except Exception as e:
            self.logger.error(f"Error getting state: {e}")
            return np.zeros(self.state_size)

    def calculate_reward(self, action: int, trade_result: str, profit_loss: float,
                        confidence: float) -> float:
        """Calculate reward based on trade outcome"""
        base_reward = 0

        if trade_result == "WIN":
            # Reward scaled by profit and confidence
            base_reward = 1.0 + (profit_loss / 10) + (confidence - 0.5)
        elif trade_result == "LOSS":
            # Penalty scaled by loss and low confidence
            base_reward = -1.0 + (profit_loss / 10) - (0.5 - confidence)
        else:  # HOLD
            # Small penalty for holding (encourage action)
            base_reward = -0.01

        # Bonus for high confidence correct predictions
        if trade_result == "WIN" and confidence > 0.8:
            base_reward += 0.5

        # Extra penalty for high confidence wrong predictions
        if trade_result == "LOSS" and confidence > 0.8:
            base_reward -= 0.5

        return np.clip(base_reward, -3.0, 3.0)

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool = False):
        """Store experience for replay learning"""
        experience = (state, action, reward, next_state, done)
        self.experience_replay.append(experience)

        # Also store in history
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)

        self.logger.info(f"Stored experience: action={action}, reward={reward:.3f}")

class QLearningAgent:
    """Basic Q-Learning agent for trading decisions"""

    def __init__(self, state_size: int = 5, action_size: int = 3,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01

        # Q-table as dictionary (state_hash -> action_values)
        self.q_table: Dict[str, np.ndarray] = {}
        self.training_history: List[Dict[str, Any]] = []

    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to string key for Q-table"""
        # Discretize continuous state to reduce state space
        discretized = np.round(state * 10).astype(int)
        return str(discretized.tolist())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state"""
        key = self._state_to_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.random.normal(0, 0.1, self.action_size)
        return self.q_table[key]

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)

        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def update_q_value(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray):
        """Update Q-value using Q-learning formula"""
        current_q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        # Q-learning update
        target = reward + self.discount_factor * np.max(next_q_values)
        current_q_values[action] += self.learning_rate * (target - current_q_values[action])

        # Update exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'state': state.tolist(),
            'action': action,
            'reward': reward,
            'q_value': current_q_values[action],
            'exploration_rate': self.exploration_rate
        })

    def save_model(self, filepath: str):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': self.q_table,
            'exploration_rate': self.exploration_rate,
            'training_history': self.training_history[-1000:],  # Last 1000 entries
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_decay': self.exploration_decay
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load Q-table and parameters"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.q_table = model_data.get('q_table', {})
            self.exploration_rate = model_data.get('exploration_rate', self.exploration_rate)
            self.training_history = model_data.get('training_history', [])

            print(f"✅ Q-Learning model loaded with {len(self.q_table)} states")
            return True
        except Exception as e:
            print(f"❌ Failed to load Q-Learning model: {e}")
            return False

class DQNAgent:
    """Deep Q-Network agent using neural networks"""

    def __init__(self, state_size: int = 5, action_size: int = 3,
                 learning_rate: float = 0.001, memory_size: int = 10000):

        if not HAS_TENSORFLOW:
                        raise ImportError("TensorFlow required for DQN agent. Please install "
            "tensorflow or use Q-Learning instead.")

        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=memory_size)
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.batch_size = 32

        # Neural networks - only create if TensorFlow is available
        self.q_network = None
        self.target_network = None

        if HAS_TENSORFLOW:
            self.q_network = self._build_model()
            self.target_network = self._build_model()
            self.update_target_network()

    def _build_model(self):
        """Build neural network for Q-value approximation"""
        if not HAS_TENSORFLOW:
            return None

        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)

        state_batch = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state_batch, verbose=0)[0]
        return np.argmax(q_values)

    def replay_experience(self) -> float:
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        # Predict Q-values for current and next states
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)

        # Update Q-values
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += 0.95 * np.max(next_q_values[i])
            current_q_values[i][actions[i]] = target

        # Train the network
        history = self.q_network.fit(states, current_q_values,
                                   epochs=1, verbose=0, batch_size=self.batch_size)

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

        return history.history['loss'][0]

    def save_model(self, filepath: str):
        """Save DQN model"""
        self.q_network.save(f"{filepath}_dqn.h5")

        # Save parameters
        params = {
            'exploration_rate': self.exploration_rate,
            'memory_size': len(self.memory),
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        with open(f"{filepath}_params.json", 'w') as f:
            json.dump(params, f)

    def load_model(self, filepath: str):
        """Load DQN model"""
        try:
            self.q_network = keras.models.load_model(f"{filepath}_dqn.h5")
            self.target_network = keras.models.load_model(f"{filepath}_dqn.h5")

            with open(f"{filepath}_params.json", 'r') as f:
                params = json.load(f)

            self.exploration_rate = params.get('exploration_rate', self.exploration_rate)
            print("✅ DQN model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load DQN model: {e}")
            return False

class ReinforcementLearningManager:
    """Main RL manager for the trading bot"""

    def __init__(self, model_type: str = "qlearning", use_dqn: bool = None):
        self.environment = RLTradeEnvironment()

        # Auto-detect DQN availability
        if use_dqn is None:
            use_dqn = HAS_TENSORFLOW

        if use_dqn and HAS_TENSORFLOW:
            self.agent = DQNAgent()
            self.model_type = "dqn"
            print("🧠 Using Deep Q-Network (DQN) agent")
        else:
            self.agent = QLearningAgent()
            self.model_type = "qlearning"
            print("🧠 Using Q-Learning agent")

        self.current_state = None
        self.last_action = None
        self.training_enabled = True

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def get_action_recommendation(self, indicators: Dict[str, Any],
                                 price_history: List[float],
                                 training: bool = True) -> Dict[str, Any]:
        """Get trading action recommendation from RL agent"""
        try:
            # Get current state
            state = self.environment.get_state(indicators, price_history)

            # Choose action
            action = self.agent.choose_action(state, training=training)

            # Store state for next update
            self.current_state = state
            self.last_action = action

            # Convert action to trading signal
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

            return {
                "prediction": action_map[action],
                "confidence": self._calculate_confidence(state, action),
                "method": f"RL_{self.model_type.upper()}",
                "state": state.tolist(),
                "raw_action": action
            }

        except Exception as e:
            print(f"❌ RL recommendation error: {e}")
            return {
                "prediction": "HOLD",
                "confidence": 0.5,
                "method": "RL_ERROR",
                "state": [],
                "raw_action": 0
            }

    def get_action(self, indicators: Dict[str, Any],
                   price_history: List[float]) -> Dict[str, Any]:
        """Alias for get_action_recommendation - for backward compatibility"""
        return self.get_action_recommendation(indicators, price_history)

    def _calculate_confidence(self, state: np.ndarray, action: int) -> float:
        """Calculate confidence based on Q-values"""
        try:
            if self.model_type == "qlearning":
                q_values = self.agent.get_q_values(state)
            else:  # DQN
                state_batch = np.expand_dims(state, axis=0)
                q_values = self.agent.q_network.predict(state_batch, verbose=0)[0]

            # Confidence based on difference between best and second-best Q-values
            sorted_q = np.sort(q_values)
            if len(sorted_q) >= 2:
                confidence = min((sorted_q[-1] - sorted_q[-2]) + 0.5, 1.0)
            else:
                confidence = 0.5

            return max(confidence, 0.1)  # Minimum confidence

        except:
            return 0.5

    def update_with_result(self, trade_result: str, profit_loss: float,
                          confidence: float, indicators: Dict[str, Any],
                          price_history: List[float]):
        """Update RL agent with trade result"""
        if not self.training_enabled or self.current_state is None:
            return

        try:
            # Get next state
            next_state = self.environment.get_state(indicators, price_history)

            # Calculate reward
            reward = self.environment.calculate_reward(
                self.last_action, trade_result, profit_loss, confidence
            )

            # Update agent
            if self.model_type == "qlearning":
                self.agent.update_q_value(
                    self.current_state, self.last_action, reward, next_state
                )
            else:  # DQN
                self.agent.remember(
                    self.current_state, self.last_action, reward, next_state, False
                )
                loss = self.agent.replay_experience()

                # Update target network periodically
                if len(self.agent.memory) % 100 == 0:
                    self.agent.update_target_network()

            # Track episode performance
            self.current_episode_reward += reward
            self.current_episode_length += 1

            print(f"🎯 RL Update: action={self.last_action}, reward={reward:.3f}, "
                  f"result={trade_result}")

        except Exception as e:
            print(f"❌ RL update error: {e}")

    def end_episode(self):
        """End current episode and start new one"""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)

        print(f"📊 Episode ended: reward={self.current_episode_reward:.2f}, "
              f"length={self.current_episode_length}")

        # Reset episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_state = None
        self.last_action = None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL performance metrics"""
        if not self.episode_rewards:
            return {"total_episodes": 0}

        return {
            "total_episodes": len(self.episode_rewards),
            "average_reward": np.mean(self.episode_rewards),
            "best_reward": np.max(self.episode_rewards),
            "worst_reward": np.min(self.episode_rewards),
            "recent_avg_reward": np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards),
            "exploration_rate": getattr(self.agent, 'exploration_rate', 0),
            "memory_size": len(getattr(self.agent, 'memory', [])),
            "q_table_size": len(getattr(self.agent, 'q_table', {}))
        }

    def save_model(self, filepath: str = "models/rl_model"):
        """Save RL model"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            self.agent.save_model(filepath)

            # Save performance metrics
            metrics = {
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "model_type": self.model_type,
                "training_enabled": self.training_enabled
            }

            with open(f"{filepath}_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"✅ RL model saved: {filepath}")

        except Exception as e:
            print(f"❌ Failed to save RL model: {e}")

    def load_model(self, filepath: str = "models/rl_model"):
        """Load RL model"""
        try:
            success = self.agent.load_model(filepath)

            if success:
                # Load performance metrics
                try:
                    with open(f"{filepath}_metrics.json", 'r') as f:
                        metrics = json.load(f)

                    self.episode_rewards = metrics.get("episode_rewards", [])
                    self.episode_lengths = metrics.get("episode_lengths", [])
                    self.training_enabled = metrics.get("training_enabled", True)

                except:
                    pass  # Metrics file might not exist

            return success

        except Exception as e:
            print(f"❌ Failed to load RL model: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Test RL manager
    rl_manager = ReinforcementLearningManager(use_dqn=False)

    # Simulate some trading data
    dummy_indicators = {"rsi": 30, "ema_fast": 100, "ema_slow": 99}
    dummy_prices = [100, 101, 102, 101, 100]

    # Get recommendation
    recommendation = rl_manager.get_action_recommendation(dummy_indicators, dummy_prices)
    print(f"Recommendation: {recommendation}")

    # Update with result
    rl_manager.update_with_result("WIN", 5.0, 0.8, dummy_indicators, dummy_prices)

    # Get metrics
    metrics = rl_manager.get_performance_metrics()
    print(f"Metrics: {metrics}")
