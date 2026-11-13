import os
import threading
import time
from functools import lru_cache

import gymnasium as gym
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


# ---------- environment ----------
@lru_cache(maxsize=16)
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna()


class PortfolioEnv(gym.Env):
    def __init__(self, prices, window=10, initial_balance=10000.0, fee=0.001, episode_length=None):
        self.prices = prices.values
        self.returns = prices.pct_change(fill_method=None).fillna(0).values
        self.tickers = list(prices.columns)
        self.n_assets = prices.shape[1]
        self.window = window
        self.initial_balance = initial_balance
        self.fee = fee
        self.max_steps = episode_length or (len(prices) - window - 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window * self.n_assets + self.n_assets,), dtype=np.float32,
        )
        self.reset()

    def _obs(self):
        i = self.step_idx
        rets = self.returns[i - self.window + 1: i + 1].flatten()
        return np.concatenate([rets, self.weights]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.weights[0] = 1.0  # all cash (last column treated as cash)
        self.step_idx = self.window
        self.step_count = 0
        self.portfolio_value = self.initial_balance
        self.history = [self.initial_balance]
        self.weight_history = [self.weights.copy()]
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1) / 2
        new_weights = action / (action.sum() + 1e-8)
        i = self.step_idx + 1
        asset_rets = self.returns[i][: self.n_assets - 1]
        growth = np.concatenate([asset_rets, [0.0]])
        self.portfolio_value *= float(1 + np.dot(self.weights, growth))
        turnover = float(np.abs(new_weights - self.weights).sum())
        self.portfolio_value *= 1 - self.fee * turnover
        self.weights = new_weights
        self.step_idx = i
        self.history.append(self.portfolio_value)
        self.weight_history.append(self.weights.copy())
        self.step_count = getattr(self, "step_count", 0) + 1
        prev_ret = (self.history[-1] / self.history[-2]) - 1
        reward = prev_ret * 100
        terminated = False
        truncated = self.step_count >= self.max_steps or i >= len(self.prices) - 1
        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ---------- training ----------
state = {"phase": "idle"}  # phases: idle, fetching, training, acting, done, error
lock = threading.Lock()


def set_state(**kw):
    with lock:
        state.clear()
        state.update(kw)


def snapshot():
    with lock:
        return dict(state)


class ProgressCallback(BaseCallback):
    def __init__(self, prices, dates, balance, total):
        super().__init__()
        self.prices, self.dates, self.balance, self.total = prices, dates, balance, total

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            eps = self.model.ep_info_buffer
            ep_rew = float(np.mean([e["r"] for e in eps])) if eps else 0.0
            env = self.model.env.envs[0]
            base = self.prices.iloc[env.window]
            curve = [
                {"date": str(self.dates[env.window + k].date()), "rl": round(v, 2),
                 "bh": round(self.balance * float((self.prices.iloc[env.window + k] / base).mean()), 2)}
                for k, v in enumerate(env.history)
            ]
            set_state(phase="training", timesteps=self.num_timesteps,
                      ep_reward=round(ep_rew, 2), total=self.total,
                      curve=curve, tickers=env.tickers, balance=self.balance)
        return True


def run_training(p: dict):
    try:
        set_state(phase="fetching")
        prices = load_data(tuple(p["tickers"]), p["start"], p["end"])
        if prices.empty:
            raise ValueError("No data for these tickers/dates.")
        dates = prices.index

        env = DummyVecEnv([lambda: PortfolioEnv(prices, p["window"], p["balance"], p["fee"], p["episode_length"])])
        model = {"PPO": PPO, "A2C": A2C}[p["algo"]]("MlpPolicy", env, verbose=0, seed=42)
        set_state(phase="training", timesteps=0, ep_reward=0.0, total=p["timesteps"])
        model.learn(total_timesteps=p["timesteps"], callback=ProgressCallback(prices, dates, p["balance"], p["timesteps"]))

        # act out one full episode, streaming every 100 steps
        eval_env = PortfolioEnv(prices, p["window"], p["balance"], p["fee"], episode_length=None)
        first = prices.iloc[p["window"]]
        obs, _ = eval_env.reset()
        curve, weights = [], []
        done = False
        n = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            i = eval_env.step_idx
            bh = p["balance"] * float((prices.iloc[i] / first).mean())
            curve.append({"date": str(dates[i].date()), "rl": round(eval_env.portfolio_value, 2), "bh": round(bh, 2)})
            weights.append([round(float(w), 3) for w in eval_env.weights])
            if n % 100 == 0 or done:
                set_state(phase="acting", curve=curve, weights=weights, tickers=p["tickers"], balance=p["balance"])
                time.sleep(0.02)  # pace the animation
            n += 1
        set_state(phase="done", curve=curve, weights=weights, tickers=p["tickers"], balance=p["balance"])
    except Exception as e:
        set_state(phase="error", message=str(e))


def start_training(p: dict):
    with lock:
        if state.get("phase") in ("training", "acting", "fetching"):
            return False
        state.clear()
        state.update(phase="fetching")  # claim the slot before spawning
    threading.Thread(target=run_training, args=(p,), daemon=True).start()
    return True


# ---------- UI ----------
st.set_page_config(page_title="RL Portfolio Manager", page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "capy.png"), layout="wide")

ui = components.declare_component("rltrader", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"))

msg = ui(state=snapshot(), key="rltrader")

if msg and msg.get("event") == "train" and st.session_state.get("handled") != msg.get("nonce"):
    st.session_state.handled = msg.get("nonce")
    if not start_training(msg["params"]):
        st.toast("Busy — another session is already training, you'll see its progress")

if snapshot().get("phase") in ("fetching", "training", "acting"):
    time.sleep(1.0)
    st.rerun()
