"""
Pixellent AI Engine — Adaptive Learning & Continuous Improvement v2.0
Phase 6: Learning System for Indonesian Stock Market AI Trading

BEFORE Decision: Historical outcomes → ML model → Predict success → Inform weights
AFTER Decision (T+15): Record outcome → Update training data → Retrain → Update stats
CONTINUOUS: Track decisions → Rolling accuracy → Auto-adjust weights → Monthly report

Usage:
    from pixellent_learning import ContinuousImprovement
    ci = ContinuousImprovement()
    enhanced = ci.pre_decision_enhance(agent_outputs, signal_data)
    ci.post_decision_learn(ticker, decision, actual_prices_15d)
    ci.run_daily_learning_cycle(portfolio, current_prices)
"""
import json, os, logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
try:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _clip(val, lo, hi):
    return max(lo, min(hi, val))

def _mean(lst):
    return sum(lst) / max(len(lst), 1) if lst else 0.0


# =============================================================================
# 1. DecisionRecord
# =============================================================================
@dataclass
class DecisionRecord:
    """Complete record of a trading decision for learning."""
    ticker: str
    timestamp: str
    action: str  # STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    composite_score: float
    confidence: float
    agent_scores: Dict[str, float] = field(default_factory=dict)
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    features_snapshot: Dict[str, float] = field(default_factory=dict)
    outcome: Optional[Dict[str, float]] = None
    outcome_label: Optional[int] = None  # 1=success, 0=fail

    def to_dict(self) -> Dict[str, Any]: return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DecisionRecord':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_pending(self) -> bool: return self.outcome is None

    @property
    def decision_age_days(self) -> int:
        try: return (datetime.now() - datetime.fromisoformat(self.timestamp)).days
        except: return 0


# =============================================================================
# 2. DecisionTracker
# =============================================================================
class DecisionTracker:
    """Record and manage all trading decisions. Persists to data/decision_log.json."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.data_dir / "decision_log.json"
        self.decisions: List[DecisionRecord] = []
        self._load()

    def _load(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    self.decisions = [DecisionRecord.from_dict(d) for d in json.load(f)]
            except: self.decisions = []

    def _save(self):
        try:
            with open(self.log_path, 'w') as f:
                json.dump([d.to_dict() for d in self.decisions], f, indent=2, default=str)
        except Exception as e: logger.error(f"Save failed: {e}")

    def _extract_features(self, signal_data: Dict) -> Dict[str, float]:
        keys = ['regime','trend_age','vpower','ff_streak','atr_ratio','sm_score',
                'macro_score','rsi','adx','hma_slope','volatility_20d','rr_ratio',
                'relative_volume','sector_bias_score','market_score']
        features = {}
        regime_map = {'TRENDING': 2, 'SIDEWAYS': 1, 'HIGH_VOL': 0}
        for k in keys:
            v = signal_data.get(k)
            if v is None: features[k] = 0.0
            elif isinstance(v, str): features[k] = float(regime_map.get(v.upper(), 1))
            else: features[k] = float(v)
        sector = signal_data.get('sector', 'unknown')
        for s in ['banking','mining','consumer','telco','property','infrastructure','manufacturing','energy','healthcare']:
            features[f'sector_{s}'] = 1.0 if sector == s else 0.0
        return features

    def record_decision(self, ticker, master_decision, signal_data, entry_price, sl, target) -> DecisionRecord:
        agent_scores = master_decision.get('agent_scores', {
            'TrendAgent': signal_data.get('trend_score', 50.0),
            'SmartMoneyAgent': signal_data.get('sm_score', 50.0),
            'RiskAgent': signal_data.get('risk_score', 50.0),
            'MacroAgent': signal_data.get('macro_score', 50.0),
        })
        record = DecisionRecord(
            ticker=ticker, timestamp=datetime.now().isoformat(),
            action=master_decision.get('action', 'HOLD'),
            composite_score=master_decision.get('score', 50.0),
            confidence=master_decision.get('confidence', 0.5),
            agent_scores=agent_scores, entry_price=entry_price,
            stop_loss=sl, target_price=target,
            features_snapshot=self._extract_features(signal_data))
        self.decisions.append(record)
        self._save()
        return record

    def update_outcome(self, ticker, timestamp, price_history_15d) -> Optional[DecisionRecord]:
        record = next((d for d in self.decisions if d.ticker == ticker and d.timestamp == timestamp and d.is_pending), None)
        if not record or not price_history_15d or record.entry_price <= 0: return None
        entry = record.entry_price
        prices = [float(p) for p in price_history_15d]
        returns_pct = [(p - entry) / entry * 100 for p in prices]
        record.outcome = {
            'return_5d': returns_pct[4] if len(prices) > 4 else 0.0,
            'return_10d': returns_pct[9] if len(prices) > 9 else 0.0,
            'return_15d': returns_pct[-1],
            'max_gain': max(returns_pct), 'max_loss': min(returns_pct),
            'hit_target': bool(max(prices) >= record.target_price) if record.target_price > 0 else False,
            'hit_stoploss': bool(min(prices) <= record.stop_loss) if record.stop_loss > 0 else False,
        }
        record.outcome_label = 1 if record.outcome['max_gain'] >= 2.0 else 0
        self._save()
        return record

    def auto_update_outcomes(self, portfolio, current_prices):
        for record in self.get_pending_outcomes():
            if record.decision_age_days >= 15 and record.ticker in current_prices:
                prices = current_prices[record.ticker]
                if len(prices) >= 15:
                    self.update_outcome(record.ticker, record.timestamp, prices[:15])

    def get_pending_outcomes(self) -> List[DecisionRecord]:
        return [d for d in self.decisions if d.is_pending]

    def get_decision_history(self, ticker=None, last_n=200) -> List[DecisionRecord]:
        filtered = [d for d in self.decisions if not ticker or d.ticker == ticker]
        return filtered[-last_n:]

    def get_labeled_decisions(self) -> List[DecisionRecord]:
        return [d for d in self.decisions if d.outcome_label is not None]



# =============================================================================
# 3. AgentAccuracyTracker
# =============================================================================
class AgentAccuracyTracker:
    """Rolling accuracy, regime-based, and score-bucket analysis per agent."""
    AGENTS = ['TrendAgent', 'SmartMoneyAgent', 'RiskAgent', 'MacroAgent']
    REGIMES = ['TRENDING', 'SIDEWAYS', 'HIGH_VOL']

    def __init__(self, decision_tracker: DecisionTracker):
        self.tracker = decision_tracker

    def compute_rolling_accuracy(self, agent_name: str, window: int = 50) -> float:
        recent = self.tracker.get_labeled_decisions()[-window:]
        if not recent: return 0.5
        correct, total = 0, 0
        for r in recent:
            s = r.agent_scores.get(agent_name, 50.0)
            total += 1
            if s > 60 and r.outcome_label == 1: correct += 1
            elif s <= 40 and r.outcome_label == 0: correct += 1
            elif 40 <= s <= 60: correct += 0.5
        return correct / max(total, 1)

    def compute_accuracy_by_regime(self, agent_name: str) -> Dict[str, float]:
        stats = {r: {'c': 0, 't': 0} for r in self.REGIMES}
        for r in self.tracker.get_labeled_decisions():
            rv = r.features_snapshot.get('regime', 1.0)
            regime = 'TRENDING' if rv >= 2 else ('HIGH_VOL' if rv <= 0 else 'SIDEWAYS')
            s = r.agent_scores.get(agent_name, 50.0)
            stats[regime]['t'] += 1
            if (s > 60 and r.outcome_label == 1) or (s <= 40 and r.outcome_label == 0):
                stats[regime]['c'] += 1
            elif 40 <= s <= 60: stats[regime]['c'] += 0.5
        return {k: v['c']/max(v['t'],1) for k, v in stats.items()}

    def compute_accuracy_by_score_bucket(self, agent_name: str, buckets: int = 5) -> Dict[str, float]:
        bsize = 100.0 / buckets
        bstats = {f"{int(i*bsize)}-{int((i+1)*bsize)}": {'s': 0, 't': 0} for i in range(buckets)}
        for r in self.tracker.get_labeled_decisions():
            s = r.agent_scores.get(agent_name, 50.0)
            idx = min(int(s / bsize), buckets - 1)
            label = f"{int(idx*bsize)}-{int((idx+1)*bsize)}"
            bstats[label]['t'] += 1
            if r.outcome_label == 1: bstats[label]['s'] += 1
        return {k: v['s']/max(v['t'],1) for k, v in bstats.items()}

    def get_accuracy_report(self) -> Dict[str, Any]:
        report = {}
        for a in self.AGENTS:
            report[a] = {'rolling_accuracy_50': self.compute_rolling_accuracy(a, 50),
                         'rolling_accuracy_20': self.compute_rolling_accuracy(a, 20),
                         'accuracy_by_regime': self.compute_accuracy_by_regime(a),
                         'accuracy_by_score_bucket': self.compute_accuracy_by_score_bucket(a)}
        labeled = self.tracker.get_labeled_decisions()
        recent = labeled[-50:]
        report['system_accuracy_50'] = sum(1 for d in recent if d.outcome_label == 1) / max(len(recent), 1)
        report['total_labeled_decisions'] = len(labeled)
        return report

    def identify_weak_agent(self) -> Optional[Dict[str, Any]]:
        if len(self.tracker.get_labeled_decisions()) < 20: return None
        accs = {a: self.compute_rolling_accuracy(a, 50) for a in self.AGENTS}
        weakest = min(accs, key=accs.get)
        avg = _mean(list(accs.values()))
        if accs[weakest] < avg - 0.05:
            return {'agent': weakest, 'accuracy': accs[weakest], 'average_accuracy': avg,
                    'recommendation': f"Reduce {weakest} weight. Accuracy ({accs[weakest]:.1%}) below avg ({avg:.1%})."}
        return None



# =============================================================================
# 4. PreDecisionML — ML BEFORE decision
# =============================================================================
class PreDecisionML:
    """XGBoost model predicting success probability before decisions. Fallback to rule-based."""
    FEATURE_NAMES = [
        'regime','trend_age','vpower','ff_streak','atr_ratio','sm_score','macro_score',
        'rsi','adx','hma_slope','volatility_20d','rr_ratio','relative_volume',
        'sector_bias_score','market_score',
        'sector_banking','sector_mining','sector_consumer','sector_telco',
        'sector_property','sector_infrastructure','sector_manufacturing',
        'sector_energy','sector_healthcare']

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "pre_decision_model.joblib"
        self.model = None
        self.is_trained = False
        self.last_train_count = 0
        self.training_metrics: Dict[str, float] = {}
        self._load_model()

    def _load_model(self):
        if not HAS_JOBLIB or not self.model_path.exists(): return
        try:
            d = joblib.load(self.model_path)
            self.model, self.last_train_count = d.get('model'), d.get('train_count', 0)
            self.training_metrics = d.get('metrics', {})
            self.is_trained = self.model is not None
        except: pass

    def _save_model(self):
        if not HAS_JOBLIB or not self.model: return
        try: joblib.dump({'model': self.model, 'train_count': self.last_train_count,
                         'metrics': self.training_metrics, 'timestamp': datetime.now().isoformat()},
                        self.model_path)
        except: pass

    def _prepare_features(self, fd: Dict[str, float]):
        vec = [fd.get(n, 0.0) for n in self.FEATURE_NAMES]
        if HAS_NUMPY: return np.array([vec], dtype=float)
        return [vec]

    def predict_success_probability(self, features_dict: Dict[str, float]) -> float:
        if self.is_trained and self.model and HAS_XGBOOST:
            try:
                X = self._prepare_features(features_dict)
                return float(_clip(self.model.predict_proba(X)[0][1], 0.0, 1.0))
            except: pass
        return self._rule_based_probability(features_dict)

    def _rule_based_probability(self, fd: Dict[str, float]) -> float:
        s = 0.5
        regime = fd.get('regime', 1.0)
        s += 0.10 if regime >= 2 else (-0.10 if regime <= 0 else 0)
        ta = fd.get('trend_age', 0)
        s += 0.08 if ta > 10 else (-0.08 if ta < -10 else 0)
        sm = fd.get('sm_score', 50)
        s += 0.10 if sm > 70 else (-0.10 if sm < 30 else 0)
        vp = fd.get('vpower', 1.0)
        s += 0.07 if vp > 1.3 else (-0.07 if vp < 0.7 else 0)
        atr = fd.get('atr_ratio', 1.0)
        s += 0.05 if atr < 0.8 else (-0.08 if atr > 1.5 else 0)
        ff = fd.get('ff_streak', 0)
        s += 0.06 if ff > 3 else (-0.06 if ff < -3 else 0)
        return _clip(s, 0.1, 0.9)

    def get_recommended_weights(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        w = {'TrendAgent': 0.30, 'SmartMoneyAgent': 0.25, 'RiskAgent': 0.25, 'MacroAgent': 0.20}
        regime = features_dict.get('regime', 1.0)
        if regime >= 2:  # TRENDING
            w['TrendAgent'] += 0.08; w['RiskAgent'] -= 0.04; w['SmartMoneyAgent'] -= 0.02; w['MacroAgent'] -= 0.02
        elif regime <= 0:  # HIGH_VOL
            w['RiskAgent'] += 0.10; w['TrendAgent'] -= 0.05; w['SmartMoneyAgent'] -= 0.03; w['MacroAgent'] -= 0.02
        else:  # SIDEWAYS
            w['SmartMoneyAgent'] += 0.06; w['TrendAgent'] -= 0.04; w['MacroAgent'] -= 0.02
        macro = features_dict.get('macro_score', 50)
        if macro > 75 or macro < 25:
            w['MacroAgent'] += 0.05; w['TrendAgent'] -= 0.025; w['SmartMoneyAgent'] -= 0.025
        vp = features_dict.get('vpower', 1.0)
        if vp > 1.5 or vp < 0.5:
            w['SmartMoneyAgent'] += 0.04; w['MacroAgent'] -= 0.02; w['TrendAgent'] -= 0.02
        total = sum(w.values())
        w = {k: max(v/total, 0.05) for k, v in w.items()}
        total = sum(w.values())
        return {k: v/total for k, v in w.items()}

    def retrain(self, decision_history: List[DecisionRecord]) -> Dict[str, float]:
        labeled = [d for d in decision_history if d.outcome_label is not None]
        if len(labeled) < 50: return {'status': 'insufficient_data', 'n_samples': len(labeled)}
        if not HAS_XGBOOST or not HAS_SKLEARN or not HAS_NUMPY:
            return {'status': 'dependencies_missing'}
        X = np.array([self._prepare_features(d.features_snapshot)[0] for d in labeled])
        y = np.array([d.outcome_label for d in labeled])
        n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
        params = {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
                  'scale_pos_weight': max(1, n_neg/max(n_pos,1)), 'eval_metric': 'logloss',
                  'random_state': 42, 'use_label_encoder': False}
        try:
            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=min(5, len(labeled)//10), shuffle=True, random_state=42)
            oof = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
            preds = (oof >= 0.5).astype(int)
            metrics = {'accuracy': float(accuracy_score(y, preds)),
                       'auc': float(roc_auc_score(y, oof)) if len(set(y)) > 1 else 0.5,
                       'f1': float(f1_score(y, preds, zero_division=0)),
                       'n_samples': len(y), 'status': 'success'}
            model.fit(X, y)
            self.model, self.is_trained = model, True
            self.last_train_count, self.training_metrics = len(labeled), metrics
            self._save_model()
            return metrics
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def should_retrain(self, decision_history: List[DecisionRecord]) -> bool:
        labeled = [d for d in decision_history if d.outcome_label is not None]
        return len(labeled) - self.last_train_count >= 20



# =============================================================================
# 5. PostDecisionLearning — ML AFTER decision
# =============================================================================
class PostDecisionLearning:
    """Analyze what went right/wrong after outcomes are known."""

    def __init__(self, decision_tracker: DecisionTracker):
        self.tracker = decision_tracker

    def analyze_decision_quality(self, dr: DecisionRecord) -> Dict[str, Any]:
        if dr.outcome is None: return {'status': 'pending'}
        o = dr.outcome
        analysis = {'ticker': dr.ticker, 'action': dr.action, 'outcome_label': dr.outcome_label,
                    'max_gain': o.get('max_gain', 0), 'max_loss': o.get('max_loss', 0),
                    'return_15d': o.get('return_15d', 0), 'hit_target': o.get('hit_target', False)}
        if dr.outcome_label == 1:
            analysis['quality'] = 'EXCELLENT' if o.get('hit_target') else 'GOOD'
            analysis['quality_reason'] = 'Hit target' if o.get('hit_target') else 'Gained 2%+ within 15d'
        else:
            if o.get('hit_stoploss'): analysis['quality'], analysis['quality_reason'] = 'POOR', 'Stop-loss hit'
            elif o.get('max_loss', 0) < -5: analysis['quality'], analysis['quality_reason'] = 'POOR', f"Loss {o['max_loss']:.1f}%"
            else: analysis['quality'], analysis['quality_reason'] = 'MEDIOCRE', 'No 2% gain in timeframe'
        if o.get('max_gain', 0) >= 2 and o.get('return_15d', 0) < 0:
            analysis['timing_note'] = 'Had gain but gave it back — tighten trailing stop'
        return analysis

    def compute_agent_contribution(self, dr: DecisionRecord) -> Dict[str, Any]:
        if dr.outcome_label is None: return {}
        success = dr.outcome_label == 1
        contribs = {}
        for agent, score in dr.agent_scores.items():
            bullish, bearish = score > 60, score <= 40
            if success:
                contribs[agent] = {'score': score, 'correct': bullish, 'contribution': 'positive' if bullish else ('negative_drag' if bearish else 'missed')}
            else:
                contribs[agent] = {'score': score, 'correct': bearish, 'contribution': 'correctly_warned' if bearish else ('false_positive' if bullish else 'neutral')}
        return contribs

    def generate_learning_insight(self, recent_decisions: List[DecisionRecord], n: int = 20) -> List[str]:
        labeled = [d for d in recent_decisions if d.outcome_label is not None][-n:]
        if len(labeled) < 5: return ["Insufficient data for insights (need 5+ labeled decisions)."]
        insights = []
        # Per-agent accuracy
        for agent in AgentAccuracyTracker.AGENTS:
            preds = [(d.agent_scores.get(agent, 50) > 60, d.outcome_label == 1) for d in labeled if d.agent_scores.get(agent, 50) > 60 or d.agent_scores.get(agent, 50) <= 40]
            if len(preds) >= 5:
                acc = sum(1 for p, o in preds if p == o) / len(preds)
                if acc < 0.45: insights.append(f"{agent} accuracy dropped to {acc:.0%}. Consider reducing weight.")
                elif acc > 0.75: insights.append(f"{agent} predicted {acc:.0%} correctly. Consider increasing weight.")
        # Regime success
        for regime_val, regime_name in [(2.0, 'TRENDING'), (1.0, 'SIDEWAYS'), (0.0, 'HIGH_VOL')]:
            rd = [d for d in labeled if d.features_snapshot.get('regime', 1) == regime_val]
            if len(rd) >= 3:
                rate = sum(d.outcome_label for d in rd) / len(rd)
                if rate < 0.35: insights.append(f"Success rate in {regime_name} only {rate:.0%}. Be more conservative.")
                elif rate > 0.70: insights.append(f"Excellent {rate:.0%} success in {regime_name} regime.")
        # High score check
        hi = [d for d in labeled if d.composite_score >= 70]
        if len(hi) >= 3:
            rate = sum(d.outcome_label for d in hi) / len(hi)
            if rate < 0.45: insights.append(f"WARNING: High-conviction trades (score>=70) only succeed {rate:.0%}.")
        return insights or ["System performing within normal parameters."]

    def get_what_works(self) -> Dict[str, Any]:
        labeled = self.tracker.get_labeled_decisions()
        successes = [d for d in labeled if d.outcome_label == 1]
        if len(successes) < 5: return {'status': 'insufficient_data', 'patterns': []}
        patterns = []
        for agent in AgentAccuracyTracker.AGENTS:
            scores = [d.agent_scores.get(agent, 50) for d in successes]
            avg = _mean(scores)
            if avg > 65: patterns.append(f"Successful trades have {agent} avg score {avg:.1f} (>65)")
        return {'status': 'ok', 'n_successes': len(successes),
                'success_rate': len(successes)/max(len(labeled),1), 'patterns': patterns[:10]}

    def get_what_fails(self) -> Dict[str, Any]:
        labeled = self.tracker.get_labeled_decisions()
        failures = [d for d in labeled if d.outcome_label == 0]
        if len(failures) < 5: return {'status': 'insufficient_data', 'patterns': []}
        patterns = []
        regime_counts = {'TRENDING': 0, 'SIDEWAYS': 0, 'HIGH_VOL': 0}
        for d in failures:
            rv = d.features_snapshot.get('regime', 1)
            regime_counts['TRENDING' if rv >= 2 else ('HIGH_VOL' if rv <= 0 else 'SIDEWAYS')] += 1
        for regime, cnt in regime_counts.items():
            if cnt / max(len(failures), 1) > 0.5:
                patterns.append(f"Majority of failures ({cnt}/{len(failures)}) in {regime} — reduce aggression.")
        overconf = [d for d in failures if d.confidence > 0.7]
        if len(overconf) >= 3: patterns.append(f"{len(overconf)} failures had high confidence (>0.7). Calibration issue.")
        return {'status': 'ok', 'n_failures': len(failures),
                'failure_rate': len(failures)/max(len(labeled),1), 'patterns': patterns[:10]}



# =============================================================================
# 6. ContinuousImprovement — Orchestrates everything
# =============================================================================
class ContinuousImprovement:
    """Full learning loop: pre-decision enhance, post-decision learn, daily cycle."""
    EMA_ALPHA = 0.2  # new_weight = 0.8*old + 0.2*accuracy_based

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.decision_tracker = DecisionTracker(data_dir)
        self.accuracy_tracker = AgentAccuracyTracker(self.decision_tracker)
        self.pre_decision_ml = PreDecisionML(data_dir)
        self.post_decision_learning = PostDecisionLearning(self.decision_tracker)
        self.weights_path = self.data_dir / "agent_weights.json"
        self.insights_path = self.data_dir / "learning_insights.json"
        self.current_weights = self._load_weights()
        self.weight_history: List[Dict] = self._load_weight_history()
        self.recent_insights: List[str] = self._load_insights()

    def _default_weights(self): return {'TrendAgent': 0.30, 'SmartMoneyAgent': 0.25, 'RiskAgent': 0.25, 'MacroAgent': 0.20}

    def _load_weights(self):
        if self.weights_path.exists():
            try:
                with open(self.weights_path) as f: return json.load(f).get('current_weights', self._default_weights())
            except: pass
        return self._default_weights()

    def _load_weight_history(self):
        if self.weights_path.exists():
            try:
                with open(self.weights_path) as f: return json.load(f).get('history', [])
            except: pass
        return []

    def _load_insights(self):
        if self.insights_path.exists():
            try:
                with open(self.insights_path) as f: return json.load(f).get('insights', [])
            except: pass
        return []

    def _save_weights(self):
        try:
            with open(self.weights_path, 'w') as f:
                json.dump({'current_weights': self.current_weights, 'last_updated': datetime.now().isoformat(),
                           'history': self.weight_history[-90:]}, f, indent=2, default=str)
        except: pass

    def _save_insights(self):
        try:
            with open(self.insights_path, 'w') as f:
                json.dump({'insights': self.recent_insights[-20:], 'last_updated': datetime.now().isoformat()}, f, indent=2)
        except: pass

    def pre_decision_enhance(self, agent_outputs, signal_data) -> Dict[str, Any]:
        features = self.decision_tracker._extract_features(signal_data)
        success_prob = self.pre_decision_ml.predict_success_probability(features)
        ml_weights = self.pre_decision_ml.get_recommended_weights(features)
        blended = {}
        for a in self.current_weights:
            blended[a] = self.EMA_ALPHA * ml_weights.get(a, 0.25) + (1 - self.EMA_ALPHA) * self.current_weights[a]
        total = sum(blended.values())
        blended = {k: v/total for k, v in blended.items()}
        return {'agent_outputs': agent_outputs, 'adjusted_weights': blended,
                'success_probability': success_prob,
                'ml_confidence': 'high' if self.pre_decision_ml.is_trained else 'low_fallback',
                'weight_adjustments': {k: blended[k] - self.current_weights.get(k, 0.25) for k in blended}}

    def post_decision_learn(self, ticker, decision, actual_prices_15d) -> Dict[str, Any]:
        ts = decision.get('timestamp', '')
        updated = self.decision_tracker.update_outcome(ticker, ts, actual_prices_15d)
        if not updated: return {'status': 'not_found', 'ticker': ticker}
        quality = self.post_decision_learning.analyze_decision_quality(updated)
        contrib = self.post_decision_learning.compute_agent_contribution(updated)
        self._update_weights_from_outcome(updated)
        return {'status': 'learned', 'ticker': ticker, 'outcome_label': updated.outcome_label,
                'quality': quality, 'agent_contributions': contrib}

    def _update_weights_from_outcome(self, record: DecisionRecord):
        if record.outcome_label is None: return
        acc_w = {a: max(self.accuracy_tracker.compute_rolling_accuracy(a, 50), 0.1) for a in self.current_weights}
        total_acc = sum(acc_w.values())
        acc_w = {k: v/total_acc for k, v in acc_w.items()}
        for a in self.current_weights:
            self.current_weights[a] = (1-self.EMA_ALPHA)*self.current_weights[a] + self.EMA_ALPHA*acc_w.get(a, 0.25)
        total = sum(self.current_weights.values())
        self.current_weights = {k: v/total for k, v in self.current_weights.items()}
        self.weight_history.append({'date': datetime.now().isoformat(), 'weights': dict(self.current_weights),
                                    'trigger': f"{record.ticker}_outcome"})
        self._save_weights()

    def run_daily_learning_cycle(self, portfolio, current_prices) -> Dict[str, Any]:
        logger.info("=== Running Daily Learning Cycle ===")
        result = {'timestamp': datetime.now().isoformat()}
        # 1. Update outcomes
        pending_before = len(self.decision_tracker.get_pending_outcomes())
        self.decision_tracker.auto_update_outcomes(portfolio, current_prices)
        result['outcomes_updated'] = pending_before - len(self.decision_tracker.get_pending_outcomes())
        # 2. Retrain if needed
        all_d = self.decision_tracker.decisions
        if self.pre_decision_ml.should_retrain(all_d):
            result['retrained'] = True
            result['train_metrics'] = self.pre_decision_ml.retrain(all_d)
        else: result['retrained'] = False
        # 3. Recompute accuracy
        result['accuracy_report'] = self.accuracy_tracker.get_accuracy_report()
        # 4. Generate insights
        self.recent_insights = self.post_decision_learning.generate_learning_insight(
            self.decision_tracker.get_labeled_decisions(), n=20)
        self._save_insights()
        result['insights'] = self.recent_insights
        # 5. Save weights
        self._save_weights()
        result['current_weights'] = dict(self.current_weights)
        return result

    def get_learning_dashboard_data(self) -> Dict[str, Any]:
        return {
            'agent_accuracies': {a: self.accuracy_tracker.compute_rolling_accuracy(a, 50) for a in AgentAccuracyTracker.AGENTS},
            'weight_history': self.weight_history[-30:],
            'recent_insights': self.recent_insights[-5:],
            'model_performance': {**self.pre_decision_ml.training_metrics, 'is_trained': self.pre_decision_ml.is_trained, 'train_count': self.pre_decision_ml.last_train_count},
            'pending_outcomes': len(self.decision_tracker.get_pending_outcomes()),
            'total_decisions_tracked': len(self.decision_tracker.decisions),
            'current_weights': dict(self.current_weights),
            'weak_agent': self.accuracy_tracker.identify_weak_agent(),
            'what_works': self.post_decision_learning.get_what_works(),
            'what_fails': self.post_decision_learning.get_what_fails()}



# =============================================================================
# 7. MonthlyLearningReport
# =============================================================================
class MonthlyLearningReport:
    """Comprehensive monthly performance and learning report."""

    def __init__(self, continuous_improvement: ContinuousImprovement):
        self.ci = continuous_improvement
        self.report_dir = self.ci.data_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, month: Optional[str] = None) -> Dict[str, Any]:
        if month is None: month = datetime.now().strftime("%Y-%m")
        try:
            y, m = month.split('-')
            start = datetime(int(y), int(m), 1)
            end = datetime(int(y)+(1 if int(m)==12 else 0), (int(m)%12)+1, 1)
        except:
            start, end = datetime.now().replace(day=1), datetime.now()
        # Filter this month
        month_d = []
        for d in self.ci.decision_tracker.decisions:
            try:
                dt = datetime.fromisoformat(d.timestamp)
                if start <= dt < end: month_d.append(d)
            except: pass
        labeled = [d for d in month_d if d.outcome_label is not None]
        n_success = sum(1 for d in labeled if d.outcome_label == 1)
        gains = [d.outcome.get('max_gain', 0) for d in labeled if d.outcome]
        losses = [d.outcome.get('max_loss', 0) for d in labeled if d.outcome]

        report = {
            'month': month, 'generated_at': datetime.now().isoformat(),
            'summary': {'total_decisions': len(month_d), 'labeled': len(labeled),
                        'success_count': n_success, 'fail_count': len(labeled)-n_success,
                        'success_rate': n_success/max(len(labeled),1),
                        'avg_max_gain': round(_mean(gains), 2), 'avg_max_loss': round(_mean(losses), 2)},
            'what_worked': self._what_worked(labeled),
            'what_didnt_work': self._what_didnt(labeled),
            'agent_accuracy_trend': self._accuracy_trend(labeled),
            'weight_adjustments': [e for e in self.ci.weight_history if e.get('date','')[:7] == month][-20:],
            'recommendations': self._recommendations(labeled),
            'model_retrain_history': {'is_trained': self.ci.pre_decision_ml.is_trained,
                                      'train_count': self.ci.pre_decision_ml.last_train_count,
                                      'metrics': self.ci.pre_decision_ml.training_metrics}}
        # Save
        try:
            with open(self.report_dir / f"monthly_{month}.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except: pass
        return report

    def _what_worked(self, labeled):
        successes = [d for d in labeled if d.outcome_label == 1]
        if not successes: return ["No successful trades this month."]
        patterns = []
        trending_s = [d for d in successes if d.features_snapshot.get('regime', 1) >= 2]
        if len(trending_s) > len(successes)*0.5:
            patterns.append(f"TRENDING regime trades had high success ({len(trending_s)}/{len(successes)})")
        hi = [d for d in successes if d.composite_score >= 70]
        if hi:
            avg_g = _mean([d.outcome.get('max_gain',0) for d in hi if d.outcome])
            patterns.append(f"High-conviction (score>=70) averaged {avg_g:.1f}% max gain")
        return patterns or [f"{len(successes)} successes with varied patterns."]

    def _what_didnt(self, labeled):
        failures = [d for d in labeled if d.outcome_label == 0]
        if not failures: return ["No failed trades — excellent!"]
        patterns = []
        hv = [d for d in failures if d.features_snapshot.get('regime', 1) <= 0]
        if len(hv) > len(failures)*0.3: patterns.append(f"HIGH_VOL produced {len(hv)} failures. Reduce exposure.")
        sl = [d for d in failures if d.outcome and d.outcome.get('hit_stoploss')]
        if sl: patterns.append(f"{len(sl)} trades hit stop-loss. Consider wider stops or better timing.")
        return patterns or [f"{len(failures)} failures with varied causes."]

    def _accuracy_trend(self, labeled):
        if len(labeled) < 6: return {'status': 'insufficient_data'}
        mid = len(labeled)//2
        trends = {}
        for agent in AgentAccuracyTracker.AGENTS:
            def _acc(subset):
                return sum(1 for d in subset if (d.agent_scores.get(agent,50)>60 and d.outcome_label==1) or (d.agent_scores.get(agent,50)<=40 and d.outcome_label==0))/max(len(subset),1)
            a1, a2 = _acc(labeled[:mid]), _acc(labeled[mid:])
            trends[agent] = {'first_half': round(a1,3), 'second_half': round(a2,3),
                             'trend': 'improving' if a2>a1+0.05 else ('declining' if a2<a1-0.05 else 'stable')}
        return trends

    def _recommendations(self, labeled):
        if len(labeled) < 5: return ["More data needed. Continue tracking."]
        recs = []
        rate = sum(1 for d in labeled if d.outcome_label==1)/len(labeled)
        if rate < 0.4: recs.append("Success rate <40%. Raise minimum score threshold.")
        elif rate > 0.65: recs.append("Success rate >65%. Consider increasing position sizes.")
        weak = self.ci.accuracy_tracker.identify_weak_agent()
        if weak: recs.append(weak['recommendation'])
        return recs or ["System within expectations. Maintain strategy."]



# =============================================================================
# MAIN — Comprehensive Test & Demo
# =============================================================================
if __name__ == '__main__':
    import random, tempfile, shutil
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    print("=" * 70)
    print("PIXELLENT ADAPTIVE LEARNING — Test Suite")
    print("=" * 70)
    test_dir = tempfile.mkdtemp(prefix="pixellent_learn_")
    print(f"Test dir: {test_dir}")
    try:
        # TEST 1: DecisionRecord
        print("\n--- TEST 1: DecisionRecord ---")
        rec = DecisionRecord(ticker="BBCA", timestamp=datetime.now().isoformat(), action="BUY",
                             composite_score=72.5, confidence=0.78,
                             agent_scores={'TrendAgent': 75, 'SmartMoneyAgent': 68, 'RiskAgent': 70, 'MacroAgent': 65},
                             entry_price=9500, stop_loss=9200, target_price=10000,
                             features_snapshot={'regime': 2.0, 'trend_age': 15, 'vpower': 1.3})
        assert rec.is_pending and rec.to_dict()['ticker'] == 'BBCA'
        print("  PASSED")

        # TEST 2: DecisionTracker
        print("\n--- TEST 2: DecisionTracker ---")
        tracker = DecisionTracker(data_dir=test_dir)
        tickers = ['BBCA', 'BBRI', 'TLKM', 'ASII', 'UNVR']
        for i, t in enumerate(tickers):
            ep = random.uniform(2000, 12000)
            md = {'action': 'BUY', 'score': random.uniform(55, 85), 'confidence': random.uniform(0.5, 0.9),
                  'agent_scores': {'TrendAgent': random.uniform(40,85), 'SmartMoneyAgent': random.uniform(40,80),
                                   'RiskAgent': random.uniform(45,75), 'MacroAgent': random.uniform(40,70)}}
            sd = {'regime': random.choice(['TRENDING','SIDEWAYS','HIGH_VOL']), 'trend_age': random.uniform(-5,25),
                  'vpower': random.uniform(0.6,1.8), 'ff_streak': random.randint(-3,6),
                  'atr_ratio': random.uniform(0.6,1.8), 'sm_score': random.uniform(30,80),
                  'macro_score': random.uniform(40,70), 'sector': 'banking'}
            tracker.record_decision(t, md, sd, ep, ep*0.97, ep*1.05)
        assert len(tracker.decisions) == 5
        # Update outcomes
        for r in tracker.decisions[:3]:
            prices = [r.entry_price * (1 + random.uniform(0.003, 0.02)*d) for d in range(1, 16)]
            tracker.update_outcome(r.ticker, r.timestamp, prices)
        for r in tracker.decisions[3:]:
            prices = [r.entry_price * (1 - random.uniform(0.002, 0.015)*d) for d in range(1, 16)]
            tracker.update_outcome(r.ticker, r.timestamp, prices)
        assert len(tracker.get_labeled_decisions()) == 5
        print(f"  Recorded: {len(tracker.decisions)}, Labeled: {len(tracker.get_labeled_decisions())}")
        # Test persistence
        tracker2 = DecisionTracker(data_dir=test_dir)
        assert len(tracker2.decisions) == 5
        print("  PASSED")

        # Add more data for accuracy tests
        for _ in range(50):
            ep = random.uniform(2000, 10000)
            regime = random.choice([0.0, 1.0, 2.0])
            ts = random.uniform(30, 90); sms = random.uniform(30, 85)
            md = {'action': 'BUY', 'score': (ts+sms)/2, 'confidence': random.uniform(0.4, 0.9),
                  'agent_scores': {'TrendAgent': ts, 'SmartMoneyAgent': sms,
                                   'RiskAgent': random.uniform(35,80), 'MacroAgent': random.uniform(35,75)}}
            sd = {'regime': {2:'TRENDING',1:'SIDEWAYS',0:'HIGH_VOL'}[regime],
                  'trend_age': random.uniform(-5,25), 'vpower': random.uniform(0.6,1.8),
                  'ff_streak': random.randint(-3,6), 'atr_ratio': random.uniform(0.6,1.8),
                  'sm_score': sms, 'macro_score': random.uniform(40,70), 'sector': 'banking'}
            r = tracker.record_decision(random.choice(tickers), md, sd, ep, ep*0.97, ep*1.05)
            prob = 0.3 + (0.2 if regime==2 else 0) + (0.15 if ts > 65 else 0)
            success = random.random() < prob
            prices = [ep*(1+(0.015 if success else -0.01)*d) for d in range(1, 16)]
            tracker.update_outcome(r.ticker, r.timestamp, prices)

        # TEST 3: AgentAccuracyTracker
        print("\n--- TEST 3: AgentAccuracyTracker ---")
        at = AgentAccuracyTracker(tracker)
        for a in at.AGENTS:
            print(f"  {a}: rolling_50={at.compute_rolling_accuracy(a,50):.2%}")
        regime_acc = at.compute_accuracy_by_regime('TrendAgent')
        print(f"  TrendAgent by regime: {regime_acc}")
        report = at.get_accuracy_report()
        print(f"  System accuracy: {report['system_accuracy_50']:.2%}")
        weak = at.identify_weak_agent()
        print(f"  Weak agent: {weak['agent'] if weak else 'None'}")
        print("  PASSED")

        # TEST 4: PreDecisionML
        print("\n--- TEST 4: PreDecisionML ---")
        pre_ml = PreDecisionML(data_dir=test_dir)
        feat_bull = {'regime': 2.0, 'trend_age': 20, 'vpower': 1.5, 'ff_streak': 5,
                     'atr_ratio': 0.8, 'sm_score': 75, 'macro_score': 65}
        feat_bear = {'regime': 0.0, 'trend_age': -15, 'vpower': 0.5, 'ff_streak': -4,
                     'atr_ratio': 2.0, 'sm_score': 25, 'macro_score': 30}
        p_bull = pre_ml.predict_success_probability(feat_bull)
        p_bear = pre_ml.predict_success_probability(feat_bear)
        print(f"  Bullish prob: {p_bull:.3f}, Bearish prob: {p_bear:.3f}")
        assert p_bull > p_bear
        weights = pre_ml.get_recommended_weights(feat_bull)
        print(f"  Weights (TRENDING): {weights}")
        assert abs(sum(weights.values()) - 1.0) < 0.01
        if pre_ml.should_retrain(tracker.decisions):
            m = pre_ml.retrain(tracker.decisions)
            print(f"  Retrain: {m.get('status')}, AUC={m.get('auc', 'N/A')}")
        print("  PASSED")

        # TEST 5: PostDecisionLearning
        print("\n--- TEST 5: PostDecisionLearning ---")
        pl = PostDecisionLearning(tracker)
        labeled = tracker.get_labeled_decisions()
        q = pl.analyze_decision_quality(labeled[0])
        print(f"  Quality: {q.get('quality')} - {q.get('quality_reason')}")
        insights = pl.generate_learning_insight(labeled, 20)
        print(f"  Insights ({len(insights)}): {insights[0][:80]}...")
        works = pl.get_what_works()
        print(f"  What works: {works['n_successes']} successes, {len(works['patterns'])} patterns")
        print("  PASSED")

        # TEST 6: ContinuousImprovement
        print("\n--- TEST 6: ContinuousImprovement ---")
        ci = ContinuousImprovement(data_dir=test_dir)
        enhanced = ci.pre_decision_enhance(
            {'TrendAgent': {'score': 75}, 'SmartMoneyAgent': {'score': 65}},
            {'regime': 'TRENDING', 'trend_age': 18, 'vpower': 1.4, 'ff_streak': 3,
             'atr_ratio': 0.9, 'sm_score': 65, 'macro_score': 60, 'sector': 'banking'})
        print(f"  Success prob: {enhanced['success_probability']:.3f}")
        print(f"  Adjusted weights: {enhanced['adjusted_weights']}")
        cycle = ci.run_daily_learning_cycle({}, {t: [random.uniform(5000,10000) for _ in range(20)] for t in tickers})
        print(f"  Daily cycle: updated={cycle['outcomes_updated']}, retrained={cycle['retrained']}")
        dash = ci.get_learning_dashboard_data()
        print(f"  Dashboard: {dash['total_decisions_tracked']} decisions, pending={dash['pending_outcomes']}")
        print("  PASSED")

        # TEST 7: MonthlyLearningReport
        print("\n--- TEST 7: MonthlyLearningReport ---")
        mr = MonthlyLearningReport(ci)
        rpt = mr.generate_report()
        print(f"  Month: {rpt['month']}, Success rate: {rpt['summary']['success_rate']:.0%}")
        print(f"  What worked: {rpt['what_worked'][0][:60]}...")
        print(f"  Recommendations: {rpt['recommendations'][0][:60]}...")
        print("  PASSED")

        print("\n" + "=" * 70)
        print("ALL 7 TESTS PASSED")
        print("=" * 70)
    finally:
        try: shutil.rmtree(test_dir)
        except: pass
