"""
Pixellent AI Engine — Backtesting Framework v1.0
Phase 2: AI Scoring Validation

Backtesting framework for the ML scoring system.
Validates AI score performance against the old formula (RSI*0.6 + AC*0.4).

Features:
    1. Historical label generation (forward-looking 15 bars)
    2. Walk-forward validation (train N months, test next month, roll)
    3. Performance metrics (accuracy, precision, recall, win_rate, profit, drawdown)
    4. Old vs New score comparison
    5. Signal quality analysis (win rate per score bucket)
    6. Report generation

Usage:
    from pixellent_backtesting import (
        run_walk_forward,
        compare_scoring_methods,
        analyze_signal_quality,
        generate_backtest_report,
    )

    report = run_walk_forward(signal_df, foreign_buy, foreign_sell,
                              train_months=6, test_months=1)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Import scoring module
try:
    from pixellent_scoring import (
        PixellentScorer, build_features, generate_labels,
        heuristic_score, score_stock, ALL_FEATURES, FEATURE_GROUPS,
    )
    HAS_SCORING = True
except ImportError:
    HAS_SCORING = False
    logger.warning("pixellent_scoring not importable. Some functions may fail.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, brier_score_loss,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# 1. HISTORICAL LABEL GENERATION
# =============================================================================

def generate_historical_labels(
    signal_df: pd.DataFrame,
    target_pct: float = 2.0,
    max_bars: int = 15,
    use_open_price: bool = True,
) -> pd.DataFrame:
    """
    Generate historical labels by looking forward for each buy signal.

    For each bar with buy_signal=True, checks if price reaches
    target_pct% gain within max_bars bars.

    Args:
        signal_df: Output of compute_signals().
        target_pct: Min % gain for success label.
        max_bars: Forward look window.
        use_open_price: If True, use next bar's open as entry price (realistic).

    Returns:
        DataFrame with columns:
            - label: 1=success, 0=fail, NaN=not a buy signal
            - entry_price: Price at entry
            - max_gain_pct: Maximum gain % within window
            - max_loss_pct: Maximum loss % within window
            - exit_bar: Bar index where target was first hit (or max_bars)
            - actual_return_pct: Return at exit bar
    """
    c = signal_df['close']
    o = signal_df.get('open', c)
    h = signal_df.get('high', c)
    l = signal_df.get('low', c)
    buy = signal_df.get('buy_signal', pd.Series(False, index=c.index))

    result = pd.DataFrame(index=c.index)
    result['label'] = np.nan
    result['entry_price'] = np.nan
    result['max_gain_pct'] = np.nan
    result['max_loss_pct'] = np.nan
    result['exit_bar'] = np.nan
    result['actual_return_pct'] = np.nan

    buy_indices = c.index[buy.astype(bool)]

    for idx in buy_indices:
        pos = c.index.get_loc(idx)

        # Entry price: next bar open (realistic) or current close
        if use_open_price and pos + 1 < len(c):
            entry = o.iloc[pos + 1]
        else:
            entry = c.iloc[pos]

        # Forward window
        start_pos = pos + 1
        end_pos = min(pos + max_bars + 1, len(c))

        if start_pos >= len(c):
            continue  # No forward data

        future_high = h.iloc[start_pos:end_pos]
        future_low = l.iloc[start_pos:end_pos]
        future_close = c.iloc[start_pos:end_pos]

        if len(future_close) == 0:
            continue

        # Max gain (using high prices for optimistic)
        max_price = future_high.max()
        max_gain = (max_price - entry) / entry * 100

        # Max loss (using low prices)
        min_price = future_low.min()
        max_loss = (min_price - entry) / entry * 100

        # Find first bar where gain target hit
        gains = (future_high - entry) / entry * 100
        target_hit = gains >= target_pct
        if target_hit.any():
            exit_bar = target_hit.values.argmax() + 1  # 1-based
            label = 1.0
        else:
            exit_bar = len(future_close)
            label = 0.0

        # Actual return at exit bar
        exit_pos = min(start_pos + exit_bar - 1, len(c) - 1)
        actual_return = (c.iloc[exit_pos] - entry) / entry * 100

        result.loc[idx, 'label'] = label
        result.loc[idx, 'entry_price'] = entry
        result.loc[idx, 'max_gain_pct'] = max_gain
        result.loc[idx, 'max_loss_pct'] = max_loss
        result.loc[idx, 'exit_bar'] = exit_bar
        result.loc[idx, 'actual_return_pct'] = actual_return

    return result


# =============================================================================
# 2. WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(
    signal_df: pd.DataFrame,
    foreign_buy: Optional[pd.Series] = None,
    foreign_sell: Optional[pd.Series] = None,
    train_months: int = 6,
    test_months: int = 1,
    target_pct: float = 2.0,
    max_bars: int = 15,
    min_train_signals: int = 30,
    xgb_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Walk-forward validation: train on N months, test on next month, roll.

    Args:
        signal_df: Full signal DataFrame (compute_signals output).
        foreign_buy: Foreign buy volume (optional).
        foreign_sell: Foreign sell volume (optional).
        train_months: Training window in months.
        test_months: Test window in months.
        target_pct: Target gain % for labels.
        max_bars: Forward look window for labels.
        min_train_signals: Minimum buy signals needed in train set.
        xgb_params: Custom XGBoost parameters.

    Returns:
        Dict with: fold_results, aggregate_metrics, predictions_df.
    """
    if not HAS_SCORING:
        raise RuntimeError("pixellent_scoring module required for walk-forward.")
    if not HAS_XGBOOST or not HAS_SKLEARN:
        raise RuntimeError("xgboost and sklearn required for walk-forward.")

    # Build features and labels
    feat_df = build_features(signal_df, foreign_buy, foreign_sell)
    labels_df = generate_historical_labels(signal_df, target_pct, max_bars)
    labels = labels_df['label']

    # Get date range
    dates = signal_df.index
    start_date = dates.min()
    end_date = dates.max()

    # Generate fold boundaries
    folds = []
    current_train_start = start_date

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        folds.append({
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })

        # Roll forward
        current_train_start = current_train_start + pd.DateOffset(months=test_months)

    if not folds:
        return {
            'fold_results': [],
            'aggregate_metrics': {},
            'predictions_df': pd.DataFrame(),
            'error': 'Not enough data for walk-forward validation.',
        }

    # Run walk-forward
    fold_results = []
    all_predictions = []

    for fold_idx, fold in enumerate(folds):
        # Split data
        train_mask = (dates >= fold['train_start']) & (dates < fold['train_end'])
        test_mask = (dates >= fold['test_start']) & (dates < fold['test_end'])

        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        # Filter to labeled samples
        train_valid = train_labels.notna()
        test_valid = test_labels.notna()

        n_train_signals = train_valid.sum()
        n_test_signals = test_valid.sum()

        if n_train_signals < min_train_signals or n_test_signals < 3:
            fold_results.append({
                'fold': fold_idx,
                'skipped': True,
                'reason': f'Insufficient signals (train={n_train_signals}, test={n_test_signals})',
                **fold,
            })
            continue

        # Train
        X_train = feat_df.loc[train_mask & train_valid]
        y_train = train_labels[train_valid].astype(int)
        X_test = feat_df.loc[test_mask & test_valid]
        y_test = test_labels[test_valid].astype(int)

        # Ensure all features present
        for col in ALL_FEATURES:
            if col not in X_train.columns:
                X_train[col] = 0.0
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_train = X_train[ALL_FEATURES].fillna(0)
        X_test = X_test[ALL_FEATURES].fillna(0)

        # Default params
        params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
        }
        if xgb_params:
            params.update(xgb_params)

        try:
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Predict on test set
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            # Metrics
            fold_metric = {
                'fold': fold_idx,
                'skipped': False,
                'train_start': fold['train_start'],
                'train_end': fold['train_end'],
                'test_start': fold['test_start'],
                'test_end': fold['test_end'],
                'n_train': len(y_train),
                'n_test': len(y_test),
                'train_pos_rate': float(y_train.mean()),
                'test_pos_rate': float(y_test.mean()),
                'accuracy': accuracy_score(y_test, preds),
                'precision': precision_score(y_test, preds, zero_division=0),
                'recall': recall_score(y_test, preds, zero_division=0),
                'f1': f1_score(y_test, preds, zero_division=0),
            }

            if len(y_test.unique()) > 1:
                fold_metric['roc_auc'] = roc_auc_score(y_test, probs)
            else:
                fold_metric['roc_auc'] = 0.5

            fold_results.append(fold_metric)

            # Store predictions
            pred_df = pd.DataFrame({
                'date': X_test.index,
                'y_true': y_test.values,
                'y_prob': probs,
                'y_pred': preds,
                'fold': fold_idx,
            })
            all_predictions.append(pred_df)

        except Exception as e:
            fold_results.append({
                'fold': fold_idx,
                'skipped': True,
                'reason': str(e),
                **fold,
            })

    # Aggregate metrics
    valid_folds = [f for f in fold_results if not f.get('skipped', False)]
    agg_metrics = {}

    if valid_folds:
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            values = [f[metric_name] for f in valid_folds if metric_name in f]
            if values:
                agg_metrics[f'{metric_name}_mean'] = np.mean(values)
                agg_metrics[f'{metric_name}_std'] = np.std(values)
                agg_metrics[f'{metric_name}_min'] = np.min(values)
                agg_metrics[f'{metric_name}_max'] = np.max(values)

        agg_metrics['n_folds'] = len(valid_folds)
        agg_metrics['n_skipped'] = len(folds) - len(valid_folds)
        agg_metrics['total_test_signals'] = sum(f.get('n_test', 0) for f in valid_folds)

    # Combine predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    return {
        'fold_results': fold_results,
        'aggregate_metrics': agg_metrics,
        'predictions_df': predictions_df,
    }



# =============================================================================
# 3. PERFORMANCE METRICS
# =============================================================================

def compute_trade_metrics(
    labels_df: pd.DataFrame,
    scores: Optional[pd.Series] = None,
    score_threshold: float = 50.0,
) -> Dict[str, float]:
    """
    Compute trading performance metrics from labeled signals.

    Args:
        labels_df: Output of generate_historical_labels().
        scores: AI scores (0-100) for filtering signals.
        score_threshold: Minimum score to count as a "taken" trade.

    Returns:
        Dict of performance metrics.
    """
    # Filter to labeled signals
    valid = labels_df['label'].notna()
    df = labels_df[valid].copy()

    if df.empty:
        return {'error': 'No labeled signals', 'n_trades': 0}

    # Apply score filter if provided
    if scores is not None:
        aligned_scores = scores.reindex(df.index)
        score_mask = aligned_scores >= score_threshold
        df = df[score_mask]

    if df.empty:
        return {'error': 'No trades pass score threshold', 'n_trades': 0}

    n_trades = len(df)
    n_wins = int((df['label'] == 1).sum())
    n_losses = int((df['label'] == 0).sum())

    returns = df['actual_return_pct']
    gains = returns[returns > 0]
    losses = returns[returns <= 0]

    # Core metrics
    metrics = {
        'n_trades': n_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': n_wins / n_trades if n_trades > 0 else 0,
        'avg_return_pct': returns.mean() if len(returns) > 0 else 0,
        'median_return_pct': returns.median() if len(returns) > 0 else 0,
        'total_return_pct': returns.sum(),
        'avg_win_pct': gains.mean() if len(gains) > 0 else 0,
        'avg_loss_pct': losses.mean() if len(losses) > 0 else 0,
        'max_win_pct': returns.max() if len(returns) > 0 else 0,
        'max_loss_pct': returns.min() if len(returns) > 0 else 0,
    }

    # Profit factor: gross profit / gross loss
    gross_profit = gains.sum() if len(gains) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    metrics['profit_factor'] = gross_profit / max(gross_loss, 0.01)

    # Expectancy: avg win * win_rate - avg_loss * loss_rate
    loss_rate = n_losses / n_trades if n_trades > 0 else 0
    metrics['expectancy'] = (
        metrics['avg_win_pct'] * metrics['win_rate'] +
        metrics['avg_loss_pct'] * loss_rate
    )

    # Max drawdown (sequential returns)
    if len(returns) > 0:
        cum_returns = (1 + returns / 100).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max * 100
        metrics['max_drawdown_pct'] = drawdown.min()
    else:
        metrics['max_drawdown_pct'] = 0

    # Avg bars held
    if 'exit_bar' in df.columns:
        metrics['avg_bars_held'] = df['exit_bar'].mean()
    else:
        metrics['avg_bars_held'] = 0

    return metrics


# =============================================================================
# 4. COMPARISON: OLD FORMULA vs NEW ML SCORE
# =============================================================================

def compare_scoring_methods(
    signal_df: pd.DataFrame,
    foreign_buy: Optional[pd.Series] = None,
    foreign_sell: Optional[pd.Series] = None,
    target_pct: float = 2.0,
    max_bars: int = 15,
    score_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compare old formula (RSI*0.6 + AC*0.4) vs new ML/heuristic score.

    Args:
        signal_df: Output of compute_signals().
        foreign_buy: Foreign buy volume (optional).
        foreign_sell: Foreign sell volume (optional).
        target_pct: Target gain % for labels.
        max_bars: Forward look window.
        score_thresholds: List of thresholds to test.

    Returns:
        Dict with comparison results.
    """
    if score_thresholds is None:
        score_thresholds = [30, 40, 50, 60, 70, 80]

    # Generate labels
    labels_df = generate_historical_labels(signal_df, target_pct, max_bars)

    # Old score: RSI*0.6 + AC_rel*100*0.4
    rsi = signal_df.get('rsi', pd.Series(50, index=signal_df.index))
    ac_rel = signal_df.get('ac_rel', pd.Series(0, index=signal_df.index))
    old_score = rsi * 0.6 + ac_rel * 100 * 0.4

    # Normalize old score to 0-100 range
    old_min = old_score.min()
    old_max = old_score.max()
    if old_max > old_min:
        old_score_norm = ((old_score - old_min) / (old_max - old_min) * 100).clip(0, 100)
    else:
        old_score_norm = pd.Series(50, index=signal_df.index)

    # New score (heuristic or model-based)
    if HAS_SCORING:
        new_result = score_stock(signal_df, foreign_buy, foreign_sell)
        new_score = new_result['ai_score'] if not new_result.empty else pd.Series(50, index=signal_df.index)
    else:
        new_score = pd.Series(50, index=signal_df.index)

    # Compare at different thresholds
    old_metrics_by_threshold = {}
    new_metrics_by_threshold = {}

    for threshold in score_thresholds:
        old_metrics_by_threshold[threshold] = compute_trade_metrics(
            labels_df, old_score_norm, threshold
        )
        new_metrics_by_threshold[threshold] = compute_trade_metrics(
            labels_df, new_score, threshold
        )

    # Overall (no threshold)
    overall_old = compute_trade_metrics(labels_df, old_score_norm, 0)
    overall_new = compute_trade_metrics(labels_df, new_score, 0)

    return {
        'old_score_metrics': old_metrics_by_threshold,
        'new_score_metrics': new_metrics_by_threshold,
        'overall_old': overall_old,
        'overall_new': overall_new,
        'score_thresholds': score_thresholds,
        'n_total_signals': int(labels_df['label'].notna().sum()),
        'old_score_stats': {
            'mean': float(old_score_norm.mean()),
            'std': float(old_score_norm.std()),
            'min': float(old_score_norm.min()),
            'max': float(old_score_norm.max()),
        },
        'new_score_stats': {
            'mean': float(new_score.mean()),
            'std': float(new_score.std()),
            'min': float(new_score.min()),
            'max': float(new_score.max()),
        },
    }


# =============================================================================
# 5. SIGNAL QUALITY ANALYSIS
# =============================================================================

def analyze_signal_quality(
    signal_df: pd.DataFrame,
    scores: pd.Series,
    target_pct: float = 2.0,
    max_bars: int = 15,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """
    Analyze signal quality by grouping signals into score buckets.

    Args:
        signal_df: Output of compute_signals().
        scores: AI scores (0-100).
        target_pct: Target gain % for labels.
        max_bars: Forward look window.
        n_buckets: Number of score buckets.

    Returns:
        DataFrame with per-bucket statistics:
            bucket, score_range, n_signals, win_rate, avg_return,
            avg_max_gain, avg_max_loss, profit_factor
    """
    labels_df = generate_historical_labels(signal_df, target_pct, max_bars)
    valid = labels_df['label'].notna()

    if valid.sum() == 0:
        return pd.DataFrame()

    df = labels_df[valid].copy()
    df['score'] = scores.reindex(df.index).fillna(50)

    # Create buckets
    bucket_edges = np.linspace(0, 100, n_buckets + 1)
    bucket_labels = [f"{int(bucket_edges[i])}-{int(bucket_edges[i+1])}"
                     for i in range(n_buckets)]

    df['bucket'] = pd.cut(df['score'], bins=bucket_edges, labels=bucket_labels,
                          include_lowest=True)

    # Aggregate per bucket
    results = []
    for bucket_name in bucket_labels:
        bucket_data = df[df['bucket'] == bucket_name]
        if len(bucket_data) == 0:
            results.append({
                'bucket': bucket_name,
                'n_signals': 0,
                'win_rate': np.nan,
                'avg_return_pct': np.nan,
                'avg_max_gain_pct': np.nan,
                'avg_max_loss_pct': np.nan,
                'profit_factor': np.nan,
                'avg_bars_held': np.nan,
            })
            continue

        n_signals = len(bucket_data)
        n_wins = int((bucket_data['label'] == 1).sum())
        returns = bucket_data['actual_return_pct']
        gains = returns[returns > 0]
        losses = returns[returns <= 0]

        gross_profit = gains.sum() if len(gains) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0

        results.append({
            'bucket': bucket_name,
            'n_signals': n_signals,
            'win_rate': n_wins / n_signals if n_signals > 0 else 0,
            'avg_return_pct': returns.mean() if len(returns) > 0 else 0,
            'avg_max_gain_pct': bucket_data['max_gain_pct'].mean(),
            'avg_max_loss_pct': bucket_data['max_loss_pct'].mean(),
            'profit_factor': gross_profit / max(gross_loss, 0.01),
            'avg_bars_held': bucket_data['exit_bar'].mean() if 'exit_bar' in bucket_data.columns else 0,
        })

    return pd.DataFrame(results)


# =============================================================================
# 6. REPORT GENERATION
# =============================================================================

def generate_backtest_report(
    signal_df: pd.DataFrame,
    foreign_buy: Optional[pd.Series] = None,
    foreign_sell: Optional[pd.Series] = None,
    target_pct: float = 2.0,
    max_bars: int = 15,
    run_walk_forward_test: bool = True,
    train_months: int = 6,
    test_months: int = 1,
) -> Dict[str, Any]:
    """
    Generate comprehensive backtest report.

    Includes:
        - Label statistics
        - Signal quality analysis
        - Old vs New score comparison
        - Walk-forward results (optional)
        - Summary recommendations

    Args:
        signal_df: Output of compute_signals().
        foreign_buy: Foreign buy volume (optional).
        foreign_sell: Foreign sell volume (optional).
        target_pct: Target gain %.
        max_bars: Forward look window.
        run_walk_forward_test: Whether to run walk-forward (slower).
        train_months: Training window for walk-forward.
        test_months: Test window for walk-forward.

    Returns:
        Dict with full report.
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'target_pct': target_pct,
            'max_bars': max_bars,
            'train_months': train_months,
            'test_months': test_months,
        },
    }

    # ── Label Statistics ──
    labels_df = generate_historical_labels(signal_df, target_pct, max_bars)
    valid = labels_df['label'].notna()
    n_signals = int(valid.sum())
    n_wins = int((labels_df['label'] == 1).sum())
    n_losses = int((labels_df['label'] == 0).sum())

    report['label_statistics'] = {
        'total_bars': len(signal_df),
        'total_buy_signals': n_signals,
        'buy_signal_rate': n_signals / max(len(signal_df), 1),
        'win_count': n_wins,
        'loss_count': n_losses,
        'base_win_rate': n_wins / max(n_signals, 1),
        'avg_return_pct': float(labels_df.loc[valid, 'actual_return_pct'].mean()) if n_signals > 0 else 0,
        'avg_max_gain_pct': float(labels_df.loc[valid, 'max_gain_pct'].mean()) if n_signals > 0 else 0,
        'avg_max_loss_pct': float(labels_df.loc[valid, 'max_loss_pct'].mean()) if n_signals > 0 else 0,
    }

    # ── Score Comparison ──
    try:
        comparison = compare_scoring_methods(
            signal_df, foreign_buy, foreign_sell, target_pct, max_bars
        )
        report['score_comparison'] = comparison
    except Exception as e:
        report['score_comparison'] = {'error': str(e)}

    # ── Signal Quality ──
    if HAS_SCORING:
        try:
            result = score_stock(signal_df, foreign_buy, foreign_sell)
            if not result.empty:
                quality = analyze_signal_quality(
                    signal_df, result['ai_score'], target_pct, max_bars
                )
                report['signal_quality'] = quality.to_dict('records') if not quality.empty else []
            else:
                report['signal_quality'] = []
        except Exception as e:
            report['signal_quality'] = {'error': str(e)}
    else:
        report['signal_quality'] = {'error': 'scoring module not available'}

    # ── Walk-Forward (optional) ──
    if run_walk_forward_test and HAS_XGBOOST and HAS_SKLEARN and HAS_SCORING:
        try:
            wf_result = run_walk_forward(
                signal_df, foreign_buy, foreign_sell,
                train_months=train_months, test_months=test_months,
                target_pct=target_pct, max_bars=max_bars,
            )
            report['walk_forward'] = {
                'aggregate_metrics': wf_result['aggregate_metrics'],
                'n_folds': len(wf_result['fold_results']),
                'fold_summaries': [
                    {k: v for k, v in f.items()
                     if k not in ('train_start', 'train_end', 'test_start', 'test_end')
                     or isinstance(v, (int, float, str, bool))}
                    for f in wf_result['fold_results']
                ],
            }
        except Exception as e:
            report['walk_forward'] = {'error': str(e)}
    else:
        report['walk_forward'] = {'skipped': True, 'reason': 'Dependencies or flag'}

    # ── Summary ──
    report['summary'] = _generate_summary(report)

    return report


def _generate_summary(report: Dict) -> Dict[str, Any]:
    """Generate human-readable summary from report."""
    summary = {}

    label_stats = report.get('label_statistics', {})
    base_wr = label_stats.get('base_win_rate', 0)
    n_signals = label_stats.get('total_buy_signals', 0)

    summary['data_sufficiency'] = 'GOOD' if n_signals >= 50 else ('FAIR' if n_signals >= 20 else 'LOW')
    summary['base_win_rate'] = f"{base_wr*100:.1f}%"

    # Check if new score improves over old
    comparison = report.get('score_comparison', {})
    if isinstance(comparison, dict) and 'overall_old' in comparison and 'overall_new' in comparison:
        old_wr = comparison['overall_old'].get('win_rate', 0)
        new_wr = comparison['overall_new'].get('win_rate', 0)
        summary['old_formula_win_rate'] = f"{old_wr*100:.1f}%"
        summary['new_score_win_rate'] = f"{new_wr*100:.1f}%"
        summary['improvement'] = f"{(new_wr - old_wr)*100:+.1f}pp"

    # Walk-forward
    wf = report.get('walk_forward', {})
    if isinstance(wf, dict) and 'aggregate_metrics' in wf:
        agg = wf['aggregate_metrics']
        if agg:
            summary['wf_auc_mean'] = f"{agg.get('roc_auc_mean', 0):.3f}"
            summary['wf_precision_mean'] = f"{agg.get('precision_mean', 0):.3f}"

    # Recommendation
    if base_wr >= 0.5:
        summary['recommendation'] = "Base win rate is solid. ML score can further filter low-quality signals."
    elif base_wr >= 0.35:
        summary['recommendation'] = "Base win rate moderate. ML score filtering at threshold 60+ recommended."
    else:
        summary['recommendation'] = "Base win rate low. Consider adjusting signal parameters or using higher score threshold."

    return summary



# =============================================================================
# 7. UTILITY: MULTI-STOCK BACKTEST
# =============================================================================

def backtest_multi_stock(
    signal_dfs: Dict[str, pd.DataFrame],
    foreign_buys: Optional[Dict[str, pd.Series]] = None,
    foreign_sells: Optional[Dict[str, pd.Series]] = None,
    target_pct: float = 2.0,
    max_bars: int = 15,
) -> pd.DataFrame:
    """
    Run backtest across multiple stocks and aggregate.

    Args:
        signal_dfs: Dict of ticker -> signal_df (from compute_signals).
        foreign_buys: Dict of ticker -> foreign_buy series.
        foreign_sells: Dict of ticker -> foreign_sell series.
        target_pct: Target gain %.
        max_bars: Forward look window.

    Returns:
        DataFrame with per-stock summary metrics.
    """
    results = []

    for ticker, sig_df in signal_dfs.items():
        if sig_df.empty:
            continue

        fb = foreign_buys.get(ticker) if foreign_buys else None
        fs = foreign_sells.get(ticker) if foreign_sells else None

        labels_df = generate_historical_labels(sig_df, target_pct, max_bars)
        metrics = compute_trade_metrics(labels_df)

        if metrics.get('n_trades', 0) > 0:
            metrics['ticker'] = ticker
            results.append(metrics)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).set_index('ticker')


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Pixellent Backtesting Framework — Test Mode")
    print("=" * 70)

    np.random.seed(42)
    n = 400
    idx = pd.date_range('2022-01-01', periods=n, freq='B')

    # Simulate OHLCV with slight upward bias
    returns = np.random.randn(n) * 0.015 + 0.0003
    close = pd.Series(5000 * np.cumprod(1 + returns), index=idx)
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(5_000_000, 80_000_000, n), index=idx, dtype=float)

    # Generate synthetic buy signals (roughly 5% of bars)
    buy_probs = np.random.rand(n)
    buy_signal = pd.Series(buy_probs < 0.05, index=idx)

    # Build signal_df mimicking compute_signals() output
    rsi_vals = 50 + np.cumsum(np.random.randn(n) * 2)
    rsi_vals = np.clip(rsi_vals, 10, 90)

    signal_df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'atr14': (high - low).rolling(14).mean().fillna(high - low),
        'ma8': close.rolling(8).mean(),
        'ma21': close.rolling(21).mean(),
        'ma55': close.rolling(55).mean(),
        'hma5': close.rolling(5).mean(),
        'ao': pd.Series(np.random.randn(n) * 20, index=idx),
        'ac': pd.Series(np.random.randn(n) * 10, index=idx),
        'ac_rel': pd.Series(np.random.randn(n) * 0.001, index=idx),
        'ac_naik': pd.Series(np.random.choice([True, False], n), index=idx),
        'vpower': pd.Series(np.random.uniform(0.5, 2.0, n), index=idx),
        'ha_bull': pd.Series(np.random.choice([True, False], n), index=idx),
        'ema_full': pd.Series(np.random.choice([True, False], n, p=[0.3, 0.7]), index=idx),
        'ema_half': pd.Series(np.random.choice([True, False], n, p=[0.5, 0.5]), index=idx),
        'trend_age': pd.Series(np.random.randint(0, 100, n), index=idx),
        'rsi': pd.Series(rsi_vals, index=idx),
        'rr_ratio': pd.Series(np.random.uniform(0.5, 3.0, n), index=idx),
        'regime': pd.Series(
            np.random.choice(['TRENDING', 'SIDEWAYS', 'HIGH_VOL', 'UNKNOWN'], n, p=[0.4, 0.3, 0.1, 0.2]),
            index=idx
        ),
        'buy_signal': buy_signal,
    })

    # Simulate foreign flow
    foreign_buy = pd.Series(np.random.randint(100_000, 10_000_000, n), index=idx, dtype=float)
    foreign_sell = pd.Series(np.random.randint(100_000, 8_000_000, n), index=idx, dtype=float)

    # ── Test 1: Historical Labels ──
    print("\n1. Historical Label Generation:")
    labels_df = generate_historical_labels(signal_df, target_pct=2.0, max_bars=15)
    valid = labels_df['label'].notna()
    n_labeled = int(valid.sum())
    n_wins = int((labels_df['label'] == 1).sum())
    print(f"   Total buy signals: {n_labeled}")
    print(f"   Wins (>=2% in 15 bars): {n_wins} ({n_wins/max(n_labeled,1)*100:.1f}%)")
    print(f"   Avg max gain: {labels_df.loc[valid, 'max_gain_pct'].mean():.2f}%")
    print(f"   Avg max loss: {labels_df.loc[valid, 'max_loss_pct'].mean():.2f}%")
    print(f"   Avg actual return: {labels_df.loc[valid, 'actual_return_pct'].mean():.2f}%")

    # ── Test 2: Trade Metrics ──
    print("\n2. Trade Metrics (all signals):")
    metrics = compute_trade_metrics(labels_df)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # ── Test 3: Signal Quality Analysis ──
    print("\n3. Signal Quality by Score Bucket:")
    if HAS_SCORING:
        try:
            result = score_stock(signal_df, foreign_buy, foreign_sell)
            if not result.empty:
                quality = analyze_signal_quality(
                    signal_df, result['ai_score'], target_pct=2.0, max_bars=15, n_buckets=5
                )
                if not quality.empty:
                    print(quality.to_string(index=False))
                else:
                    print("   No quality data (insufficient signals per bucket)")
            else:
                print("   Scoring returned empty")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("   SKIPPED (pixellent_scoring not available)")

    # ── Test 4: Score Comparison ──
    print("\n4. Old vs New Score Comparison:")
    if HAS_SCORING:
        try:
            comp = compare_scoring_methods(signal_df, foreign_buy, foreign_sell)
            print(f"   Total signals: {comp['n_total_signals']}")
            print(f"   Old score stats: mean={comp['old_score_stats']['mean']:.1f}, "
                  f"std={comp['old_score_stats']['std']:.1f}")
            print(f"   New score stats: mean={comp['new_score_stats']['mean']:.1f}, "
                  f"std={comp['new_score_stats']['std']:.1f}")
            print(f"\n   Win rate at threshold 50:")
            old_50 = comp['old_score_metrics'].get(50, {})
            new_50 = comp['new_score_metrics'].get(50, {})
            print(f"     Old: {old_50.get('win_rate', 0)*100:.1f}% ({old_50.get('n_trades', 0)} trades)")
            print(f"     New: {new_50.get('win_rate', 0)*100:.1f}% ({new_50.get('n_trades', 0)} trades)")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("   SKIPPED (pixellent_scoring not available)")

    # ── Test 5: Walk-Forward (only if deps available) ──
    print("\n5. Walk-Forward Validation:")
    if HAS_XGBOOST and HAS_SKLEARN and HAS_SCORING:
        try:
            wf = run_walk_forward(
                signal_df, foreign_buy, foreign_sell,
                train_months=4, test_months=1,
                target_pct=2.0, max_bars=15,
                min_train_signals=10,
            )
            agg = wf['aggregate_metrics']
            if agg:
                print(f"   Folds completed: {agg.get('n_folds', 0)}")
                print(f"   Folds skipped: {agg.get('n_skipped', 0)}")
                print(f"   Total test signals: {agg.get('total_test_signals', 0)}")
                print(f"   AUC mean: {agg.get('roc_auc_mean', 0):.3f} "
                      f"(+/- {agg.get('roc_auc_std', 0):.3f})")
                print(f"   Precision mean: {agg.get('precision_mean', 0):.3f}")
                print(f"   Recall mean: {agg.get('recall_mean', 0):.3f}")
            else:
                print("   No valid folds completed")
                if wf.get('error'):
                    print(f"   Error: {wf['error']}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        missing = []
        if not HAS_XGBOOST:
            missing.append('xgboost')
        if not HAS_SKLEARN:
            missing.append('sklearn')
        if not HAS_SCORING:
            missing.append('pixellent_scoring')
        print(f"   SKIPPED (missing: {', '.join(missing)})")

    # ── Test 6: Report Generation ──
    print("\n6. Report Generation:")
    if HAS_SCORING:
        try:
            report = generate_backtest_report(
                signal_df, foreign_buy, foreign_sell,
                target_pct=2.0, max_bars=15,
                run_walk_forward_test=False,  # Skip WF for speed in test
            )
            summary = report.get('summary', {})
            print(f"   Data sufficiency: {summary.get('data_sufficiency', 'N/A')}")
            print(f"   Base win rate: {summary.get('base_win_rate', 'N/A')}")
            print(f"   Recommendation: {summary.get('recommendation', 'N/A')}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("   SKIPPED (pixellent_scoring not available)")

    print("\n" + "=" * 70)
    print("Pixellent Backtesting Framework ready!")
    print(f"  - HAS_SCORING: {HAS_SCORING}")
    print(f"  - HAS_XGBOOST: {HAS_XGBOOST}")
    print(f"  - HAS_SKLEARN: {HAS_SKLEARN}")
    print("=" * 70)
