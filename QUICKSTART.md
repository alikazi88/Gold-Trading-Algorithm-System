# Quick Start Guide

Get your Gold Scalping System up and running in 5 steps.

## Step 1: Install Dependencies

```bash
cd /Users/ali/Developer/trader/gold_scalping_system
pip install -r requirements.txt
```

## Step 2: Configure API Keys

### AllTick API
1. Sign up at https://alltick.co
2. Get your API token
3. Edit `config/config.yaml`:
```yaml
api:
  alltick:
    token: "YOUR_ALLTICK_API_TOKEN_HERE"
```

### Telegram Bot (Optional)
1. Create a bot via @BotFather on Telegram
2. Get your chat ID from @userinfobot
3. Edit `config/telegram_config.yaml`:
```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
```

## Step 3: Fetch Historical Data

```bash
python main.py --fetch-data --days 365
```

This downloads 1 year of 5-minute GOLD candles. Wait for completion (may take 10-20 minutes due to rate limits).

Expected output:
```
Fetching historical data for XAUUSD from 2023-01-01 to 2024-01-01
Fetched 1500 candles for period...
Total candles fetched: 105120
Inserted 105120 candles into database
Data fetch completed successfully
```

## Step 4: Train the Model

```bash
python train_model.py
```

This will:
- Calculate 56 features for all candles
- Label data based on future price action
- Train Random Forest with hyperparameter tuning
- Evaluate performance
- Save model to `models/saved/`

Expected duration: 15-30 minutes

Expected output:
```
[7/7] Evaluating model performance...
Accuracy: 0.6234
Precision: 0.6789
Recall: 0.6123
F1 Score: 0.6432

MODEL EVALUATION SUMMARY
Trading Metrics:
  Buy Signals: 4523 (Win Rate: 68.20%)
  Sell Signals: 4312 (Win Rate: 66.80%)
  Overall Trade Win Rate: 67.50%

Model saved to models/saved/gold_scalping_model_20240116_143022.pkl
```

## Step 5: Run Live Trading

```bash
python main.py
```

The system will:
- Load the trained model
- Connect to AllTick API
- Process each 5-minute candle
- Generate and execute signals
- Send Telegram notifications

Press `Ctrl+C` to stop gracefully.

## Verification

### Check Database
```python
import sqlite3
conn = sqlite3.connect('data/gold_trading.db')
cursor = conn.cursor()

# Check candles
cursor.execute("SELECT COUNT(*) FROM candles")
print(f"Candles: {cursor.fetchone()[0]}")

# Check signals
cursor.execute("SELECT COUNT(*) FROM signals")
print(f"Signals: {cursor.fetchone()[0]}")

# Check trades
cursor.execute("SELECT COUNT(*) FROM trades")
print(f"Trades: {cursor.fetchone()[0]}")
```

### Check Logs
```bash
tail -f logs/trading_system.log
```

### Test Telegram
```python
from notifications.telegram_bot import TelegramNotifier
from utils.helpers import load_config

config = load_config('config/config.yaml')
telegram_config = load_config('config/telegram_config.yaml')
config.update(telegram_config)

notifier = TelegramNotifier(config)
notifier.send_custom_message("🎉 System test successful!")
```

## Common Issues

### Issue: "No module named 'sklearn'"
**Solution**: Install scikit-learn
```bash
pip install scikit-learn==1.3.0
```

### Issue: "API connection validation failed"
**Solution**: Check your AllTick API token in `config/config.yaml`

### Issue: "No trained model found"
**Solution**: Run `python train_model.py` first

### Issue: "Insufficient data for processing"
**Solution**: Fetch more historical data with `--days 365` or higher

## Next Steps

1. **Paper Trade**: Run the system for 1-2 weeks without real money
2. **Monitor Performance**: Check daily summaries and trade outcomes
3. **Optimize**: Adjust confidence threshold, position size, etc.
4. **Retrain**: Run `python train_model.py --retrain` every 2 weeks
5. **Backtest**: Validate strategy on different time periods

## Performance Monitoring

### Daily Summary
Check Telegram for daily stats at 22:00 UTC:
```
📊 Daily Stats: 7 trades, 5W-2L, Win Rate: 71%, Net: +85 pips
```

### Database Queries
```sql
-- Today's trades
SELECT * FROM trades 
WHERE entry_time >= strftime('%s', 'now', 'start of day');

-- Win rate last 30 days
SELECT 
  COUNT(*) as total,
  SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
  ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
FROM trades
WHERE entry_time >= strftime('%s', 'now', '-30 days');
```

## Safety Checklist

- [ ] Tested in paper trading mode
- [ ] Configured proper position size
- [ ] Set daily drawdown limit
- [ ] Telegram notifications working
- [ ] Logs being written
- [ ] Database backup enabled
- [ ] Understand the strategy logic
- [ ] Know how to stop the system

## Support

- Read full documentation in `README.md`
- Check logs in `logs/trading_system.log`
- Review code comments for details
- Test individual components before live trading

**Remember**: Start small, test thoroughly, never risk more than you can afford to lose.
