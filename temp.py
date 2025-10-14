from pathlib import Path
text = Path('train_intraday_trend_model.py').read_text()
Path('train_intraday_trend_model.py').write_text(text)
