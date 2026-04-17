import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER", "")

# Pushover
PUSHOVER_APP_TOKEN = os.getenv("PUSHOVER_APP_TOKEN", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")

# App
TICKER = os.getenv("TICKER", "")
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "60"))

# Circuit breaker
VIX_THRESHOLD      = float(os.getenv("VIX_THRESHOLD",      "25"))
SPY_DROP_THRESHOLD = float(os.getenv("SPY_DROP_THRESHOLD", "-1.5"))

# Risk management
PORTFOLIO_SIZE      = float(os.getenv("PORTFOLIO_SIZE",     "25000"))  # total account $
MAX_RISK_PER_TRADE  = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))   # 2% of portfolio per trade
MAX_DAILY_LOSS      = float(os.getenv("MAX_DAILY_LOSS",     "0.04"))   # 4% daily loss limit
MAX_OPEN_POSITIONS  = int(os.getenv("MAX_OPEN_POSITIONS",   "5"))      # max concurrent trades
MAX_SECTOR_PCT      = float(os.getenv("MAX_SECTOR_PCT",     "0.40"))   # max 40% in one sector
MAX_SMALLCAP_PCT    = float(os.getenv("MAX_SMALLCAP_PCT",   "0.30"))   # max 30% in small-caps
ENTRY_ZONE_SLACK    = float(os.getenv("ENTRY_ZONE_SLACK",   "0.03"))   # 3% above zone = extended
SIGNAL_MAX_AGE_MIN  = int(os.getenv("SIGNAL_MAX_AGE_MIN",  "20"))      # signal stale after 20 min

# LangSmith — tracing is activated automatically when these are set in the environment.
# No explicit SDK calls needed; LangGraph picks them up on import.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "argus")
