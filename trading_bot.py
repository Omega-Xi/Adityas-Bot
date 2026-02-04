import threading
import upstox_client
from upstox_client import MarketDataStreamerV3
from upstox_client.rest import ApiException
import keyboard
import sys
import logging
import ctypes
import platform
import winsound
import pandas as pd
import os
import math
import numpy as np
import pytz
from dataclasses import dataclass,field
from typing import Optional
from datetime import datetime,date,timedelta
from dotenv import load_dotenv,set_key
import webbrowser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if platform.system() != "Windows":
    logging.warning("Wake Lock is only supported on Windows. Ensure Network connection is enabled during sleep.")

@dataclass
class Trade:
    trade_id: int = field(init=False)
    instrument: str
    type: str
    entry_time: datetime
    entry_price: float
    quantity: int
    trigger_price: float
    target_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "ACTIVE"
    exit_reason: Optional[str] = None
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    charges: float = 0.0
    trailing_trigger: Optional[float] = 0.0
    highest_price: Optional[float] = 0.0

    _id_counter: int = 0

    def __post_init__(self):
        type(self)._id_counter += 1
        self.trade_id = type(self)._id_counter
        self.highest_price = self.entry_price

class Config:
    API_VERSION = "2.0"
    CONFIGURATION=upstox_client.Configuration()
    DRY_RUN_MARGIN=100000
    DRY_RUN=True
    TRADE_LIMIT=5
    MARKET_OPEN_TIME = "09:15"
    MARKET_CLOSE_TIME = "15:00"
    TIME_UNIT="minutes"
    INTERVALS=["1","3"]
    ENTRY_COOLDOWN = 10 # cooldown period in seconds between trades
    LOTS=1
    STRIKE_DIFF=50
    STRIKE_OFFSET=2
    RSI_TRESHOLD_LOW=0
    RSI_MID_TRESHOLD=50
    RSI_TRESHOLD_HIGH=100
    VWAP_CLOSE_TOLERANCE=0.001
    ADX_TRESHOLD=25
    SL_ATR_TIMEFRAME="3"
    SL_POINTS=13
    TARGET_POINTS=100
    TRAILING_TRIGGER=21
    MINIMUM_SL_PRICE=20
    TRAILING_MULTIPLIER=1.5
    TRAILING_PERIOD=14
    INTERVAL_MAP = {
        "1": "1min",
        "3": "3min",
    }

class Terminator:
    def __init__(self,bot):
        self.bot=bot
    
    def listen_for_kill(self):
        keyboard.on_press_key("esc",self.emergency_kill)
        keyboard.on_press_key("q",self.kill_bot)

    def kill_bot(self,event=None):
        self.bot.kill_switch = True
        if not Config.DRY_RUN:
            if self.bot.check_position():
                self.bot.exit_trade()
                logging.info("Open Position Found. Exiting Trade Before Shutdown")
        if hasattr(self.bot, "streamer"):
            self.bot.streamer.disconnect()
        logging.info("Bot Stopped Gracefully")

    def emergency_kill(self,event=None):
        self.bot.kill_switch=True
        if hasattr(self.bot, "streamer"):
            self.bot.streamer.disconnect()
        logging.critical("Emergency stop. Bot Terminated>>>")
        sys.exit(1)

class Wake_Lock:
    def __init__(self):
        self.ES_CONTINUOUS       = 0x80000000
        self.ES_SYSTEM_REQUIRED  = 0x00000001
        self.ES_DISPLAY_REQUIRED = 0x00000002
        self.active=False

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()
    
    def activate(self):
        if platform.system() != "Windows":
            return

        if not ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_DISPLAY_REQUIRED):
            logging.error("Failed To Activate Wake Lock")
        else:
            self.active=True
            logging.info("Wake Lock Activated")

    def deactivate(self):
        if platform.system() != "Windows":
            return
        
        if not ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS):
            logging.error("Failed To Deactivate Wake Lock")
        else:
            self.active=False
            logging.info("Wake Lock Deactivated")

class Alerts:
    @staticmethod
    def websocket_connected():
        winsound.Beep(2000,200)

    @staticmethod
    def websocket_error():
        winsound.Beep(400,200)
        winsound.Beep(400,200)
        winsound.Beep(400,200)
        winsound.Beep(400,3000)

    @staticmethod
    def websocket_disconnected():
        winsound.Beep(1200,1000)

    @staticmethod
    def trade_entered():
        winsound.Beep(900,200)

    @staticmethod
    def trade_exited():
        winsound.Beep(700,200)

    @staticmethod
    def error():
        winsound.Beep(1000,1000)

class Dry_Run_Services:
    @staticmethod
    def calculate_charges(entry_price, exit_price, quantity, product_type="intraday", brokerage_fee=30):
        turnover = (entry_price + exit_price) * quantity
        buy_value = entry_price * quantity
        sell_value = exit_price * quantity

        brokerage = brokerage_fee * 2
        stt = {
            "delivery": 0.001 * (buy_value + sell_value),
            "intraday": 0.00025 * sell_value,
            "futures": 0.000125 * sell_value,
            "options": 0.000625 * sell_value
        }.get(product_type, 0)

        txn_charges = 0.0000325 * turnover
        sebi_fees = 0.000001 * turnover
        stamp_duty = {
            "delivery": 0.00015 * buy_value,
            "intraday": 0.00003 * buy_value,
            "futures": 0.00002 * buy_value,
            "options": 0.00003 * buy_value
        }.get(product_type, 0)

        gst = 0.18 * (brokerage + txn_charges)
        total_charges = brokerage + stt + txn_charges + sebi_fees + stamp_duty + gst

        return total_charges
    
    @staticmethod
    def export_trades_to_csv(trades, filename="trade_log.csv"):
        if not trades:
            logging.info("No trades to export.")
            return

        trade_dicts = [trade.__dict__ for trade in trades]
        df = pd.DataFrame(trade_dicts)

        try:
            df.to_csv(filename, index=False)
            logging.info(f"Trade log exported to {filename}")
        except Exception as e:
            logging.error(f"Failed to export trades: {e}")

    @staticmethod
    def generate_performance_report(transcriber):
        trades = transcriber.trades
        closed_trades = [t for t in trades if t.status == "CLOSED"]
        active_trades = [t for t in trades if t.status == "ACTIVE"]

        print("\n" + "="*60)
        print("FINAL TRADING PERFORMANCE REPORT")
        print("="*60)

        if active_trades:
            logging.info(f"\nACTIVE TRADES: {len(active_trades)}")
            for t in active_trades:
                print(f"Trade {t.trade_id}: {t.type} | Entry: {t.entry_price} | Qty: {t.quantity} | Target: {t.target_price}")

        if closed_trades:
            total_pnl = sum(t.pnl for t in closed_trades)
            win_rate = (sum(1 for t in closed_trades if t.pnl > 0) / len(closed_trades)) * 100
            avg_win = (sum(t.pnl for t in closed_trades if t.pnl > 0) / sum(1 for t in closed_trades if t.pnl > 0)) if any(t.pnl > 0 for t in closed_trades) else 0
            avg_loss = (sum(t.pnl for t in closed_trades if t.pnl <= 0) / sum(1 for t in closed_trades if t.pnl <= 0)) if any(t.pnl <= 0 for t in closed_trades) else 0
            profit_factor = (sum(t.pnl for t in closed_trades if t.pnl > 0) / abs(sum(t.pnl for t in closed_trades if t.pnl <= 0))) if any(t.pnl <= 0 for t in closed_trades) else float('inf')
            logging.info(f"\nCLOSED TRADES PERFORMANCE:")
            print(f"Initial Balance: ₹{transcriber.initial_balance:.2f}")
            print(f"Final Balance:   ₹{transcriber.current_balance:.2f}")
            print(f"Total Trades:    {len(closed_trades)}")
            print(f"Winning Trades:  {sum(1 for t in closed_trades if t.pnl > 0)}")
            print(f"Average Win:     ₹{avg_win:.2f}")
            print(f"Losing Trades:   {sum(1 for t in closed_trades if t.pnl <= 0)}")
            print(f"Average Loss:    ₹{avg_loss:.2f}")
            print(f"Profit Factor:   {profit_factor:.2f}")
            print(f"Win Rate:        {win_rate:.2f}%")
            print(f"Total P&L:       ₹{total_pnl:.2f}")
        else:
            logging.info("\nNo closed trades to analyze")

        print("="*60)

class Transcriber:
    def __init__(self, initial_margin):
        self.trades = []
        self.position = None
        self.initial_balance = initial_margin
        self.current_balance = initial_margin

    def record_entry(self, trade):
        self.trades.append(trade)
        self.position = trade.type
        logging.info(f"Trade entered: {trade.type} at {trade.entry_price}, Time: {trade.entry_time.strftime("%H:%M:%S")}")

    def record_exit(self, exit_price, exit_reason, timestamp):
        active_trade = next((t for t in reversed(self.trades) if t.status == "ACTIVE"), None)
        if not active_trade:
            logging.info("No active trade to exit")
            return

        active_trade.exit_time = timestamp
        active_trade.exit_price = exit_price
        active_trade.status = "CLOSED"
        active_trade.exit_reason = exit_reason

        charges = Dry_Run_Services.calculate_charges(active_trade.entry_price, exit_price, active_trade.quantity, product_type="intraday")
        gross_pnl = (exit_price - active_trade.entry_price) * active_trade.quantity
        net_pnl = gross_pnl - charges

        active_trade.gross_pnl = gross_pnl
        active_trade.net_pnl = net_pnl
        active_trade.charges = charges
        active_trade.pnl = net_pnl

        self.current_balance += net_pnl
        logging.info(f"Trade exited: {active_trade.type} at {exit_price}, Time: {timestamp.strftime("%H:%M:%S")}, Net P&L: {net_pnl:.2f}")

        self.position = None

class Authenticator:
    GRANT_TYPE = 'authorization_code'
    ENV_PATH = ".env"
    
    def __init__(self):
        self.check_env_file()
        load_dotenv()
        self.api_key=os.getenv('api_key')
        self.api_secret=os.getenv('api_secret')
        self.redirect_url=os.getenv('redirect_url')
        self.state=os.getenv('state')
        self.url=f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={self.api_key}&redirect_uri={self.redirect_url}&state={self.state}"
        self.access_token=os.getenv('access_token')
        self.configuration = upstox_client.Configuration()
        if self.access_token:
            self.configuration.access_token=self.access_token

    def check_env_file(self):
        if not os.path.exists(self.ENV_PATH):
            logging.warning("API details not found.")
            for key in ["api_key", "api_secret", "redirect_url", "state"]:
                set_key(self.ENV_PATH, key, input(f"Enter {key.replace('_',' ').title()}: "))
            set_key(self.ENV_PATH, "access_token", "")
            logging.info("API details saved")

    def get_access_token(self):
        if self.check_token_validity():
            logging.info("Access Token Validity Confirmed")
            return self.access_token
        self.generate_access_token()
        return self.access_token

    def generate_access_token(self):
        webbrowser.open(self.url)
        new_uri=input("Enter Redirect URL:")
        code=self.get_code(new_uri)
        if not code:
            logging.error("Invalid Redirect URL")
            return
        self.fetch_token(code)
        if self.access_token:
            self.update_access_token()
            logging.info("Access Token Updated")
        else:
            logging.error("Invalid Code")

    def get_code(self,uri):
        try:
            return uri.split("code=")[1].split("&state")[0]
        except Exception:
            return

    def fetch_token(self,code):
        api_instance = upstox_client.LoginApi()
        try:
            # Get token API
            api_response = api_instance.token(Config.API_VERSION, code=code, client_id=self.api_key, client_secret=self.api_secret,redirect_uri=self.redirect_url, grant_type=self.GRANT_TYPE)
            self.access_token=api_response.access_token
        except ApiException as e:
            logging.error(f"Access Denied : {e}")
            self.access_token=None

    def check_token_validity(self):
        api_instance = upstox_client.PortfolioApi(upstox_client.ApiClient(self.configuration))
        try:
            return bool(api_instance.get_positions(Config.API_VERSION))
        except ApiException as e:
            logging.warning("Token Expired")
            return False

    def update_access_token(self):
        set_key(self.ENV_PATH, "access_token", self.access_token)
        load_dotenv(self.ENV_PATH, override=True)
        self.access_token = os.getenv("access_token")
        self.configuration.access_token = self.access_token

class Data_Processor:
    def __init__(self):
        self.get_instruments()
        self.expiry_date=None
        self.instrument_key=None
        self.option_key=None
        self.ce_strike_price=None
        self.pe_strike_price=None

    def get_instruments(self):
        self.instruments=pd.read_json("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz")
        self.instruments['expiry']=pd.to_datetime(self.instruments['expiry'],unit='ms',errors='coerce')
        self.instruments['expiry']=self.instruments['expiry'].dt.date
        self.instruments['expiry']=pd.to_datetime(self.instruments['expiry'])
        logging.info("Loaded Instrument Data")
        
    @staticmethod
    def convert_to_df(response):
        df = pd.DataFrame(response.data.candles, columns=["timestamp","open","high","low","close","volume","open_interest"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
        df.sort_index(inplace=True)
        return df
    
    @staticmethod
    def convert_to_candles(df,interval,historic_df,intraday_df):
        time_frame=Config.INTERVAL_MAP[interval]
        resampled = df.resample(time_frame).agg({
                "price": ["first", "max", "min", "last"],
                "ltq": "sum"
            }).dropna()
        resampled.columns = ["open", "high", "low", "close", "volume"]
        frames = [historic_df, intraday_df, resampled]
        frames = [f for f in frames if f is not None and not f.empty]
        candle_df = pd.concat(frames) if frames else pd.DataFrame()
        candle_df = candle_df[~candle_df.index.duplicated(keep='last')]
        return candle_df

    def get_instrument_key(self,name):
        self.name=name
        filtered=self.instruments[self.instruments['trading_symbol']==self.name]['instrument_key']
        if filtered.empty:
            logging.warning("Instrument Key Not Found")
            return None
        self.instrument_key=filtered.squeeze()
        logging.info(f"Instrument Key: {self.instrument_key}")
        return self.instrument_key
    
    def get_lot_size(self):
        futures = self.instruments[
            (self.instruments['segment'] == "NSE_FO") &
            (self.instruments['underlying_symbol'] == self.name)
        ]
        futures_sorted = futures.sort_values(by='expiry')
        if not futures_sorted.empty:
            lot_size=int(futures_sorted.iloc[0]['lot_size'])
            logging.info(f"Lot Size For {self.name} :{lot_size}")
            return lot_size
        else:
            logging.warning(f"No Futures Found For Instrument: {self.name}")
            return None
        
    def get_expiry_date(self):
        try:
            instruments=self.instruments[self.instruments['name']==self.name]
            instruments_sorted = instruments.sort_values(by='expiry')
            first_expiry = instruments_sorted['expiry'].dropna().sort_values().iloc[0]
            first_expiry = first_expiry.strftime('%Y-%m-%d')
            self.expiry_date=first_expiry
            logging.info(f"Expiry Date :{self.expiry_date}")
            return self.expiry_date
        except Exception as e:
            logging.error(f"Unable To Fetch Expiry : {e}")
            return None

    def get_option_key(self,order_type):
        if order_type is None:
            logging.warning("Order type not provided for option key retrieval")
            return None
        if order_type == "CE":
            strike_price = self.ce_strike_price
        elif order_type == "PE":
            strike_price = self.pe_strike_price
        self.expiry_date = pd.to_datetime(self.expiry_date)
        option_key = self.instruments[
            (self.instruments['instrument_type'] == order_type) &
            (self.instruments['name'] == self.name) &
            (self.instruments['expiry'] == self.expiry_date) &
            (self.instruments['strike_price'] == strike_price)
        ]['instrument_key']
        if option_key.empty:
            logging.critical(f"Option key not found for {order_type}, {self.name}, {self.expiry_date}, {self.strike_price}")
            return None
        self.option_key=option_key.squeeze()
        return self.option_key
    
    def get_strike_price(self,index_price):
        if index_price is None:
            logging.warning("Index price not available for strike price calculation")
            return None
        else:
            self.ce_strike_price = (math.floor(index_price / Config.STRIKE_DIFF) * Config.STRIKE_DIFF) - (Config.STRIKE_OFFSET * Config.STRIKE_DIFF)
            self.pe_strike_price = (math.floor(index_price / Config.STRIKE_DIFF) * Config.STRIKE_DIFF) + (Config.STRIKE_OFFSET * Config.STRIKE_DIFF)
        logging.info(f"Calculated CE Strike Price: {self.ce_strike_price}")
        logging.info(f"Calculated PE Strike Price: {self.pe_strike_price}")
        return self.ce_strike_price, self.pe_strike_price
    
class Data_Collector:
    def __init__(self,access_token,dry_run):
        self.access_token=access_token
        Config.CONFIGURATION = upstox_client.Configuration()
        Config.CONFIGURATION.access_token = self.access_token
        self.dry_run=dry_run
        self.option_price=None
        self.available_margin=None

    def get_margin(self):
        if self.dry_run:
            self.available_margin=Config.DRY_RUN_MARGIN
            logging.info("Dry Run Margin Fetched")
            return self.available_margin
        api_instance = upstox_client.UserApi(upstox_client.ApiClient(Config.CONFIGURATION))
        try:
            # Get User Fund And Margin
            api_response = api_instance.get_user_fund_margin(Config.API_VERSION)
            self.available_margin=api_response.data['equity'].available_margin
            logging.info("Margin Fetched Successfully")
            return self.available_margin
        except ApiException as e:
            logging.error(f"Exception while fetching margin :{e}")
            return None

    def get_historic_data(self,instrument_key):
        today=date.today()
        previous_day=today-timedelta(days=7)
        str_today=str(today)
        str_previous_day=str(previous_day)
        apiInstance = upstox_client.HistoryV3Api(upstox_client.ApiClient(Config.CONFIGURATION))
        try:
            dfs=[]
            for interval in Config.INTERVALS:
                df=Data_Processor.convert_to_df(
                    apiInstance.get_historical_candle_data1(instrument_key, Config.TIME_UNIT, interval, str_today, str_previous_day)
                )
                dfs.append(df)
            return tuple(dfs)
        except Exception as e:
            logging.error(f"Exception while fetching Historic Data :{e}")
            return None

    def get_intraday_data(self,instrument_key):
        api_instance = upstox_client.HistoryV3Api(upstox_client.ApiClient(Config.CONFIGURATION))
        try:
            dfs=[]
            for interval in Config.INTERVALS:
                df=Data_Processor.convert_to_df(
                    api_instance.get_intra_day_candle_data(instrument_key, Config.TIME_UNIT, interval)
                )
                dfs.append(df)
            return tuple(dfs)
        except Exception as e:
            logging.error(f"Exception while fetching Intraday Data :{e}")
            return None

    def get_option_price(self,option_key):
        if option_key is None:
            logging.error("Invalid option key")
            return None
        Config.CONFIGURATION.access_token = self.access_token
        api_instance = upstox_client.MarketQuoteV3Api(upstox_client.ApiClient(Config.CONFIGURATION))
        try:
            response = api_instance.get_ltp(instrument_key=option_key)
            last_trade_price = response.data[next(iter(response.data))].last_price
            if last_trade_price:
                self.option_price=last_trade_price
                logging.info(f"Option Price :{self.option_price}")
                return self.option_price
            else:
                logging.warning("No candle data available.")
                return None
        except ApiException as e:
            logging.error(f"API Exception: {e}")
            return None
        except Exception as e:
            logging.error(f"Exception when fetching option price: {e}")
            return None
        
    def check_position(self):
        api_instance = upstox_client.PortfolioApi(upstox_client.ApiClient(Config.CONFIGURATION))
        try:
            api_response = api_instance.get_positions(Config.API_VERSION)
            return bool(api_response.data)
        except ApiException as e:
            logging.error(f"Exception when fetching position data :{e}")
            return None

class Calculations:

    # --- VWAP Calculation ---
    @staticmethod
    def calculate_vwap(df):
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["date"] = df.index.date
        df["cum_vol_price"] = (df["typical_price"] * df["volume"]).groupby(df["date"]).cumsum()
        df["cum_volume"] = df["volume"].groupby(df["date"]).cumsum()
        df["vwap"] = df["cum_vol_price"] / df["cum_volume"]
        return df
    
    # --- RSI Calculation ---
    @staticmethod
    def calculate_rsi(df):
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs.fillna(0)))
        return df
    
    # --- ATR Calculation (RMA) ---
    @staticmethod
    def calculate_atr(df):
        df['high_low']   = df['high'] - df['low']
        df['high_close'] = (df['high'] - df['close'].shift()).abs()
        df['low_close']  = (df['low'] - df['close'].shift()).abs()
        df['tr']         = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        return df

    # --- DX and ADX (RMA) ---
    @staticmethod
    def calculate_adx(df):
        if 'atr' not in df.columns:
            df = Calculations.calculate_atr(df)
        up_move   = df['high'].diff()
        down_move = -df['low'].diff()
        df['plus_dm']  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        df['minus_dm'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm_rma  = df['plus_dm'].ewm(alpha=1/14, adjust=False).mean()
        minus_dm_rma = df['minus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_di']  = 100 * (plus_dm_rma / df['atr'])
        df['minus_di'] = 100 * (minus_dm_rma / df['atr'])
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        df['dx'] = df['dx'].fillna(0) 
        df['adx'] = df['dx'].ewm(alpha=1/14, adjust=False).mean()
        return df

    # --- Quantity Calculation ---
    @staticmethod
    def calculate_quantity(lot_size):
        quantity = Config.LOTS * lot_size
        return quantity
    
    # --- Trigger Price Calculation ---
    @staticmethod
    def calculate_trigger_price(option_price):
        if option_price is None:
            logging.error("Option Price not available")
            return None
        trigger_price = option_price - Config.SL_POINTS
        return trigger_price
    
    # --- Exit Price Calculation ---
    @staticmethod
    def calculate_exit_price(option_price):
        if option_price is None:
            logging.error("Option Price not available")
            return None
        exit_price = option_price + Config.TARGET_POINTS
        return exit_price
    
    # --- SL Trigger Calculation ---
    @staticmethod
    def calculate_sl_trigger(option_price, trigger_price, current_atr):
        if option_price is None:
            logging.error("Option Price not available")
            return None
        if option_price >= trigger_price + Config.TRAILING_TRIGGER:
            new_trigger = option_price - (Config.TRAILING_MULTIPLIER * Config.SL_POINTS + current_atr)
            if new_trigger > trigger_price:
                return new_trigger
        return trigger_price
    
    # --- Calculate All Indicators ---
    @staticmethod
    def calculate_indicators(candle_df):
        df = candle_df.copy()
        df = Calculations.calculate_rsi(df)
        df = Calculations.calculate_atr(df)
        df = Calculations.calculate_adx(df)
        df = Calculations.calculate_vwap(df)
        return df
    
class Strategies:

    @staticmethod
    def pre_check_validation(bot):
        if bot.position_active:
            if bot.option_type  == "CE":
                print(f"[Active Trade] CE {bot.ce_strike_price} | Stop Loss: {bot.trigger_price} | Current Price: {bot.ce_option_price} | Target: {bot.exit_price}",end="\r") 
            else:
                print(f"[Active Trade] PE {bot.pe_strike_price} | Stop Loss: {bot.trigger_price} | Current Price: {bot.pe_option_price} | Target: {bot.exit_price}",end="\r") 
            return False
        elif not bot.can_enter_trade():
            print("Bot in Sleep Mode",end="\r") 
            return False
        elif bot.total_trades >= Config.TRADE_LIMIT:
            print("Trade Limit Reached",end="\r") 
            return False
        else:
            return True
        
    @staticmethod
    def vwap_rsi_strategy(bot,option_type,indicator_results:dict):
        if Strategies.pre_check_validation(bot):
            df=indicator_results.get("3")
            df_underlying=bot.candle_df.get("3")
            if df is None:
                logging.warning("3-minute indicator data not available for VWAP RSI strategy")
                return
            if df_underlying is None:
                logging.warning("3-minute underlying data not available for VWAP RSI strategy")
                return
            if len(df) < 15: 
                logging.warning("Not enough data for VWAP RSI signal check") 
                return 
            close_curr = df["close"].iloc[-1]
            close_prev = df["close"].iloc[-2]
            close_last = df["close"].iloc[-3]
            vwap_curr = df["vwap"].iloc[-1]
            vwap_prev = df["vwap"].iloc[-2]
            vwap_last = df["vwap"].iloc[-3]
            rsi_curr = df["rsi"].iloc[-1]
            rsi_prev = df["rsi"].iloc[-2]
            underlying_rsi_curr = df_underlying["rsi"].iloc[-1]

            vwap_crossover_curr = (close_prev < vwap_prev and close_curr > vwap_curr * (1 + Config.VWAP_CLOSE_TOLERANCE))
            vwap_crossover_prev = (close_last < vwap_last and close_prev > vwap_prev * (1 + Config.VWAP_CLOSE_TOLERANCE))
            rsi_borderline_prev = 40 < rsi_prev < Config.RSI_MID_TRESHOLD
            if option_type == "PE":
                rsi_strong = (Config.RSI_MID_TRESHOLD < rsi_curr < Config.RSI_TRESHOLD_HIGH) and (Config.RSI_TRESHOLD_LOW < underlying_rsi_curr < Config.RSI_MID_TRESHOLD)
            else:
                rsi_strong = (Config.RSI_MID_TRESHOLD < rsi_curr < Config.RSI_TRESHOLD_HIGH) and (Config.RSI_MID_TRESHOLD < underlying_rsi_curr < Config.RSI_TRESHOLD_HIGH)

            if vwap_crossover_curr and rsi_strong:
                logging.info(
                    f"[STAGE 1] {option_type} Immediate Entry | "
                    f"VWAP Crossover + RSI Match | "
                    f"Price:{close_curr:.2f} > VWAP:{vwap_curr:.2f} | "
                    f"Und RSI:{underlying_rsi_curr:.2f} | {option_type} RSI:{rsi_curr:.2f} | "
                    f"Index:{bot.index_price}"
                )
                bot.enter_trade(option_type)

            elif vwap_crossover_prev and rsi_strong and rsi_borderline_prev:
                logging.info(
                    f"[STAGE 2] {option_type} Delayed Entry | "
                    f"VWAP Crossover Prev + RSI Match | "
                    f"Price:{close_curr:.2f} > VWAP:{vwap_curr:.2f} | "
                    f"Und RSI:{underlying_rsi_curr:.2f} | {option_type} RSI:{rsi_curr:.2f} | "
                    f"Index:{bot.index_price}"
                )
                bot.enter_trade(option_type)

            elif vwap_crossover_prev and rsi_strong:
                logging.info(
                    f"[STAGE 3] {option_type} RSI threshold hit during monitoring | "
                    f"Price:{close_curr:.2f} | VWAP:{vwap_curr:.2f} | "
                    f"Und RSI:{underlying_rsi_curr:.2f} | {option_type} RSI:{rsi_curr:.2f} | "
                    f"Entering intrabar..."
                )
                bot.enter_trade(option_type)

class Bot:
    def __init__(self):
        self.authenticator=Authenticator()
        self.access_token=self.authenticator.get_access_token()
        self.data_processor=Data_Processor()
        self.data_collector=Data_Collector(self.access_token,Config.DRY_RUN)
        self.kill_switch=False
        self.position_active=False
        self.options_subscribed=False
        self.transcriber=None
        self.total_trades=0
        self.status="OFFLINE"
        self.tick_buffer = []
        self.ce_tick_buffer = []
        self.pe_tick_buffer = []
        self.candle_df={}
        self.historic_df={}
        self.intraday_df={}
        self.ce_candle_df={}
        self.ce_historic_df={}
        self.ce_intraday_df={}
        self.pe_candle_df={}
        self.pe_historic_df={}
        self.pe_intraday_df={}
        self.ce_strike_price=None
        self.pe_strike_price=None
        self.entry_lock=threading.Lock()
        self.exit_lock=threading.Lock()
        self.wake_lock=Wake_Lock()
        self.option_type=None
        self.option_key=None
        self.index_price=None
        self.option_price=None
        self.ce_option_price=None
        self.pe_option_price=None
        self.trigger_price=None
        self.exit_price=None
        self.latest_entry_time=None
        self.order_ids=[]
        self.last_exit_time = None
        self.streamer=None
        self.available_margin=None
        self.name=None
        self.lot_size=None
        self.main_instrument_key=None
        self.ce_instrument_key=None
        self.pe_instrument_key=None
        self.expiry_date=None

    def can_enter_trade(self):
        now_aware = datetime.now(pytz.timezone('Asia/Kolkata'))
        current_time = now_aware.time()
        if current_time < pd.to_datetime(Config.MARKET_OPEN_TIME).time() or current_time > pd.to_datetime(Config.MARKET_CLOSE_TIME).time():
            return False
        if self.last_exit_time is None:
            return True
        now_aware = datetime.now(pytz.timezone('Asia/Kolkata'))
        elapsed = (now_aware - self.last_exit_time).total_seconds()
        return elapsed >= Config.ENTRY_COOLDOWN

    def market_is_open(self):
        now_aware = datetime.now(pytz.timezone('Asia/Kolkata'))
        current_time = now_aware.time()
        return pd.to_datetime(Config.MARKET_OPEN_TIME).time() <= current_time <= pd.to_datetime(Config.MARKET_CLOSE_TIME).time()

    # --- Check for 3-minute marks in case of launching after market opens ---
    def is_three_min_mark(self, ts, tolerance=2):
        # Find the nearest 3-minute boundary before or equal to ts
        boundary_minute = (ts.minute // 3) * 3
        boundary = ts.replace(minute=boundary_minute, second=0, microsecond=0)
        return 0 <= (ts - boundary).total_seconds() <= tolerance
    
    def launch(self):
        self.name=input("Instrument Name :").upper()
        self.available_margin=self.data_collector.get_margin()
        self.transcriber=Transcriber(self.available_margin)
        self.main_instrument_key=self.data_processor.get_instrument_key(self.name)
        self.lot_size=self.data_processor.get_lot_size()
        self.expiry_date=self.data_processor.get_expiry_date()
        historic_dfs=self.data_collector.get_historic_data(self.main_instrument_key)
        if historic_dfs is None:
            logging.critical("Historical Data unavailable. Bot cannot launch safely")
            Alerts.error()
            self.status="ERROR"
            return
        else:
            logging.info("Fetched Historic Data")
            self.historic_df=dict(zip(Config.INTERVALS, historic_dfs))
        intraday_dfs=self.data_collector.get_intraday_data(self.main_instrument_key)
        if intraday_dfs is None:
            logging.critical("Intraday Data unavailable. Bot cannot launch safely")
            Alerts.error()
            self.status="ERROR"
            return
        else:
            logging.info("Fetched Intraday Data")
            self.intraday_df=dict(zip(Config.INTERVALS,intraday_dfs))
        self.start_connection()

    def start_connection(self):
        Config.CONFIGURATION.access_token=self.access_token
        self.streamer = MarketDataStreamerV3(
            upstox_client.ApiClient(Config.CONFIGURATION),
            [self.main_instrument_key],  # Only subscribe to main instrument initially
            mode="full"
        )
        self.streamer.auto_reconnect(True, 5, 3)
        self.streamer.on("open",self.on_open)
        self.streamer.on("message", self.on_message)
        self.streamer.on("error", self.on_error)
        self.streamer.on("close", self.on_close)
        self.streamer.connect()

    def on_open(self):
        self.wake_lock.activate()
        logging.info("Websocket Connection Established")
        self.status="ONLINE"
        Alerts.websocket_connected()

    def on_error(self,error):
        logging.error(f"Websocket Error :{error}")
        self.status="ERROR"
        Alerts.websocket_error()

    def on_close(self,*args):
        logging.info("Websocket Disconnected")
        self.status="OFFLINE"
        Alerts.websocket_disconnected()
        if Config.DRY_RUN:
            Dry_Run_Services.generate_performance_report(self.transcriber)
            Dry_Run_Services.export_trades_to_csv(self.transcriber.trades)
        self.wake_lock.deactivate()

    def on_message(self,message):
        if self.kill_switch:
            logging.warning("Kill Switch Active. Ignoring Incoming Messages")
            return    
        if "feeds" not in message or not message["feeds"]:
            logging.warning("No Feed Data Aailable")
            return
        data = message["feeds"]
        feed_timestamp = pd.to_datetime(int(message['currentTs']),unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')
        for instrument_key, feed_data in data.items():
            try:
                #Analyze index data
                if instrument_key == self.main_instrument_key:
                    ltp = feed_data['fullFeed']['indexFF']['ltpc']['ltp']
                    ltq = feed_data['fullFeed']['indexFF']['ltpc'].get("ltq", 0)
                    self.index_price = ltp
                    self.tick_buffer.append({"timestamp": feed_timestamp, "price": ltp, "ltq": ltq})
                    if self.options_subscribed == False and self.market_is_open():
                        if self.is_three_min_mark(feed_timestamp):
                            self.subscribe_to_options()
                    self.aggregate_candles()
                    if not self.position_active:
                        print(f"[Bot Active] Index LTP: {ltp:<10} | Time: {feed_timestamp.strftime('%H:%M:%S'):>8}", end="\r")
                #Analyze ce option data
                if instrument_key == self.ce_instrument_key:
                    ce_option_ltp = feed_data['fullFeed']['marketFF']['ltpc']['ltp']
                    ce_option_ltq = int(feed_data['fullFeed']['marketFF']['ltpc']['ltq'])
                    self.ce_option_price = ce_option_ltp
                    self.ce_tick_buffer.append({"timestamp": feed_timestamp, "price": ce_option_ltp, "ltq": ce_option_ltq})
                    self.aggregate_ce_candles()
                #Analyze pe option data
                if instrument_key == self.pe_instrument_key:
                    pe_option_ltp = feed_data['fullFeed']['marketFF']['ltpc']['ltp']
                    pe_option_ltq = int(feed_data['fullFeed']['marketFF']['ltpc']['ltq'])
                    self.pe_option_price = pe_option_ltp
                    self.pe_tick_buffer.append({"timestamp": feed_timestamp, "price": pe_option_ltp, "ltq": pe_option_ltq})
                    self.aggregate_pe_candles()
                
                if self.exit_lock.acquire(blocking=False):
                    try:
                        if self.position_active and self.option_type == "CE" and feed_timestamp > (self.latest_entry_time + pd.Timedelta(seconds=1)):
                            current_trade = next((t for t in reversed(self.transcriber.trades) if t.status == "ACTIVE"), None)
                            if not current_trade:
                                logging.warning("No active trade found.")
                                return
                            
                            # Analyze CE option for exit
                            if instrument_key == self.ce_instrument_key:
                                ce_option_ltp = feed_data['fullFeed']['marketFF']['ltpc']['ltp']
                                timestamp = pd.to_datetime(int(message['currentTs']), unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')

                                # Trailing Stop Loss Logic
                                if ce_option_ltp > current_trade.highest_price:
                                    current_trade.highest_price = ce_option_ltp
                                    if ce_option_ltp >= current_trade.entry_price + Config.TRAILING_TRIGGER:
                                        self.trigger_price = max(math.floor(current_trade.entry_price) + Config.MINIMUM_SL_PRICE, ce_option_ltp - Config.SL_POINTS)
                                        current_atr = None
                                        if not self.ce_candle_df[Config.SL_ATR_TIMEFRAME].empty and 'atr' in self.ce_candle_df[Config.SL_ATR_TIMEFRAME].columns:
                                            current_atr = self.ce_candle_df[Config.SL_ATR_TIMEFRAME]["atr"].iloc[-1]
                                        if current_atr is None or pd.isna(current_atr):
                                            current_atr = Config.SL_POINTS
                                        new_trigger = Calculations.calculate_sl_trigger(ce_option_ltp, self.trigger_price, current_atr)
                                        if new_trigger > self.trigger_price:
                                            self.trigger_price = new_trigger
                                            current_trade.trailing_trigger = new_trigger
                                            if not Config.DRY_RUN:
                                                self.order_ids=self.update_stop_loss(self.trigger_price)
                                            logging.info(f"Updated Stop Loss Trigger to {self.trigger_price} based on trailing stop logic.")

                                # Check for Stop Loss Hit
                                if not Config.DRY_RUN:
                                    self.position_active = self.data_collector.check_position()
                                    if not self.position_active:
                                        logging.info(f"Stop loss hit: {ce_option_ltp} <= {self.trigger_price}, position exited via SL-M.")
                                        self.transcriber.record_exit(ce_option_ltp, "STOPLOSS_HIT", timestamp)
                                        self.cleanup_after_exit()
                                        Alerts.trade_exited()
                                        return
                                if ce_option_ltp <= self.trigger_price:
                                    logging.info(f"Stop loss hit: {ce_option_ltp} <= {self.trigger_price}, exiting position.")
                                    if Config.DRY_RUN:
                                        self.position_active = False
                                    else:
                                        self.position_active = self.data_collector.check_position()
                                    if not self.position_active:
                                        self.transcriber.record_exit(ce_option_ltp, "STOPLOSS_HIT", timestamp)
                                        self.cleanup_after_exit()
                                        Alerts.trade_exited()
                                        return
                                    
                                # Check for Target Hit
                                if ce_option_ltp >= self.exit_price:
                                    logging.info(f"Target hit: {ce_option_ltp} >= {self.exit_price}, exiting position.")
                                    if Config.DRY_RUN:
                                        self.position_active = False
                                    else:
                                        self.position_active = self.data_collector.check_position()
                                        self.exit_trade()
                                    if not self.position_active:
                                        self.transcriber.record_exit(ce_option_ltp, "TARGET_HIT", timestamp)
                                        self.cleanup_after_exit()
                                        Alerts.trade_exited()
                                        return
                        
                        elif self.position_active and self.option_type == "PE" and feed_timestamp > (self.latest_entry_time + pd.Timedelta(seconds=1)):
                            current_trade = next((t for t in reversed(self.transcriber.trades) if t.status == "ACTIVE"), None)
                            if not current_trade:
                                logging.warning("No active trade found.")
                                return

                            # Analyze PE option for exit
                            if instrument_key == self.pe_instrument_key:
                                pe_option_ltp = feed_data['fullFeed']['marketFF']['ltpc']['ltp']
                                timestamp = pd.to_datetime(int(message['currentTs']), unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')

                                # Trailing Stop Loss Logic
                                if pe_option_ltp > current_trade.highest_price:
                                    current_trade.highest_price = pe_option_ltp
                                    if pe_option_ltp >= current_trade.entry_price + Config.TRAILING_TRIGGER:
                                        self.trigger_price = max(math.floor(current_trade.entry_price) + Config.MINIMUM_SL_PRICE, pe_option_ltp - Config.SL_POINTS)
                                        current_atr = None
                                        if not self.pe_candle_df[Config.SL_ATR_TIMEFRAME].empty and 'atr' in self.pe_candle_df[Config.SL_ATR_TIMEFRAME].columns:
                                            current_atr = self.pe_candle_df[Config.SL_ATR_TIMEFRAME]["atr"].iloc[-1]
                                        if current_atr is None or pd.isna(current_atr):
                                            current_atr = Config.SL_POINTS
                                        new_trigger = Calculations.calculate_sl_trigger(pe_option_ltp, self.trigger_price, current_atr)
                                        if new_trigger > self.trigger_price:
                                            self.trigger_price = new_trigger
                                            current_trade.trailing_trigger = new_trigger
                                            if not Config.DRY_RUN:
                                                self.order_ids=self.update_stop_loss(self.trigger_price)
                                            logging.info(f"Updated Stop Loss Trigger to {self.trigger_price} based on trailing stop logic.")
                                
                                # Check for Stop Loss Hit
                                if not Config.DRY_RUN:
                                    self.position_active = self.data_collector.check_position()
                                    if not self.position_active:
                                        logging.info(f"Stop loss hit: {pe_option_ltp} <= {self.trigger_price}, position exited via SL-M.")
                                        self.transcriber.record_exit(pe_option_ltp, "STOPLOSS_HIT", timestamp)
                                        self.cleanup_after_exit()
                                        Alerts.trade_exited()
                                        return
                                if pe_option_ltp <= self.trigger_price:
                                    logging.info(f"Stop loss hit: {pe_option_ltp} <= {self.trigger_price}, exiting position.")
                                    if Config.DRY_RUN:
                                        self.position_active = False
                                    else:
                                        self.position_active = self.data_collector.check_position()
                                    if not self.position_active:
                                        self.transcriber.record_exit(pe_option_ltp, "STOPLOSS_HIT", timestamp)
                                        self.cleanup_after_exit()
                                        Alerts.trade_exited()
                                        return
                                    
                                # Check for Target Hit
                                if pe_option_ltp >= self.exit_price:
                                    logging.info(f"Target hit: {pe_option_ltp} >= {self.exit_price}, exiting position.")
                                    if Config.DRY_RUN:
                                        self.position_active = False
                                    else:
                                        self.position_active = self.data_collector.check_position()
                                        self.exit_trade()
                                    if not self.position_active:
                                        self.transcriber.record_exit(pe_option_ltp, "TARGET_HIT", timestamp)
                                        self.cleanup_after_exit()
                                        Alerts.trade_exited()
                                        return
                    finally:
                        self.exit_lock.release()
            except KeyError as e:
                Alerts.error()
                logging.error(f"Data access error: {e} - possibly unexpected data structure")
            except Exception as e:
                Alerts.error()
                logging.error(f"Unexpected error: {e}")

    def aggregate_candles(self):
        df = pd.DataFrame(self.tick_buffer)
        if df.empty:
            print("Waiting for market data...", end="\r")
            return
        df.set_index("timestamp", inplace=True)
        df = df.between_time("09:15", "15:30")
        for interval in Config.INTERVALS:
            historic = self.historic_df.get(interval)
            intraday = self.intraday_df.get(interval)
            self.candle_df[interval] = self.data_processor.convert_to_candles(df, interval, historic, intraday)
            self.candle_df[interval] = Calculations.calculate_indicators(self.candle_df[interval])

    def aggregate_ce_candles(self):
        df = pd.DataFrame(self.ce_tick_buffer)
        if df.empty:
            return
        df.set_index("timestamp", inplace=True)
        df = df.between_time("09:15", "15:30")
        for interval in Config.INTERVALS:
            historic = self.ce_historic_df.get(interval)
            intraday = self.ce_intraday_df.get(interval)
            self.ce_candle_df[interval] = self.data_processor.convert_to_candles(df, interval, historic, intraday)
            self.ce_candle_df[interval] = Calculations.calculate_indicators(self.ce_candle_df[interval])
            if self.entry_lock.acquire(blocking=False):
                try:
                    Strategies.vwap_rsi_strategy(self, "CE", self.ce_candle_df)
                finally:
                    self.entry_lock.release()

    def aggregate_pe_candles(self):
        df = pd.DataFrame(self.pe_tick_buffer)
        if df.empty:
            return
        df.set_index("timestamp", inplace=True)
        df = df.between_time("09:15", "15:30")
        for interval in Config.INTERVALS:
            historic = self.pe_historic_df.get(interval)
            intraday = self.pe_intraday_df.get(interval)
            self.pe_candle_df[interval] = self.data_processor.convert_to_candles(df, interval, historic, intraday)
            self.pe_candle_df[interval] = Calculations.calculate_indicators(self.pe_candle_df[interval])
            if self.entry_lock.acquire(blocking=False):
                try:
                    Strategies.vwap_rsi_strategy(self, "PE", self.pe_candle_df)
                finally:
                    self.entry_lock.release()

    def subscribe_to_options(self):
        self.ce_strike_price, self.pe_strike_price = self.data_processor.get_strike_price(self.index_price)

        # Subscribe to CE option data
        self.ce_instrument_key = self.data_processor.get_option_key("CE")
        if self.ce_instrument_key is None:
            logging.error("CE Instrument Key not found.")
            return
        ce_historic_dfs=self.data_collector.get_historic_data(self.ce_instrument_key)
        if ce_historic_dfs is None:
            logging.critical("CE Historical Data unavailable.")
            Alerts.error()
            return
        else:
            logging.info("Fetched CE Historic Data")
            self.ce_historic_df=dict(zip(Config.INTERVALS, ce_historic_dfs))
        ce_intraday_dfs=self.data_collector.get_intraday_data(self.ce_instrument_key)
        if ce_intraday_dfs is None:
            logging.critical("CE Intraday Data unavailable.")
            Alerts.error()
            return
        else:
            logging.info("Fetched CE Intraday Data")
            self.ce_intraday_df=dict(zip(Config.INTERVALS,ce_intraday_dfs))
        
        # Subscribe to PE option data
        self.pe_instrument_key = self.data_processor.get_option_key("PE")
        if self.pe_instrument_key is None:
            logging.error("PE Instrument Key not found.")
            return
        pe_historic_dfs=self.data_collector.get_historic_data(self.pe_instrument_key)
        if pe_historic_dfs is None:
            logging.critical("PE Historical Data unavailable.")
            Alerts.error()
            return
        else:
            logging.info("Fetched PE Historic Data")
            self.pe_historic_df=dict(zip(Config.INTERVALS, pe_historic_dfs))
        pe_intraday_dfs=self.data_collector.get_intraday_data(self.pe_instrument_key)
        if pe_intraday_dfs is None:
            logging.critical("PE Intraday Data unavailable. Cannot enter trade safely")
            Alerts.error()
            return
        else:
            logging.info("Fetched PE Intraday Data")
            self.pe_intraday_df=dict(zip(Config.INTERVALS,pe_intraday_dfs))

        # Subscribe to option instrument data
        instrument_keys_to_subscribe = [self.main_instrument_key, self.ce_instrument_key, self.pe_instrument_key]
        if self.streamer:
            self.streamer.subscribe(instrument_keys_to_subscribe, mode="full")
            logging.info(f"Subscribed to instruments: {instrument_keys_to_subscribe}")
            self.options_subscribed = True

    def enter_trade(self,option_type):
        if self.position_active:
            logging.info("Trade Active,Skipping Entries")
            return 
        if option_type == "CE":
            option_price = self.ce_option_price
            self.option_key=self.ce_instrument_key
            self.option_type = "CE"
            if option_price is None:
                logging.warning(f"Could not get option price for {option_type}")
                return
            current_atr = None
            if not self.ce_candle_df[Config.SL_ATR_TIMEFRAME].empty and 'atr' in self.ce_candle_df[Config.SL_ATR_TIMEFRAME].columns:
                current_atr = self.ce_candle_df[Config.SL_ATR_TIMEFRAME]['atr'].iloc[-1]
            if current_atr is None or pd.isna(current_atr):
                logging.warning("ATR Unavailable Skipping Entry")
                return
        elif option_type == "PE":
            option_price = self.pe_option_price 
            self.option_key=self.pe_instrument_key
            self.option_type = "PE"
            if option_price is None:
                logging.warning(f"Could not get option price for {option_type}")
                return
            current_atr = None
            if not self.pe_candle_df[Config.SL_ATR_TIMEFRAME].empty and 'atr' in self.pe_candle_df[Config.SL_ATR_TIMEFRAME].columns:
                current_atr = self.pe_candle_df[Config.SL_ATR_TIMEFRAME]['atr'].iloc[-1]
            if current_atr is None or pd.isna(current_atr):
                logging.warning("ATR Unavailable Skipping Entry")
                return
        trigger_price = Calculations.calculate_trigger_price(option_price)
        if trigger_price is None:
            logging.error("Trigger Price Calculation Failed")
            return
        self.trigger_price = math.floor(trigger_price)
        quantity = Calculations.calculate_quantity(self.lot_size)
        if quantity is None:
            logging.error("Quantity Calculation Failed")
            return
        initial_trailing_trigger = Calculations.calculate_sl_trigger(option_price,self.trigger_price,current_atr)
        if initial_trailing_trigger is None:
            logging.error("Initial SL Trigger Calculation Failed")
            return
        exit_price = Calculations.calculate_exit_price(option_price)
        if exit_price is None:
            logging.error("Exit Price Calculation Failed")
            return
        self.exit_price = exit_price
        if self.available_margin < (option_price * quantity):
            logging.error("Insufficient Margin to Enter Trade")
            return
        self.position_active=True
        self.latest_entry_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        trade=Trade(self.option_key,self.option_type,self.latest_entry_time,option_price,quantity,self.trigger_price,self.exit_price,trailing_trigger=initial_trailing_trigger)
        self.transcriber.record_entry(trade)
        self.total_trades+=1
        if Config.DRY_RUN:
            logging.info(f"[DRY RUN]Entered {option_type} Trade | Option Price:{option_price} | Trigger Price:{self.trigger_price} | Exit Price:{self.exit_price} | Quantity:{quantity} | Trailing Trigger:{initial_trailing_trigger}")
        else:
            logging.info(f"Order Sent {option_type} Trade | Option Price:{option_price} | Trigger Price:{self.trigger_price} | Exit Price:{self.exit_price} | Quantity:{quantity} | Trailing Trigger:{initial_trailing_trigger}")
            self.order_ids = self.place_order(quantity)
            if self.order_ids is None:
                logging.error("Order ID Unavailable, Trade Placement Failed")
                self.position_active=False
                return
        Alerts.trade_entered()

    def place_order(self,quantity):
        api_instance = upstox_client.OrderApiV3(upstox_client.ApiClient(Config.CONFIGURATION))
        body = upstox_client.PlaceOrderV3Request(quantity=quantity, product="I", validity="DAY", 
            price=0, tag="order", instrument_token=self.option_key, 
            order_type="SL-M", transaction_type="BUY", disclosed_quantity=0, 
            trigger_price=self.trigger_price, is_amo=False, slice=True)
        try:
            api_response = api_instance.place_order(body)
            placement_status = api_response.status
            if placement_status != "success":
                logging.error(f"Order status: {placement_status}")
                logging.error(f"Order placement failed. Response: {api_response}")
                return None
            order_ids = api_response.data.order_id
            logging.info(f"Order placed successfully. Order IDs: {order_ids}")
            return order_ids
        except ApiException as e:
            Alerts.error()
            logging.error(f"Exception While Placing Order :{e}")
            return None
        
    def exit_trade(self):
        if not self.position_active:
            logging.info("No Active Position to Exit")
            return 
        api_instance = upstox_client.OrderApi(upstox_client.ApiClient(Config.CONFIGURATION))
        try:
            api_response = api_instance.exit_positions()
            exit_status = api_response.status
            if exit_status != "success":
                logging.error(f"Exit status: {exit_status}")
                logging.error(f"Position exit failed. Response: {api_response.errors}")
                return
            order_ids = api_response.data.get("order_ids", [])
            logging.info(f"Exited Positions. Order IDs: {order_ids}")
        except ApiException as e:
            Alerts.error()
            logging.error(f"Exception When Exiting Position :{e}")

    def update_stop_loss(self,new_trigger_price):
        if not self.order_ids:
            logging.error("No order IDs available to update stop loss")
            return self.order_ids
        api_instance = upstox_client.OrderApiV3(upstox_client.ApiClient(Config.CONFIGURATION))
        order_ids=[]
        for order_id in self.order_ids:
            body = upstox_client.ModifyOrderRequest(
                validity="DAY",
                price=0,
                order_id=order_id,
                order_type="SL-M",
                trigger_price=new_trigger_price
            )
            try:
                api_response = api_instance.modify_order(body)
                modification_status = api_response.status
                if modification_status != "success":
                    logging.error(f"Modification status: {modification_status}")
                    logging.error(f"Stop loss trailing failed for Order ID {order_id}. Response: {api_response}")
                    return self.order_ids
                new_order_id = api_response.data.order_id
                logging.info(f"Stop loss updated to {new_trigger_price}.Order ID: {new_order_id}")
                order_ids.append(new_order_id)
            except ApiException as e:
                Alerts.error()
                logging.error(f"Exception While Updating Stop Loss :{e}")
                return self.order_ids
        return order_ids

    def cleanup_after_exit(self):
        self.position_active=False
        self.option_type=None
        self.option_key=None
        self.index_price=None
        self.option_price=None
        self.trigger_price=None
        self.exit_price=None
        self.order_ids=[]
        if not Config.DRY_RUN:
            self.available_margin=self.data_collector.get_margin()
        self.last_exit_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        if hasattr(self, 'latest_entry_time'):
            del self.latest_entry_time
        Alerts.trade_exited()

if __name__=="__main__":
    bot=Bot()
    terminator=Terminator(bot)
    threading.Thread(target=terminator.listen_for_kill, daemon=True).start()
    bot.launch()
