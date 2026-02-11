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
import time
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

class Config:
    API_VERSION = "2.0"
    CONFIGURATION=upstox_client.Configuration()
    TRADE_LIMIT=5
    MARKET_OPEN_TIME = "09:15"
    MARKET_CLOSE_TIME = "15:30"
    TIME_UNIT="minutes"
    INTERVALS=["1","3"]
    ENTRY_COOLDOWN = 10
    LOTS=1
    STRIKE_DIFF=50
    STRIKE_OFFSET=1
    RSI_TRESHOLD_LOW=0
    RSI_MID_TRESHOLD=50
    RSI_TRESHOLD_HIGH=100
    VWAP_CLOSE_TOLERANCE=0
    SL_POINTS=13
    TARGET_POINTS=100
    TRAILING_GAP=10
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
        if self.bot.data_collector.check_position():
            logging.info("Open Position Found. Exiting Trade Before Shutdown")
            self.bot.exit_trade()
        if hasattr(self.bot, "streamer") and self.bot.streamer is not None:
            self.bot.streamer.disconnect()
        logging.info("Bot Stopped Gracefully")

    def emergency_kill(self,event=None):
        self.bot.kill_switch=True
        if hasattr(self.bot, "streamer") and self.bot.streamer is not None:
            self.bot.streamer.disconnect()
        logging.critical("Emergency stop. Bot Terminated>>>")
        sys.exit(1)

class Wake_Lock:
    def __init__(self):
        self.ES_CONTINUOUS       = 0x80000000
        self.ES_SYSTEM_REQUIRED  = 0x00000001
        self.ES_DISPLAY_REQUIRED = 0x00000002
        self.active=False
    
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
        self.futures_key=None
        self.option_key=None

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
                "vtt": lambda x: x.iloc[-1] - x.iloc[0]
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
    
    def get_futures_key(self):
        try:
            futures = self.instruments[
                (self.instruments['segment'] == "NSE_FO") &
                (self.instruments['underlying_symbol'] == self.name) &
                (self.instruments['instrument_type'] == "FUT")
            ]
            futures_sorted = futures.sort_values(by='expiry')
            if not futures_sorted.empty:
                nearest_future = futures_sorted.iloc[0]
                self.futures_key = nearest_future['instrument_key']
                logging.info(f"Nearest Futures Key: {self.futures_key}")
                return self.futures_key
            else:
                logging.warning(f"No Futures Found For Instrument: {self.name}")
                return None
        except Exception as e:
            logging.error(f"Error fetching futures key: {e}")
            return None

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

    def get_option_key(self,order_type,index_price):
        if order_type is None:
            logging.warning("Order type not provided for option key retrieval")
            return None
        if order_type == "CE":
            strike_price = self.get_ce_strike_price(index_price)
        elif order_type == "PE":
            strike_price = self.get_pe_strike_price(index_price)
        self.expiry_date = pd.to_datetime(self.expiry_date)
        option_key = self.instruments[
            (self.instruments['instrument_type'] == order_type) &
            (self.instruments['name'] == self.name) &
            (self.instruments['expiry'] == self.expiry_date) &
            (self.instruments['strike_price'] == strike_price)
        ]['instrument_key']
        if option_key.empty:
            logging.critical(f"Option key not found for {order_type}, {self.name}, {self.expiry_date}, {strike_price}")
            return None
        self.option_key=option_key.squeeze()
        return self.option_key
    
    def get_ce_strike_price(self,index_price):
        if index_price is None:
            logging.warning("Index price not available for strike price calculation")
            return None
        else:
            return (math.floor(index_price / Config.STRIKE_DIFF) * Config.STRIKE_DIFF) - (Config.STRIKE_OFFSET * Config.STRIKE_DIFF)
    
    def get_pe_strike_price(self,index_price):
        if index_price is None:
            logging.warning("Index price not available for strike price calculation")
            return None
        else:
            return (math.floor(index_price / Config.STRIKE_DIFF) * Config.STRIKE_DIFF) + (Config.STRIKE_OFFSET * Config.STRIKE_DIFF)

class Data_Collector:
    def __init__(self,access_token):
        self.access_token=access_token
        Config.CONFIGURATION = upstox_client.Configuration()
        Config.CONFIGURATION.access_token = self.access_token
        self.option_price=None
        self.available_margin=None

    def get_margin(self):
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
            if not api_response or not api_response.data:
                return False
            for pos in api_response.data:
                if pos.quantity != 0:
                    return True
            return False
        except ApiException as e:
            logging.error(f"Exception when fetching position data :{e}")
            return None

class Calculations:

    @staticmethod
    def calculate_vwap(df):
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["date"] = df.index.date
        df["cum_vol_price"] = (df["typical_price"] * df["volume"]).groupby(df["date"]).cumsum()
        df["cum_volume"] = df["volume"].groupby(df["date"]).cumsum()
        df["vwap"] = df["cum_vol_price"] / df["cum_volume"]
        return df

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

    @staticmethod
    def calculate_quantity(lot_size):
        quantity = Config.LOTS * lot_size
        return quantity

    @staticmethod
    def calculate_trigger_price(option_price):
        if option_price is None:
            logging.error("Option Price not available")
            return None
        trigger_price = option_price - Config.SL_POINTS
        return trigger_price

    @staticmethod
    def calculate_exit_price(option_price):
        if option_price is None:
            logging.error("Option Price not available")
            return None
        exit_price = option_price + Config.TARGET_POINTS
        return exit_price

    @staticmethod
    def calculate_indicators(candle_df):
        df = candle_df.copy()
        df = Calculations.calculate_rsi(df)
        df = Calculations.calculate_vwap(df)
        return df

class Strategies:

    @staticmethod
    def pre_check_validation(bot):
        if bot.position_active:
            if bot.option_type  == "CE":
                print(f"[Active Trade] CE | Stop Loss: {bot.trigger_price} | Current Price: {bot.current_price} | Target: {bot.exit_price}  ",end="\r") 
            else:
                print(f"[Active Trade] PE | Stop Loss: {bot.trigger_price} | Current Price: {bot.current_price} | Target: {bot.exit_price}  ",end="\r") 
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
    def vwap_rsi_strategy(bot,indicator_results:dict):
        if Strategies.pre_check_validation(bot):
            df=indicator_results.get("3")
            if df is None:
                logging.warning("3-minute indicator data not available for VWAP RSI strategy")
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

            print(f"Price: {close_curr:.2f} | VWAP: {vwap_curr:.2f} | RSI: {rsi_curr:.2f}  ", end="\r")

            vwap_crossover_up_curr = (close_prev < vwap_prev and close_curr > vwap_curr * (1 + Config.VWAP_CLOSE_TOLERANCE))
            vwap_crossover_up_prev = (close_last < vwap_last and close_prev > vwap_prev * (1 + Config.VWAP_CLOSE_TOLERANCE))
            rsi_borderline_up_prev = 40 < rsi_prev < Config.RSI_MID_TRESHOLD
            rsi_strong_up = (Config.RSI_MID_TRESHOLD < rsi_curr < Config.RSI_TRESHOLD_HIGH)
            vwap_crossover_down_curr = (close_prev > vwap_prev and close_curr < vwap_curr * (1 - Config.VWAP_CLOSE_TOLERANCE))
            vwap_crossover_down_prev = (close_last > vwap_last and close_prev < vwap_prev * (1 - Config.VWAP_CLOSE_TOLERANCE))
            rsi_borderline_down_prev = 60 > rsi_prev > Config.RSI_MID_TRESHOLD
            rsi_strong_down = (Config.RSI_TRESHOLD_LOW < rsi_curr < Config.RSI_MID_TRESHOLD)

            if vwap_crossover_up_curr and rsi_strong_up:
                logging.info(f"[STAGE 1] CE Immediate Entry VWAP Crossover + RSI Match | Price:{close_curr:.2f} > VWAP:{vwap_curr:.2f} | RSI:{rsi_curr:.2f} | Index:{bot.index_price}")
                bot.enter_trade("CE")

            elif vwap_crossover_up_prev and rsi_strong_up and rsi_borderline_up_prev:
                logging.info(f"[STAGE 2] CE Delayed Entry VWAP Crossover Prev + RSI Match | Price:{close_curr:.2f} > VWAP:{vwap_curr:.2f} | RSI:{rsi_curr:.2f} | Index:{bot.index_price}")
                bot.enter_trade("CE")

            elif vwap_crossover_down_curr and rsi_strong_down:
                logging.info(f"[STAGE 1] PE Immediate Entry VWAP Crossover + RSI Match | Price:{close_curr:.2f} < VWAP:{vwap_curr:.2f} | RSI:{rsi_curr:.2f} | Index:{bot.index_price}")
                bot.enter_trade("PE")

            elif vwap_crossover_down_prev and rsi_strong_down and rsi_borderline_down_prev:
                logging.info(f"[STAGE 2] PE Delayed Entry VWAP Crossover Prev + RSI Match | Price:{close_curr:.2f} < VWAP:{vwap_curr:.2f} | RSI:{rsi_curr:.2f} | Index:{bot.index_price}")
                bot.enter_trade("PE")

class Bot:
    def __init__(self):
        self.authenticator=Authenticator()
        self.access_token=self.authenticator.get_access_token()
        self.data_processor=Data_Processor()
        self.data_collector=Data_Collector(self.access_token)
        self.kill_switch=False
        self.position_active=False
        self.total_trades=0
        self.status="OFFLINE"
        self.tick_buffer = []
        self.candle_df={}
        self.historic_df={}
        self.intraday_df={}
        self.entry_lock=threading.Lock()
        self.exit_lock=threading.Lock()
        self.wake_lock=Wake_Lock()
        self.option_type=None
        self.option_key=None
        self.index_price=None
        self.futures_price=None
        self.strike_price=None
        self.option_price=None
        self.current_price=None
        self.entry_price=None
        self.trigger_price=None
        self.exit_price=None
        self.latest_entry_time=None
        self.last_exit_time = None
        self.streamer=None
        self.available_margin=None
        self.name=None
        self.lot_size=None
        self.main_instrument_key=None
        self.futures_key=None
        self.expiry_date=None

    def can_enter_trade(self):
        return self.market_is_open() and self.cooldown_has_passed()

    def cooldown_has_passed(self):
        if self.last_exit_time is None:
            return True
        now_aware = datetime.now(pytz.timezone('Asia/Kolkata'))
        elapsed = (now_aware - self.last_exit_time).total_seconds()
        return elapsed >= Config.ENTRY_COOLDOWN

    def market_is_open(self):
        now_aware = datetime.now(pytz.timezone('Asia/Kolkata'))
        current_time = now_aware.time()
        return pd.to_datetime(Config.MARKET_OPEN_TIME).time() <= current_time <= pd.to_datetime(Config.MARKET_CLOSE_TIME).time()

    def is_three_min_mark(self, ts, tolerance=2):
        boundary_minute = (ts.minute // 3) * 3
        boundary = ts.replace(minute=boundary_minute, second=0, microsecond=0)
        return 0 <= (ts - boundary).total_seconds() <= tolerance

    def launch(self):
        self.name=input("Instrument Name :").upper()
        self.available_margin=self.data_collector.get_margin()
        self.main_instrument_key=self.data_processor.get_instrument_key(self.name)
        self.lot_size=self.data_processor.get_lot_size()
        self.expiry_date=self.data_processor.get_expiry_date()
        self.futures_key=self.data_processor.get_futures_key()
        while not (self.is_three_min_mark(datetime.now()) and self.market_is_open()):
            now = datetime.now().strftime("%H:%M:%S")
            print(f"Bot idle → waiting for 3‑minute mark/market open [{now}]", end="\r")
            time.sleep(0.5)
            if self.kill_switch:
                return
        if self.preload_futures_data():
            self.start_connection()

    def preload_futures_data(self):
        historic_dfs=self.data_collector.get_historic_data(self.futures_key)
        if historic_dfs is None:
            logging.critical("Futures Historical Data unavailable. Bot cannot launch safely")
            Alerts.error()
            self.status="ERROR"
            return False
        else:
            logging.info("Fetched Futures Historic Data")
            self.historic_df=dict(zip(Config.INTERVALS, historic_dfs))
        intraday_dfs=self.data_collector.get_intraday_data(self.futures_key)
        if intraday_dfs is None:
            logging.critical("Futures Intraday Data unavailable. Bot cannot launch safely")
            Alerts.error()
            self.status="ERROR"
            return False
        else:
            logging.info("Fetched Futures Intraday Data")
            self.intraday_df=dict(zip(Config.INTERVALS,intraday_dfs))
        return True

    def start_connection(self):
        instruments_to_subscribe=[self.main_instrument_key,self.futures_key]
        Config.CONFIGURATION.access_token=self.access_token
        self.streamer = MarketDataStreamerV3(
            upstox_client.ApiClient(Config.CONFIGURATION),
            instruments_to_subscribe,
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
        self.wake_lock.deactivate()

    def on_message(self,message):
        if self.kill_switch:
            logging.warning("Kill Switch Active. Ignoring Incoming Messages")
            return    
        if "feeds" not in message or not message["feeds"]:
            logging.warning("No Feed Data Available")
            return
        data = message["feeds"]
        feed_timestamp = pd.to_datetime(int(message['currentTs']),unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')
        for instrument_key, feed_data in data.items():
            try:
                #Analyze Index Data
                if instrument_key == self.main_instrument_key:
                    ltp = feed_data['fullFeed']['indexFF']['ltpc']['ltp']
                    self.index_price = ltp
                #Analyze Futures Data
                if instrument_key == self.futures_key:
                    ltp = feed_data['fullFeed']['marketFF']['ltpc']['ltp']
                    vtt = int(feed_data['fullFeed']['marketFF']['vtt'])
                    self.futures_price = ltp
                    self.tick_buffer.append({"timestamp": feed_timestamp, "price": ltp, "vtt": vtt})
                    self.aggregate_candles()
                if self.exit_lock.acquire(blocking=False):
                    try:
                        if hasattr(self, 'option_key') and instrument_key == self.option_key and self.position_active and feed_timestamp>(self.latest_entry_time + pd.Timedelta(seconds=1)):
                            ltp = feed_data['fullFeed']['marketFF']['ltpc']['ltp']
                            self.current_price=ltp
                            timestamp = pd.to_datetime(int(message['currentTs']), unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')

                            if not self.data_collector.check_position():
                                logging.info(f"Trade Exited By Broker: {ltp} at {timestamp}")
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
            return
        df.set_index("timestamp", inplace=True)
        df = df.between_time("09:15", "15:30")
        for interval in Config.INTERVALS:
            historic = self.historic_df.get(interval)
            intraday = self.intraday_df.get(interval)
            self.candle_df[interval] = self.data_processor.convert_to_candles(df, interval, historic, intraday)
            self.candle_df[interval] = Calculations.calculate_indicators(self.candle_df[interval])

        if self.entry_lock.acquire(blocking=False):
            try:
                Strategies.vwap_rsi_strategy(self,self.candle_df)
            finally:
                self.entry_lock.release()
        else:
            logging.info("Entry lock busy, Skipping signal check")

    def enter_trade(self,option_type):
        if self.position_active:
            logging.info("Trade Active,Skipping Entries")
            return
        self.option_type = option_type 
        self.option_key = self.data_processor.get_option_key(option_type,self.index_price)
        if self.option_key is None:
            logging.warning(f"Could not find option key for {option_type}")
            return
        self.option_price=self.data_collector.get_option_price(self.option_key)
        if self.option_price is None:
            logging.warning(f"Could not get option price for {option_type}")
            return
        self.entry_price=self.option_price
        self.trigger_price=Calculations.calculate_trigger_price(self.option_price)
        if self.trigger_price is None:
            logging.error("Trigger Price Calculation Failed")
            return
        quantity = Calculations.calculate_quantity(self.lot_size)
        if quantity <= 0:
            logging.warning("Lot Size Must Be Greater Than Zero")
            return
        self.exit_price=Calculations.calculate_exit_price(self.option_price)
        self.latest_entry_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        logging.info(f"{option_type} Trade: Price={self.option_price}, Qty={quantity}, "
              f"Trigger={self.trigger_price}, Target={self.exit_price}")
        instrument_keys_to_subscribe = [self.main_instrument_key,self.futures_key, self.option_key]
        self.streamer.subscribe(instrument_keys_to_subscribe,"full")
        logging.info(f"Subscribing to {self.option_key}")
        self.place_order(quantity)
        Alerts.trade_entered()

    def place_order(self, quantity):
        api_instance = upstox_client.OrderApiV3(upstox_client.ApiClient(Config.CONFIGURATION))

        try:
            entry_rule = upstox_client.GttRule(
                strategy="ENTRY",
                trigger_type="IMMEDIATE",
                trigger_price=self.entry_price
            )

            target_rule = upstox_client.GttRule(
                strategy="TARGET",
                trigger_type="IMMEDIATE",
                trigger_price=self.exit_price
            )
            stoploss_rule = upstox_client.GttRule(
                strategy="STOPLOSS",
                trigger_type="IMMEDIATE",
                trigger_price=self.trigger_price,
                trailing_gap=Config.TRAILING_GAP
            )

            gtt_body = upstox_client.GttPlaceOrderRequest(
                type="MULTIPLE",
                quantity=quantity,
                product="I",
                rules=[entry_rule, target_rule, stoploss_rule],
                instrument_token=self.option_key,
                transaction_type="BUY"
            )

            gtt_response = api_instance.place_gtt_order(body=gtt_body)
            order_ids=gtt_response.data.gtt_order_ids
            time.sleep(3)

            if self.data_collector.check_position():
                logging.info("Order + GTT exits placed successfully")
                self.total_trades += 1
                self.position_active = True
            else:
                logging.error("Order Placement Failed")
                self.cancel_gtt_orders(order_ids)

        except ApiException as e:
            logging.error("Exception when calling OrderApi: %s\n" % e)

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

    def cancel_gtt_orders(self, order_ids):
        api_instance = upstox_client.OrderApiV3(upstox_client.ApiClient(Config.CONFIGURATION))
        for order_id in order_ids:
            body = upstox_client.GttCancelOrderRequest(gtt_order_id=order_id)
            try:
                api_response = api_instance.cancel_gtt_order(body=body)
                logging.info(f"GTT order canceled: {order_id}")
            except ApiException as e:
                logging.error(f"Exception when canceling GTT order {order_id}: %s\n" % e)

    def cleanup_after_exit(self):
        self.streamer.unsubscribe([self.option_key])
        self.streamer.subscribe([self.main_instrument_key,self.futures_key], "full")
        self.position_active=False
        self.option_type=None
        self.option_key=None
        self.option_price=None
        self.trigger_price=None
        self.exit_price=None
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
