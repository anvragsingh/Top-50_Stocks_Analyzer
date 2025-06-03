from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os
import threading
import time
import warnings
from typing import Tuple, Dict, List, Optional
import logging

# Machine Learning imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Popular stocks across different sectors
POPULAR_STOCKS = {
    'Technology': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC'],
    'Consumer': ['KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT'],
    'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX']
}

# Global variable to store real-time data
real_time_data = {}

class StockDataValidator:
    """Validates stock data quality and completeness."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns and reasonable data."""
        try:
            # Check required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for sufficient data
            if len(df) < 30:
                raise ValueError("Insufficient data: need at least 30 rows for meaningful analysis")
            
            # Check for reasonable price values
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    if (df[col] <= 0).any():
                        raise ValueError(f"Invalid {col} prices: found non-positive values")
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise

class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis."""
    
    @staticmethod
    def moving_averages(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """Add multiple moving averages."""
        df = df.copy()
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(df) >= period:
                df[f'MA_{period}'] = df[price_col].rolling(window=period).mean()
            
        # Add exponential moving averages
        if len(df) >= 12:
            df['EMA_12'] = df[price_col].ewm(span=12).mean()
        if len(df) >= 26:
            df['EMA_26'] = df[price_col].ewm(span=26).mean()
        
        return df
    
    @staticmethod
    def rsi(df: pd.DataFrame, price_col: str = 'Close', window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        if len(df) < window:
            window = max(5, len(df) // 2)
        
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        loss = loss.replace(0, 0.0001)  # Avoid division by zero
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def macd(df: pd.DataFrame, price_col: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(df) >= 26:
            ema_12 = df[price_col].ewm(span=12).mean()
            ema_26 = df[price_col].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
        else:
            # Simplified MACD for shorter periods
            short_span = max(5, len(df) // 4)
            long_span = max(10, len(df) // 2)
            ema_short = df[price_col].ewm(span=short_span).mean()
            ema_long = df[price_col].ewm(span=long_span).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=max(3, len(df) // 6)).mean()
            histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, price_col: str = 'Close', window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if len(df) < window:
            window = max(5, len(df) // 2)
        
        rolling_mean = df[price_col].rolling(window=window).mean()
        rolling_std = df[price_col].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, lower_band, rolling_mean

class StockPredictor:
    """Advanced stock price prediction with machine learning."""
    
    def __init__(self, forecast_horizon: int = 5):
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        self.model_scores = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive feature set with technical indicators."""
        df = df.copy()
        
        # Add basic price features
        df['price_change'] = df['Close'].pct_change()
        df['price_change_lag1'] = df['price_change'].shift(1)
        df['high_low_ratio'] = df['High'] / df['Low']
        df['open_close_ratio'] = df['Open'] / df['Close']
        
        # Add moving averages
        df = TechnicalIndicators.moving_averages(df)
        
        # Add RSI
        df['RSI'] = TechnicalIndicators.rsi(df)
        
        # Add MACD
        macd, signal, histogram = TechnicalIndicators.macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        # Add Bollinger Bands
        bb_upper, bb_lower, bb_middle = TechnicalIndicators.bollinger_bands(df)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # Add volatility
        df['Volatility_10'] = df['Close'].rolling(window=min(10, len(df)//2)).std()
        
        # Add momentum
        momentum_period = min(5, len(df)//3)
        if momentum_period > 0:
            df['Momentum'] = df['Close'] / df['Close'].shift(momentum_period) - 1
        
        # Add target variable (future prices)
        df['target'] = df['Close'].shift(-self.forecast_horizon)
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select relevant features for modeling."""
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove features with too many NaN values
        nan_threshold = 0.5
        feature_cols = [col for col in feature_cols 
                       if df[col].isna().sum() / len(df) < nan_threshold]
        
        return feature_cols
    
    def train_and_predict(self, df: pd.DataFrame) -> Dict:
        """Train models and make predictions."""
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            df_clean = df_features.dropna()
            
            if len(df_clean) < 30:
                raise ValueError("Insufficient clean data for prediction")
            
            self.feature_columns = self.select_features(df_clean)
            
            X = df_clean[self.feature_columns].values
            y = df_clean['target'].values
            
            # Remove rows where target is NaN
            valid_idx = ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 20:
                raise ValueError("Insufficient data after cleaning")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train models
            models_config = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            }
            
            predictions = {}
            
            for name, model in models_config.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    self.models[name] = model
                    self.model_scores[name] = {
                        'R2': max(0, min(1, r2)),  # Clamp between 0 and 1
                        'RMSE': rmse,
                        'MAE': mae
                    }
                    
                    # Make prediction for next period
                    if len(X_scaled) > 0:
                        last_features = X_scaled[-1].reshape(1, -1)
                        pred = model.predict(last_features)[0]
                        predictions[name] = pred
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {}

def clean_dataframe(df):
    """Clean dataframe to handle data type issues"""
    try:
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(how='all')
        df = df.fillna(method='ffill')
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error cleaning dataframe: {e}")
        return df

def calculate_technical_indicators(df):
    """Calculate various technical indicators with error handling"""
    try:
        df = clean_dataframe(df)
        
        if df.empty:
            return df
        
        data_length = len(df)
        
        # Simple Moving Averages
        periods = [20, 50]
        for period in periods:
            if data_length >= period:
                df[f'SMA{period}'] = df['Close'].rolling(window=period).mean()
            else:
                df[f'SMA{period}'] = df['Close']
        
        # Exponential Moving Averages
        if data_length >= 12:
            df['EMA12'] = df['Close'].ewm(span=12).mean()
            df['EMA26'] = df['Close'].ewm(span=26).mean()
        else:
            df['EMA12'] = df['Close']
            df['EMA26'] = df['Close']
        
        # MACD
        if data_length >= 26:
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        else:
            df['MACD'] = pd.Series(0, index=df.index)
            df['MACD_Signal'] = pd.Series(0, index=df.index)
            df['MACD_Histogram'] = pd.Series(0, index=df.index)
        
        # RSI
        df['RSI'] = TechnicalIndicators.rsi(df)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = TechnicalIndicators.bollinger_bands(df)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['BB_Middle'] = bb_middle
        
        # Trading Signals
        if data_length >= 20:
            df['Signal'] = 0
            df.loc[df['SMA20'] > df['SMA50'], 'Signal'] = 1
            df['Position'] = df['Signal'].diff()
        else:
            df['Signal'] = pd.Series(0, index=df.index)
            df['Position'] = pd.Series(0, index=df.index)
        
        # Clean up inf/-inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return df

def create_advanced_plot_with_prediction(df, ticker, indicators, predictions=None, model_scores=None):
    """Create advanced plot with prediction and accuracy bars"""
    try:
        # Determine number of subplots based on whether we have predictions
        n_subplots = 4 if predictions else 3
        height_ratios = [3, 1, 1, 1] if predictions else [3, 1, 1]
        
        fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 14), 
                                gridspec_kw={'height_ratios': height_ratios})
        
        # Main price chart
        ax1 = axes[0]
        valid_data = df.dropna(subset=['Close'])
        
        if valid_data.empty:
            raise ValueError("No valid price data to plot")
        
        # Plot price and indicators
        ax1.plot(valid_data.index, valid_data['Close'], label='Close Price', linewidth=2, color='black')
        
        if 'sma' in indicators:
            if 'SMA20' in valid_data.columns:
                sma20_data = valid_data.dropna(subset=['SMA20'])
                if not sma20_data.empty:
                    ax1.plot(sma20_data.index, sma20_data['SMA20'], label='SMA20', alpha=0.8, color='blue')
            
            if 'SMA50' in valid_data.columns:
                sma50_data = valid_data.dropna(subset=['SMA50'])
                if not sma50_data.empty:
                    ax1.plot(sma50_data.index, sma50_data['SMA50'], label='SMA50', alpha=0.8, color='red')
        
        if 'bollinger' in indicators:
            bb_cols = ['BB_Upper', 'BB_Lower']
            if all(col in valid_data.columns for col in bb_cols):
                bb_data = valid_data.dropna(subset=bb_cols)
                if not bb_data.empty:
                    ax1.plot(bb_data.index, bb_data['BB_Upper'], label='BB Upper', alpha=0.5, color='gray', linestyle='--')
                    ax1.plot(bb_data.index, bb_data['BB_Lower'], label='BB Lower', alpha=0.5, color='gray', linestyle='--')
                    ax1.fill_between(bb_data.index, bb_data['BB_Upper'], bb_data['BB_Lower'], alpha=0.1, color='gray')
        
        # Add prediction line if available
        if predictions:
            current_price = valid_data['Close'].iloc[-1]
            avg_prediction = np.mean(list(predictions.values()))
            
            # Create future dates for prediction
            last_date = valid_data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=6, freq='D')[1:]
            
            # Create prediction line
            pred_prices = np.linspace(current_price, avg_prediction, 5)
            ax1.plot(future_dates, pred_prices, 'r--', linewidth=2, label=f'Prediction: ${avg_prediction:.2f}', alpha=0.8)
            ax1.scatter(future_dates[-1], avg_prediction, color='red', s=100, zorder=5)
        
        ax1.set_title(f"{ticker.upper()} - Technical Analysis with ML Prediction", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # MACD
        ax2 = axes[1]
        if 'MACD' in valid_data.columns:
            macd_data = valid_data.dropna(subset=['MACD'])
            if not macd_data.empty:
                ax2.plot(macd_data.index, macd_data['MACD'], label='MACD', color='blue')
                if 'MACD_Signal' in macd_data.columns:
                    signal_data = macd_data.dropna(subset=['MACD_Signal'])
                    if not signal_data.empty:
                        ax2.plot(signal_data.index, signal_data['MACD_Signal'], label='Signal', color='red')
                if 'MACD_Histogram' in macd_data.columns:
                    hist_data = macd_data.dropna(subset=['MACD_Histogram'])
                    if not hist_data.empty:
                        ax2.bar(hist_data.index, hist_data['MACD_Histogram'], label='Histogram', alpha=0.6, color='green')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel("MACD", fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # RSI
        ax3 = axes[2]
        if 'RSI' in valid_data.columns:
            rsi_data = valid_data.dropna(subset=['RSI'])
            if not rsi_data.empty:
                ax3.plot(rsi_data.index, rsi_data['RSI'], label='RSI', color='purple', linewidth=2)
        
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.fill_between(valid_data.index, 30, 70, alpha=0.1, color='yellow')
        ax3.set_ylabel("RSI", fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Model Accuracy Bar (if predictions available)
        if predictions and model_scores:
            ax4 = axes[3]
            
            model_names = list(model_scores.keys())
            accuracies = [max(0, min(100, score['R2'] * 100)) for score in model_scores.values()]
            
            colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(model_names)]
            bars = ax4.barh(model_names, accuracies, color=colors, alpha=0.7)
            
            # Add percentage labels on bars
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                ax4.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                        f'{acc:.1f}%', va='center', fontweight='bold')
            
            ax4.set_xlabel('Model Accuracy (%)', fontsize=12)
            ax4.set_ylabel('ML Models', fontsize=12)
            ax4.set_title('Prediction Model Accuracy (R² Score)', fontsize=12, fontweight='bold')
            ax4.set_xlim(0, 100)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add average prediction text
            avg_pred = np.mean(list(predictions.values()))
            current_price = valid_data['Close'].iloc[-1]
            change_pct = ((avg_pred - current_price) / current_price) * 100
            
            prediction_text = f"Avg Prediction: ${avg_pred:.2f} ({change_pct:+.1f}%)"
            ax4.text(0.02, 0.95, prediction_text, transform=ax4.transAxes, 
                    fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Format x-axis for all subplots
        for ax in axes[:-1] if predictions else axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, len(valid_data)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        if not predictions:
            axes[-1].set_xlabel("Date", fontsize=12)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        # Fallback simple plot
        fig, ax = plt.subplots(figsize=(15, 8))
        if not df.empty and 'Close' in df.columns:
            valid_data = df.dropna(subset=['Close'])
            if not valid_data.empty:
                ax.plot(valid_data.index, valid_data['Close'], label='Close Price', linewidth=2)
                ax.set_title(f"{ticker.upper()} - Price Chart", fontsize=16)
                ax.set_ylabel("Price ($)", fontsize=12)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

def get_real_time_price(ticker):
    """Get real-time price for a single stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d", interval="1m")
        
        if not history.empty:
            history = clean_dataframe(history)
            if not history.empty:
                current_price = float(history['Close'].iloc[-1])
                prev_close = float(info.get('previousClose', current_price))
                change = current_price - prev_close
                change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                
                return {
                    'symbol': ticker,
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': int(history['Volume'].iloc[-1]),
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def update_real_time_data():
    """Background function to update real-time data"""
    while True:
        try:
            all_stocks = []
            for sector_stocks in POPULAR_STOCKS.values():
                all_stocks.extend(sector_stocks)
            
            for ticker in all_stocks:
                data = get_real_time_price(ticker)
                if data:
                    real_time_data[ticker] = data
            
            time.sleep(60)
        except Exception as e:
            print(f"Error in background update: {e}")
            time.sleep(60)

@app.route('/')
def index():
    return render_template('index.html', stocks=POPULAR_STOCKS)

@app.route('/analyze', methods=['POST'])
def analyze():
    context = {}
    try:
        ticker = request.form['ticker'].upper()
        period = request.form.get('period', '6mo')
        indicators = request.form.getlist('indicators')
        use_prediction = 'prediction' in request.form.getlist('features')
        
        print(f"Analyzing {ticker} for period {period}, prediction: {use_prediction}")
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            context['error'] = f"No data found for ticker {ticker}"
            return render_template('index.html', stocks=POPULAR_STOCKS, **context)
        
        df = clean_dataframe(df)
        
        if len(df) < 20:
            context['error'] = f"Insufficient data for {ticker}. Only {len(df)} days available."
            return render_template('index.html', stocks=POPULAR_STOCKS, **context)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Initialize prediction variables
        predictions = None
        model_scores = None
        
        # Run prediction if requested and sufficient data
        if use_prediction and len(df) >= 30:
            try:
                predictor = StockPredictor(forecast_horizon=5)
                predictions = predictor.train_and_predict(df)
                model_scores = predictor.model_scores
                print(f"Predictions generated: {predictions}")
                print(f"Model scores: {model_scores}")
            except Exception as e:
                print(f"Prediction failed: {e}")
                context['prediction_warning'] = f"Prediction failed: {str(e)}"
        
        # Create plot with or without predictions
        fig = create_advanced_plot_with_prediction(df, ticker, indicators, predictions, model_scores)
        
        # Save plot
        plot_path = f'static/plot_{ticker}_{int(time.time())}.png'
        os.makedirs('static', exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Get stock info
        try:
            info = stock.info
        except:
            info = {}
        
        # Calculate performance metrics
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
        
        # Prepare context
        context.update({
            'ticker': ticker,
            'image': plot_path,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'current_price': round(df['Close'].iloc[-1], 2),
            'volume': f"{df['Volume'].iloc[-1]:,.0f}",
            'volatility': f"{volatility:.2%}",
            'sharpe_ratio': round(sharpe_ratio, 2),
            'rsi': round(df['RSI'].iloc[-1], 2),
            'data_quality': {
                'days_analyzed': len(df),
                'period_requested': period,
                'prediction_enabled': use_prediction and predictions is not None
            }
        })
        
        # Add prediction results if available
        if predictions and model_scores:
            avg_prediction = np.mean(list(predictions.values()))
            current_price = df['Close'].iloc[-1]
            change_pct = ((avg_prediction - current_price) / current_price) * 100
            
            context['prediction_results'] = {
                'predictions': predictions,
                'model_scores': model_scores,
                'average_prediction': round(avg_prediction, 2),
                'change_percent': round(change_pct, 2),
                'best_model': max(model_scores, key=lambda x: model_scores[x]['R2']) if model_scores else None
            }
        
        # Add technical analysis signals
        latest_data = df.iloc[-1]
        signals = []
        
        # RSI signals
        if latest_data['RSI'] > 70:
            signals.append({'type': 'warning', 'message': 'RSI indicates overbought conditions'})
        elif latest_data['RSI'] < 30:
            signals.append({'type': 'success', 'message': 'RSI indicates oversold conditions'})
        
        # Moving average signals
        if 'SMA20' in df.columns and 'SMA50' in df.columns:
            if latest_data['SMA20'] > latest_data['SMA50']:
                signals.append({'type': 'success', 'message': 'Bullish: SMA20 above SMA50'})
            else:
                signals.append({'type': 'warning', 'message': 'Bearish: SMA20 below SMA50'})
        
        # MACD signals
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if latest_data['MACD'] > latest_data['MACD_Signal']:
                signals.append({'type': 'info', 'message': 'MACD above signal line'})
        
        context['signals'] = signals
        
    except Exception as e:
        print(f"Analysis error: {e}")
        context['error'] = f"Error analyzing {ticker if 'ticker' in locals() else 'stock'}: {str(e)}"
    
    return render_template('index.html', stocks=POPULAR_STOCKS, **context)

@app.route('/realtime')
def realtime():
    """Get real-time data for all tracked stocks"""
    try:
        return jsonify(real_time_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stock/<ticker>')
def get_stock_info(ticker):
    """Get detailed information for a specific stock"""
    try:
        ticker = ticker.upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d")
        
        if history.empty:
            return jsonify({'error': f'No data found for {ticker}'})
        
        latest = history.iloc[-1]
        
        return jsonify({
            'symbol': ticker,
            'name': info.get('longName', ticker),
            'price': round(float(latest['Close']), 2),
            'volume': int(latest['Volume']),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/compare', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks"""
    try:
        tickers = request.json.get('tickers', [])
        period = request.json.get('period', '3mo')
        
        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'Please provide at least 2 stock symbols'})
        
        comparison_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker.upper())
                df = stock.history(period=period)
                
                if not df.empty:
                    df = clean_dataframe(df)
                    returns = df['Close'].pct_change().dropna()
                    
                    comparison_data[ticker.upper()] = {
                        'current_price': round(float(df['Close'].iloc[-1]), 2),
                        'total_return': round(((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100, 2),
                        'volatility': round(returns.std() * (252 ** 0.5) * 100, 2),
                        'max_drawdown': round((df['Close'] / df['Close'].cummax() - 1).min() * 100, 2),
                        'data_points': len(df)
                    }
            except Exception as e:
                comparison_data[ticker.upper()] = {'error': str(e)}
        
        return jsonify(comparison_data)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/watchlist', methods=['GET', 'POST'])
def watchlist():
    """Manage user watchlist"""
    if request.method == 'POST':
        try:
            action = request.json.get('action')
            ticker = request.json.get('ticker', '').upper()
            
            # In a real app, you'd save to database
            # For demo, using session or simple file storage
            watchlist_file = 'watchlist.json'
            
            try:
                with open(watchlist_file, 'r') as f:
                    watchlist = json.load(f)
            except:
                watchlist = []
            
            if action == 'add' and ticker and ticker not in watchlist:
                watchlist.append(ticker)
            elif action == 'remove' and ticker in watchlist:
                watchlist.remove(ticker)
            
            with open(watchlist_file, 'w') as f:
                json.dump(watchlist, f)
            
            return jsonify({'success': True, 'watchlist': watchlist})
        
        except Exception as e:
            return jsonify({'error': str(e)})
    
    else:
        try:
            with open('watchlist.json', 'r') as f:
                watchlist = json.load(f)
        except:
            watchlist = []
        
        return jsonify({'watchlist': watchlist})

@app.route('/alerts', methods=['POST'])
def set_alert():
    """Set price alerts"""
    try:
        ticker = request.json.get('ticker', '').upper()
        price_target = float(request.json.get('price_target', 0))
        alert_type = request.json.get('type', 'above')  # above or below
        
        if not ticker or price_target <= 0:
            return jsonify({'error': 'Invalid ticker or price target'})
        
        # In a real app, you'd save to database and implement alert system
        alerts_file = 'alerts.json'
        
        try:
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        except:
            alerts = []
        
        alert = {
            'id': len(alerts) + 1,
            'ticker': ticker,
            'price_target': price_target,
            'type': alert_type,
            'created': datetime.now().isoformat(),
            'active': True
        }
        
        alerts.append(alert)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f)
        
        return jsonify({'success': True, 'alert': alert})
    
    except Exception as e:
        return jsonify({'error': str(e)})

def check_alerts():
    """Background function to check price alerts"""
    try:
        with open('alerts.json', 'r') as f:
            alerts = json.load(f)
    except:
        return
    
    for alert in alerts:
        if not alert.get('active', True):
            continue
        
        try:
            current_price = get_real_time_price(alert['ticker'])
            if not current_price:
                continue
            
            price = current_price['price']
            target = alert['price_target']
            
            triggered = False
            if alert['type'] == 'above' and price >= target:
                triggered = True
            elif alert['type'] == 'below' and price <= target:
                triggered = True
            
            if triggered:
                print(f"ALERT: {alert['ticker']} hit {price} (target: {target})")
                alert['active'] = False
                # In a real app, send email/SMS notification here
        
        except Exception as e:
            print(f"Error checking alert: {e}")
    
    # Save updated alerts
    try:
        with open('alerts.json', 'w') as f:
            json.dump(alerts, f)
    except Exception as e:
        print(f"Error saving alerts: {e}")

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Start background threads
    real_time_thread = threading.Thread(target=update_real_time_data, daemon=True)
    real_time_thread.start()
    
    # Start alert checking thread
    alert_thread = threading.Thread(
        target=lambda: [check_alerts(), time.sleep(300)] and None, 
        daemon=True
    )
    alert_thread.start()
    
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Flask Stock Analysis Application...")
    print("Available endpoints:")
    print("  / - Main analysis page")
    print("  /realtime - Real-time stock data")
    print("  /stock/<ticker> - Individual stock info")
    print("  /compare - Compare multiple stocks")
    print("  /watchlist - Manage watchlist")
    print("  /alerts - Set price alerts")
    
    app.run(debug=True, host='0.0.0.0', port=5000)