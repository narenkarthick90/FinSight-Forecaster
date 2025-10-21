# stock_forecast_app.py
"""
Stock Price Forecasting System with SmolLM3-3B-Base
A comprehensive chatbot-based forecasting system implementing all components
except D (RAG Retriever), F (Hybrid Signal Fusion), and J (Continual Learning)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Global flags for optional dependencies
SHAP_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP loaded successfully")
except ImportError:
    print("⚠️ SHAP not installed. Feature attribution will use coefficient-based method only.")

# Optional transformer import
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers loaded successfully")
except ImportError:
    print("⚠️ Transformers/Torch not installed. Using template-based narrative generation.")

# Page configuration
st.set_page_config(
    page_title="Stock Forecasting Assistant",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""

    # Conversation state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = 'AWAITING_TICKER'

    # Data storage
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None

    if 'ticker' not in st.session_state:
        st.session_state.ticker = None

    if 'forecast_days' not in st.session_state:
        st.session_state.forecast_days = 7

    # Model and results
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    if 'forecast_engine' not in st.session_state:
        st.session_state.forecast_engine = None

    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

    if 'indicators' not in st.session_state:
        st.session_state.indicators = None

    if 'report' not in st.session_state:
        st.session_state.report = None

    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None

    if 'shap_importance' not in st.session_state:
        st.session_state.shap_importance = None

    if 'llm_narrative' not in st.session_state:
        st.session_state.llm_narrative = None

# ============================================================================
# COMPONENT A: User Query Handler
# ============================================================================
def get_user_query(user_input):
    """
    Component A: get_user_query()
    Processes user input and extracts relevant information
    """
    user_input = user_input.strip().upper()
    return user_input


# ============================================================================
# COMPONENT B: Dialog Manager
# ============================================================================
class DialogManager:
    """
    Component B: dialog_manager()
    Manages conversation flow and context
    """

    @staticmethod
    def process_query(user_input, state):
        """Process user query based on conversation state"""
        response = ""
        new_state = state

        if state == 'AWAITING_TICKER':
            # Extract potential ticker symbol
            ticker = user_input.strip().upper()
            if len(ticker) <= 5 and ticker.isalpha():
                response = f"Great! I'll analyze {ticker}. How many days would you like to forecast? (Enter a number between 1-30)"
                new_state = 'AWAITING_DAYS'
                st.session_state.ticker = ticker
            else:
                response = "Please enter a valid stock ticker symbol (e.g., AAPL, GOOGL, TSLA)."

        elif state == 'AWAITING_DAYS':
            try:
                days = int(user_input)
                if 1 <= days <= 30:
                    st.session_state.forecast_days = days
                    response = f"Perfect! I'll forecast {days} days ahead. Fetching data and generating forecast..."
                    new_state = 'PROCESSING'
                else:
                    response = "Please enter a number between 1 and 30."
            except ValueError:
                response = "Please enter a valid number for forecast days."

        elif state == 'INTERACTIVE':
            response = DialogManager.handle_interactive_query(user_input)
            new_state = 'INTERACTIVE'

        return response, new_state

    @staticmethod
    def handle_interactive_query(query):
        """Handle why/what-if questions"""
        query_lower = query.lower()

        if 'why' in query_lower:
            return "This forecast is based on historical price trends, technical indicators, and linear regression modeling."
        elif 'what if' in query_lower:
            return "What-if analysis: You can modify scenarios by asking questions like 'What if volume increases by 50%?'"
        elif 'explain' in query_lower:
            return "The model uses historical data, moving averages, momentum, and volatility to predict future prices."
        else:
            return "You can ask: 'Why this forecast?', 'What if volume changes?', 'Explain the prediction', or enter a new ticker."


# ============================================================================
# COMPONENT C: Forecasting Input Processing
# ============================================================================
class ForecastingInputProcessor:
    """
    Component C: process_forecasting_inputs()
    Extracts and processes time series data and technical indicators
    """

    @staticmethod
    def fetch_stock_data(ticker, period='3mo'):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                return None

            df.reset_index(inplace=True)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    @staticmethod
    def calculate_technical_indicators(df):
        """Calculate technical indicators"""
        df = df.copy()

        # Moving Averages
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)

        # Volatility (Standard Deviation)
        df['Volatility'] = df['Close'].rolling(window=10).std()

        # Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Price Range
        df['Price_Range'] = df['High'] - df['Low']

        # Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=7).mean()

        # Remove NaN values
        df.dropna(inplace=True)

        return df

    @staticmethod
    def prepare_features(df):
        """Prepare feature matrix for modeling"""
        feature_columns = [
            'SMA_7', 'SMA_20', 'EMA_12', 'Momentum',
            'Volatility', 'ROC', 'RSI', 'Price_Range', 'Volume_MA'
        ]

        X = df[feature_columns].values
        y = df['Close'].values

        return X, y, feature_columns


# ============================================================================
# COMPONENT E: LLM Forecast Engine
# ============================================================================
class LLMForecastEngine:
    """
    Component E: llm_forecast_engine()
    Uses regression for forecasting and SmolLM3 for narrative generation
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.llm_tokenizer = None
        self.llm_model = None

    # Replace the LLMForecastEngine.load_llm method
    # (around line 250-265) with this updated version:

    @st.cache_resource
    def load_llm(_self):
        """Load SmolLM3-3B-Base model"""
        if not TRANSFORMERS_AVAILABLE:
            st.info("🤖 Transformers library not available. Using template-based narrative generation.")
            return None, None

        try:
            with st.spinner("Loading SmolLM2-1.7B-Instruct model (first time may take a few minutes)..."):
                tokenizer = AutoTokenizer.from_pretrained(
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                st.success("✅ LLM model loaded successfully!")
                return tokenizer, model
        except Exception as e:
            st.warning(f"Could not load LLM: {str(e)}. Using template-based generation.")
            return None, None

    def train_forecast_model(self, X, y, feature_names):
        """Train linear regression model"""
        self.feature_names = feature_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Ridge regression (helps with multicollinearity)
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)

        # Calculate metrics
        predictions = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'coefficients': self.model.coef_
        }

    def predict_future(self, last_features, days=7):
        """Generate future predictions"""
        predictions = []
        current_features = last_features.copy()

        for i in range(days):
            # Scale and predict
            scaled_features = self.scaler.transform(current_features.reshape(1, -1))
            predicted_price = self.model.predict(scaled_features)[0]
            predictions.append(predicted_price)

            # Update features for next prediction (simplified)
            # In a real scenario, you'd update all features based on the new price
            current_features = current_features * 0.98 + current_features * 0.02 * np.random.randn(
                len(current_features))

        return np.array(predictions)

    def generate_narrative_with_llm(self, forecast_data, indicators):
        """Generate narrative using SmolLM3"""
        if self.llm_tokenizer is None or self.llm_model is None:
            return self._generate_template_narrative(forecast_data, indicators)

        try:
            prompt = f"""Analyze this stock forecast data and provide a brief, professional summary:

Current Price: ${forecast_data['current_price']:.2f}
Predicted Price (7 days): ${forecast_data['predicted_price']:.2f}
Price Change: {forecast_data['price_change']:.2f}%
Trend: {forecast_data['trend']}

Technical Indicators:
- 7-Day SMA: {indicators['sma7']:.2f}
- 20-Day SMA: {indicators['sma20']:.2f}
- Momentum: {indicators['momentum']:.2f}
- Volatility: {indicators['volatility']:.2f}

Provide a 2-3 sentence professional analysis:"""

            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            generated = response.split("Provide a 2-3 sentence professional analysis:")[-1].strip()

            return generated if generated else self._generate_template_narrative(forecast_data, indicators)

        except Exception as e:
            st.warning(f"LLM generation failed: {str(e)}. Using template.")
            return self._generate_template_narrative(forecast_data, indicators)

    def _generate_template_narrative(self, forecast_data, indicators):
        """Fallback template-based narrative"""
        trend_word = "bullish" if forecast_data['price_change'] > 0 else "bearish"
        sma_signal = "above" if indicators['sma7'] > indicators['sma20'] else "below"

        narrative = f"""Based on historical analysis, the model predicts a {abs(forecast_data['price_change']):.2f}% {'increase' if forecast_data['price_change'] > 0 else 'decrease'} over the next {forecast_data.get('days', 7)} days. """

        narrative += f"The stock shows a {trend_word} trend. "

        narrative += f"The 7-day moving average is {sma_signal} the 20-day average, indicating short-term {trend_word} momentum. "

        if indicators['momentum'] > 0:
            narrative += f"Positive momentum of {indicators['momentum']:.2f} supports continued upward pressure. "
        else:
            narrative += f"Negative momentum of {abs(indicators['momentum']):.2f} suggests downward pressure. "

        if indicators['volatility'] > 10:
            narrative += "Current volatility indicates higher risk and potential for larger price swings."
        else:
            narrative += "Volatility remains moderate, suggesting relative price stability."

        return narrative


# ============================================================================
# COMPONENT G: Narrative Reasoner
# ============================================================================
class NarrativeReasoner:
    """
    Component G: narrative_reasoner()
    Converts predictions into detailed narrative reports
    """

    @staticmethod
    def generate_report(df, predictions, metrics, indicators):
        """Generate comprehensive narrative report"""
        current_price = df['Close'].iloc[-1]
        predicted_price = predictions[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100

        report = {
            'summary': f"Forecast Summary: ${current_price:.2f} → ${predicted_price:.2f} ({price_change:+.2f}%)",
            'model_performance': f"Model Accuracy: R² = {metrics['r2']:.3f}, RMSE = ${metrics['rmse']:.2f}",
            'technical_analysis': NarrativeReasoner._analyze_technicals(indicators),
            'risk_assessment': NarrativeReasoner._assess_risk(indicators, price_change),
            'recommendation': NarrativeReasoner._generate_recommendation(price_change, indicators)
        }

        return report

    @staticmethod
    def _analyze_technicals(indicators):
        """Analyze technical indicators"""
        analysis = []

        if indicators['sma7'] > indicators['sma20']:
            analysis.append("✓ Short-term trend is positive (SMA 7 > SMA 20)")
        else:
            analysis.append("✗ Short-term trend is negative (SMA 7 < SMA 20)")

        if indicators['rsi'] > 70:
            analysis.append("⚠ RSI indicates overbought conditions")
        elif indicators['rsi'] < 30:
            analysis.append("⚠ RSI indicates oversold conditions")
        else:
            analysis.append("✓ RSI is in neutral range")

        if indicators['momentum'] > 0:
            analysis.append("✓ Positive price momentum")
        else:
            analysis.append("✗ Negative price momentum")

        return "\n".join(analysis)

    @staticmethod
    def _assess_risk(indicators, price_change):
        """Assess investment risk"""
        risk_level = "Medium"

        if indicators['volatility'] > 15:
            risk_level = "High"
        elif indicators['volatility'] < 5:
            risk_level = "Low"

        return f"Risk Level: {risk_level} (Volatility: {indicators['volatility']:.2f})"

    @staticmethod
    def _generate_recommendation(price_change, indicators):
        """Generate investment recommendation"""
        if price_change > 5 and indicators['momentum'] > 0:
            return "📈 Strong Buy Signal: Positive trend with strong momentum"
        elif price_change > 0 and indicators['sma7'] > indicators['sma20']:
            return "📊 Buy Signal: Moderate positive outlook"
        elif price_change < -5 and indicators['momentum'] < 0:
            return "📉 Sell Signal: Negative trend with weak momentum"
        elif price_change < 0:
            return "⚠️ Hold/Sell Signal: Bearish indicators present"
        else:
            return "➡️ Hold Signal: Mixed indicators, monitor closely"


# ============================================================================
# COMPONENT H: Explainability & Feature Attribution
# ============================================================================
class ExplainabilityModule:
    """
    Component H: explainability_module()
    Provides model transparency using SHAP
    """

    @staticmethod
    def calculate_feature_importance(model, X, feature_names):
        """Calculate feature importance using coefficients"""
        importances = np.abs(model.coef_)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': (importances / importances.sum() * 100)
        }).sort_values('Importance', ascending=False)

        return importance_df

    @staticmethod
    def generate_shap_explanation(model, X_scaled, feature_names, num_samples=100):
        """Generate SHAP values for model explanation"""
        try:
            # Use a subset of data for SHAP calculation (faster)
            X_sample = X_scaled[-num_samples:] if len(X_scaled) > num_samples else X_scaled

            # Create SHAP explainer
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)

            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': mean_shap,
                'Percentage': (mean_shap / mean_shap.sum() * 100)
            }).sort_values('SHAP_Value', ascending=False)

            return shap_df

        except Exception as e:
            st.warning(f"SHAP calculation failed: {str(e)}. Using coefficient-based importance.")
            return None


# ============================================================================
# COMPONENT I: Interactive Why/What-If Module
# ============================================================================
class InteractiveModule:
    """
    Component I: interactive_why_what_if()
    Handles user interaction for scenario analysis
    """

    @staticmethod
    def handle_why_question(question, forecast_data, indicators):
        """Answer 'why' questions about the forecast"""
        question_lower = question.lower()

        if 'increase' in question_lower or 'up' in question_lower:
            reasons = []
            if indicators['momentum'] > 0:
                reasons.append(f"Positive momentum of {indicators['momentum']:.2f}")
            if indicators['sma7'] > indicators['sma20']:
                reasons.append("Short-term MA above long-term MA")
            if indicators['rsi'] < 70:
                reasons.append("RSI not in overbought territory")

            return "The forecast predicts an increase because: " + ", ".join(
                reasons) if reasons else "Based on overall positive trend patterns."

        elif 'decrease' in question_lower or 'down' in question_lower:
            reasons = []
            if indicators['momentum'] < 0:
                reasons.append(f"Negative momentum of {indicators['momentum']:.2f}")
            if indicators['sma7'] < indicators['sma20']:
                reasons.append("Short-term MA below long-term MA")

            return "The forecast predicts a decrease because: " + ", ".join(
                reasons) if reasons else "Based on overall negative trend patterns."

        else:
            return "The forecast is based on historical price patterns, technical indicators including moving averages, momentum, and volatility analysis."

    @staticmethod
    def simulate_what_if(scenario, current_prediction, indicators):
        """Simulate what-if scenarios"""
        scenario_lower = scenario.lower()
        modified_prediction = current_prediction
        explanation = ""

        if 'volume' in scenario_lower:
            if 'increase' in scenario_lower:
                modifier = 1.05
                explanation = "If volume increases by 50%, the price typically shows stronger momentum, potentially increasing the forecast by ~5%."
            else:
                modifier = 0.95
                explanation = "If volume decreases, price momentum typically weakens, potentially decreasing the forecast by ~5%."
            modified_prediction = current_prediction * modifier

        elif 'volatility' in scenario_lower:
            if 'increase' in scenario_lower:
                explanation = "Increased volatility means wider price swings and higher uncertainty in the forecast."
                modified_prediction = current_prediction * (1 + np.random.uniform(-0.1, 0.1))
            else:
                explanation = "Decreased volatility suggests more stable price movement and tighter forecast ranges."

        elif 'market' in scenario_lower and 'crash' in scenario_lower:
            modifier = 0.85
            explanation = "In a market crash scenario, the forecast would likely decrease significantly by ~15% or more."
            modified_prediction = current_prediction * modifier

        elif 'market' in scenario_lower and 'rally' in scenario_lower:
            modifier = 1.15
            explanation = "In a market rally scenario, the forecast could increase by ~15% or more."
            modified_prediction = current_prediction * modifier

        else:
            explanation = "Please specify a scenario like: 'What if volume increases?', 'What if market crashes?', or 'What if volatility doubles?'"

        return modified_prediction, explanation


# ============================================================================
# COMPONENT K: Hugging Face Backend (Simplified)
# ============================================================================
class BackendManager:
    """
    Component K: hugging_face_backend()
    Manages model deployment and serves outputs
    """

    @staticmethod
    def initialize_backend():
        """Initialize backend resources"""
        if not st.session_state.model_loaded:
            st.session_state.forecast_engine = LLMForecastEngine()
            # Load LLM model
            tokenizer, model = st.session_state.forecast_engine.load_llm()
            st.session_state.forecast_engine.llm_tokenizer = tokenizer
            st.session_state.forecast_engine.llm_model = model
            st.session_state.model_loaded = True

    @staticmethod
    def log_interaction(ticker, forecast_days, metrics):
        """Log user interactions for monitoring"""
        # In production, this would log to a database or monitoring service
        pass


# ============================================================================
# COMPONENT L: Final Output Generator
# ============================================================================
class FinalOutputGenerator:
    """
    Component L: generate_final_output()
    Aggregates all outputs into comprehensive package
    """

    @staticmethod
    def create_visualization_package(df, predictions, forecast_days, feature_importance):
        """Create visualization package"""

        # Historical + Forecast Chart
        fig_forecast = go.Figure()

        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))

        # Forecast
        last_date = df['Date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig_forecast.update_layout(
            title='Stock Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )

        # Feature Importance Chart
        fig_importance = px.bar(
            feature_importance.head(10),
            x='Percentage',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance',
            labels={'Percentage': 'Importance (%)', 'Feature': 'Technical Indicator'}
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})

        return fig_forecast, fig_importance

    @staticmethod
    def display_metrics_dashboard(metrics, indicators, report):
        """Display comprehensive metrics dashboard"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model R²", f"{metrics['r2']:.3f}")
        with col2:
            st.metric("RMSE", f"${metrics['rmse']:.2f}")
        with col3:
            st.metric("MAE", f"${metrics['mae']:.2f}")
        with col4:
            st.metric("Volatility", f"{indicators['volatility']:.2f}")

        st.divider()

        # Display narrative report
        st.subheader("📊 Forecast Analysis")
        st.info(report['summary'])
        st.success(report['recommendation'])

        with st.expander("📈 Technical Analysis Details"):
            st.text(report['technical_analysis'])
            st.warning(report['risk_assessment'])

        with st.expander("🎯 Model Performance"):
            st.write(report['model_performance'])


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():

    initialize_session_state()

    st.title("📈 Stock Price Forecasting Assistant")
    st.markdown("*Powered by SmolLM3, Scikit-learn, and Advanced Technical Analysis*")

    # Initialize backend
    BackendManager.initialize_backend()

    # Sidebar
    with st.sidebar:
        st.header("📋 System Components")
        st.markdown("""
        **Active Components:**
        - ✅ User Query Handler
        - ✅ Dialog Manager
        - ✅ Input Processing
        - ✅ LLM Forecast Engine
        - ✅ Narrative Reasoner
        - ✅ Explainability Module
        - ✅ Interactive What-If
        - ✅ Backend Manager
        - ✅ Output Generator
        """)

        st.divider()

        if st.button("🔄 Reset Conversation"):
            for key in list(st.session_state.keys()):
                if key != 'model_loaded' and key != 'forecast_engine':
                    del st.session_state[key]
            st.rerun()

    # Main chat interface
    st.subheader("💬 Chat Interface")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg['type'] == 'user':
                st.chat_message("user").write(msg['text'])
            else:
                st.chat_message("assistant").write(msg['text'])

    # User input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message
        st.session_state.messages.append({'type': 'user', 'text': user_input})

        # Process with dialog manager
        dialog_mgr = DialogManager()
        response, new_state = dialog_mgr.process_query(
            user_input,
            st.session_state.conversation_state
        )

        st.session_state.conversation_state = new_state
        st.session_state.messages.append({'type': 'bot', 'text': response})

        # If we're in processing state, run the forecast
        if new_state == 'PROCESSING':
            with st.spinner("Processing forecast..."):
                run_forecast_pipeline()

        st.rerun()


def run_forecast_pipeline():
    """Execute the complete forecasting pipeline"""

    try:
        # Component C: Input Processing
        processor = ForecastingInputProcessor()
        df = processor.fetch_stock_data(st.session_state.ticker)

        if df is None or df.empty:
            st.session_state.messages.append({
                'type': 'bot',
                'text': f"Could not fetch data for {st.session_state.ticker}. Please check the ticker symbol."
            })
            st.session_state.conversation_state = 'AWAITING_TICKER'
            return

        df = processor.calculate_technical_indicators(df)
        X, y, feature_names = processor.prepare_features(df)

        # Component E: Train forecast model
        engine = st.session_state.forecast_engine
        metrics = engine.train_forecast_model(X, y, feature_names)

        # Generate predictions
        last_features = X[-1]
        predictions = engine.predict_future(last_features, st.session_state.forecast_days)

        # Extract indicators
        indicators = {
            'sma7': df['SMA_7'].iloc[-1],
            'sma20': df['SMA_20'].iloc[-1],
            'momentum': df['Momentum'].iloc[-1],
            'volatility': df['Volatility'].iloc[-1],
            'rsi': df['RSI'].iloc[-1]
        }

        # Component G: Generate narrative
        reasoner = NarrativeReasoner()
        report = reasoner.generate_report(df, predictions, metrics, indicators)

        # Component E: LLM narrative generation
        forecast_data = {
            'current_price': df['Close'].iloc[-1],
            'predicted_price': predictions[-1],
            'price_change': ((predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100,
            'trend': 'upward' if predictions[-1] > df['Close'].iloc[-1] else 'downward',
            'days': st.session_state.forecast_days
        }

        llm_narrative = engine.generate_narrative_with_llm(forecast_data, indicators)

        # Component H: Feature importance
        explainer = ExplainabilityModule()
        feature_importance = explainer.calculate_feature_importance(
            engine.model, X, feature_names
        )

        shap_importance = explainer.generate_shap_explanation(
            engine.model, engine.scaler.transform(X), feature_names
        )

        # Component L: Generate final output
        output_gen = FinalOutputGenerator()

        # Store results in session state
        st.session_state.stock_data = df
        st.session_state.predictions = predictions
        st.session_state.metrics = metrics
        st.session_state.indicators = indicators
        st.session_state.report = report
        st.session_state.feature_importance = feature_importance
        st.session_state.shap_importance = shap_importance
        st.session_state.llm_narrative = llm_narrative

        # Display results
        st.session_state.messages.append({
            'type': 'bot',
            'text': f"✅ Forecast complete for {st.session_state.ticker}!"
        })

        st.session_state.messages.append({
            'type': 'bot',
            'text': f"**LLM Analysis:**\n\n{llm_narrative}"
        })

        # Component L: Display comprehensive dashboard
        st.divider()
        output_gen.display_metrics_dashboard(metrics, indicators, report)

        # Visualizations
        fig_forecast, fig_importance = output_gen.create_visualization_package(
            df, predictions, st.session_state.forecast_days, feature_importance
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            if shap_importance is not None:
                fig_shap = px.bar(
                    shap_importance.head(10),
                    x='Percentage',
                    y='Feature',
                    orientation='h',
                    title='SHAP Feature Attribution',
                    labels={'Percentage': 'Attribution (%)', 'Feature': 'Feature'}
                )
                fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("SHAP values not available")

        # Technical indicators table
        with st.expander("View Technical Indicators"):
            indicators_df = pd.DataFrame({
                'Indicator': ['7-Day SMA', '20-Day SMA', 'Momentum', 'Volatility', 'RSI'],
                'Value': [
                    f"${indicators['sma7']:.2f}",
                    f"${indicators['sma20']:.2f}",
                    f"{indicators['momentum']:.2f}",
                    f"{indicators['volatility']:.2f}",
                    f"{indicators['rsi']:.2f}"
                ]
            })
            st.dataframe(indicators_df, use_container_width=True)

        # Feature importance table
        with st.expander("🎯 View Feature Importance Details"):
            st.dataframe(feature_importance, use_container_width=True)

        # Historical data
        with st.expander("📈 View Historical Data"):
            st.dataframe(df.tail(20), use_container_width=True)

        # Prediction details
        with st.expander("🔮 View Forecast Details"):
            forecast_dates = pd.date_range(
                start=df['Date'].iloc[-1] + timedelta(days=1),
                periods=st.session_state.forecast_days
            )
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted Price': [f"${p:.2f}" for p in predictions]
            })
            st.dataframe(forecast_df, use_container_width=True)

        st.divider()

        # Component I: Interactive module
        st.session_state.messages.append({
            'type': 'bot',
            'text': "You can now ask:\n- 'Why is the forecast increasing/decreasing?'\n- 'What if volume increases by 50%?'\n- 'What if market crashes?'\n- Or enter a new ticker to analyze another stock."
        })

        st.session_state.conversation_state = 'INTERACTIVE'

        # Log interaction (Component K)
        BackendManager.log_interaction(
            st.session_state.ticker,
            st.session_state.forecast_days,
            metrics
        )

    except Exception as e:
        st.error(f"Error during forecast: {str(e)}")
        st.session_state.messages.append({
            'type': 'bot',
            'text': f"❌ An error occurred: {str(e)}\n\nPlease try again with a different ticker."
        })
        st.session_state.conversation_state = 'AWAITING_TICKER'


# Replace the handle_interactive_mode() function (around line 880-970) with this:

def handle_interactive_mode():
    """Handle interactive mode queries"""

    # Safe checks for session state
    if (st.session_state.get('conversation_state') != 'INTERACTIVE' or
            st.session_state.get('stock_data') is None):
        return

    st.subheader("🔍 Interactive Analysis")

    tab1, tab2, tab3 = st.tabs(["❓ Why Questions", "🔄 What-If Scenarios", "📊 Custom Analysis"])

    with tab1:
        st.markdown("### Ask 'Why' Questions")
        why_questions = [
            "Why is the forecast increasing?",
            "Why is the forecast decreasing?",
            "Why is this prediction made?"
        ]

        why_question = st.selectbox("Select a question:", why_questions)

        if st.button("Get Answer", key="why_btn"):
            interactive_mod = InteractiveModule()
            answer = interactive_mod.handle_why_question(
                why_question,
                {
                    'current_price': st.session_state.stock_data['Close'].iloc[-1],
                    'predicted_price': st.session_state.predictions[-1]
                },
                st.session_state.indicators
            )
            st.info(answer)

    with tab2:
        st.markdown("### What-If Scenario Analysis")
        scenarios = [
            "What if volume increases by 50%?",
            "What if volume decreases by 50%?",
            "What if volatility increases?",
            "What if market crashes?",
            "What if market rally occurs?"
        ]

        scenario = st.selectbox("Select a scenario:", scenarios)

        if st.button("Simulate Scenario", key="whatif_btn"):
            interactive_mod = InteractiveModule()
            current_prediction = st.session_state.predictions[-1]
            modified_pred, explanation = interactive_mod.simulate_what_if(
                scenario,
                current_prediction,
                st.session_state.indicators
            )

            st.success(explanation)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Original Forecast",
                    f"${current_prediction:.2f}"
                )
            with col2:
                change = ((modified_pred - current_prediction) / current_prediction) * 100
                st.metric(
                    "Modified Forecast",
                    f"${modified_pred:.2f}",
                    f"{change:+.2f}%"
                )

    with tab3:
        st.markdown("### Custom Analysis")
        custom_query = st.text_input("Ask any question about the forecast:")

        if st.button("Analyze", key="custom_btn") and custom_query:
            # Process custom query
            interactive_mod = InteractiveModule()
            if 'why' in custom_query.lower():
                answer = interactive_mod.handle_why_question(
                    custom_query,
                    {
                        'current_price': st.session_state.stock_data['Close'].iloc[-1],
                        'predicted_price': st.session_state.predictions[-1]
                    },
                    st.session_state.indicators
                )
            elif 'what if' in custom_query.lower():
                current_prediction = st.session_state.predictions[-1]
                modified_pred, answer = interactive_mod.simulate_what_if(
                    custom_query,
                    current_prediction,
                    st.session_state.indicators
                )
            else:
                answer = "Please ask a 'Why' or 'What-if' question about the forecast."

            st.info(answer)

# Run the application
if __name__ == "__main__":
    main()

    # Add interactive mode interface if in that state
    if st.session_state.conversation_state == 'INTERACTIVE' and st.session_state.stock_data is not None:
        st.divider()
        handle_interactive_mode()