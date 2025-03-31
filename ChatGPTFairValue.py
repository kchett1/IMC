import numpy as np
from collections import deque
from typing import Dict, Optional, List
import jsonpickle
from datamodels import OrderDepth, ConversionObservation, TradingState, Order

# -------------------------------
# Advanced Price Model Definition
# -------------------------------

class AdvancedPriceModel:
    def __init__(self, product: str, risk_limit: int, window_size: int = 20):
        self.product = product
        self.risk_limit = risk_limit
        self.window_size = window_size
        # Maintain a history of observed mid-prices for trend analysis.
        self.price_history = deque(maxlen=window_size)
        
        # Initial weights for each signal component.
        self.weights = {
            "mid": 0.3,
            "vwap": 0.3,
            "imbalance": 0.2,
            "conversion": 0.1,
            "trend": 0.1,
        }
        
        # Learning rate for online calibration.
        self.calibration_rate = 0.01

    def update_price_history(self, price: float):
        """Store the latest mid-price observation."""
        self.price_history.append(price)

    def calculate_order_book_metrics(self, order_depth: OrderDepth) -> Dict[str, float]:
        """
        Calculate key order book metrics:
         - mid_price: the average of best bid and ask.
         - vwap: volume-weighted average price across both sides.
         - imbalance: normalized difference between total buy and sell volumes.
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {"mid_price": None, "vwap": None, "imbalance": None}
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0
        
        total_value, total_quantity = 0.0, 0.0
        for price, quantity in order_depth.buy_orders.items():
            total_value += price * quantity
            total_quantity += quantity
        for price, quantity in order_depth.sell_orders.items():
            total_value += price * abs(quantity)
            total_quantity += abs(quantity)
        vwap = total_value / total_quantity if total_quantity > 0 else mid_price
        
        buy_volume = sum(order_depth.buy_orders.values())
        sell_volume = sum(abs(q) for q in order_depth.sell_orders.values())
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-6)
        
        return {"mid_price": mid_price, "vwap": vwap, "imbalance": imbalance}

    def calculate_trend_metrics(self) -> Dict[str, float]:
        """
        Compute trend signals from the price history:
         - SMA (Simple Moving Average)
         - Volatility (standard deviation)
        """
        if len(self.price_history) == 0:
            return {"sma": None, "volatility": 0.0}
        
        prices = np.array(self.price_history)
        sma = prices.mean()
        volatility = prices.std()
        return {"sma": sma, "volatility": volatility}

    def calculate_conversion_impact(self, conversion_obs: ConversionObservation) -> float:
        """
        Process conversion observation data.
        The coefficients here can be calibrated based on historical performance.
        """
        impact = (
            conversion_obs.bidPrice * 0.2 -
            conversion_obs.askPrice * 0.2 +
            conversion_obs.transportFees * 0.15 -
            (conversion_obs.exportTariff - conversion_obs.importTariff) * 0.15 +
            conversion_obs.sugarPrice * 0.1 +
            conversion_obs.sunlightIndex * 0.1
        )
        return impact

    def risk_adjustment(self, fair_value: float, current_position: int) -> float:
        """
        Adjust the computed fair value when nearing position limits.
        This discourages further exposure when risk is high.
        """
        if abs(current_position) > self.risk_limit * 0.8:
            adjustment_factor = 1 - 0.05 * (abs(current_position) / self.risk_limit)
            return fair_value * adjustment_factor
        return fair_value

    def calibrate_parameters(self, observed_price: float, predicted_price: float):
        """
        Update model weights based on prediction error.
        This is a simple online calibration mechanism that could be expanded
        with more advanced methods such as reinforcement learning.
        """
        error = observed_price - predicted_price
        # Adjust each weight slightly based on the error.
        for key in self.weights:
            self.weights[key] += self.calibration_rate * error * 0.01
        
        # Normalize weights so they sum to 1.
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total

    def calculate_fair_value(self, order_depth: OrderDepth, conversion_obs: ConversionObservation, current_position: int) -> Optional[float]:
        """
        Combine all components to compute an advanced fair value estimate.
        """
        metrics = self.calculate_order_book_metrics(order_depth)
        if metrics["mid_price"] is None:
            return None
        
        trend_metrics = self.calculate_trend_metrics()
        conversion_impact = self.calculate_conversion_impact(conversion_obs)
        
        fair_value = (
            self.weights["mid"] * metrics["mid_price"] +
            self.weights["vwap"] * metrics["vwap"] +
            self.weights["imbalance"] * (metrics["mid_price"] + metrics["imbalance"] * (metrics["vwap"] - metrics["mid_price"])) +
            self.weights["conversion"] * conversion_impact +
            self.weights["trend"] * (trend_metrics["sma"] if trend_metrics["sma"] is not None else metrics["mid_price"])
        )
        
        # Apply risk adjustments based on current position.
        fair_value = self.risk_adjustment(fair_value, current_position)
        return fair_value

# ------------------------------------------------------
# Global Structures for Persistent Model Management
# ------------------------------------------------------
# Define risk limits for each product.
risk_limits = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
}

# Create a persistent dictionary mapping product names to their models.
models: Dict[str, AdvancedPriceModel] = {}
for product, limit in risk_limits.items():
    models[product] = AdvancedPriceModel(product, risk_limit=limit, window_size=20)

# -------------------------------
# Trader Implementation
# -------------------------------

class Trader:
    def run(self, state: TradingState):
        """
        Main trading logic:
         - For each product, update the price history,
         - Calculate an advanced fair value,
         - Calibrate parameters based on prediction error,
         - Generate buy/sell orders that respect position limits.
        """
        print("traderData:", state.traderData)
        print("Observations:", state.observations)
        result = {}
        
        # Loop through each tradable product.
        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            # Retrieve conversion observation for this product.
            conversion_obs = state.observations.conversionObservations.get(product)
            if conversion_obs is None:
                # If missing, create a default conversion observation.
                conversion_obs = ConversionObservation(
                    bidPrice=0.0, askPrice=0.0, transportFees=0.0,
                    exportTariff=0.0, importTariff=0.0,
                    sugarPrice=0.0, sunlightIndex=0.0
                )
            
            # Get current position for this product.
            current_position = state.position.get(product, 0)
            
            # Retrieve the model for this product.
            model = models.get(product)
            if model is None:
                continue
            
            # Use order book metrics to obtain an observed mid-price.
            metrics = model.calculate_order_book_metrics(order_depth)
            if metrics["mid_price"] is None:
                continue
            observed_price = metrics["mid_price"]
            model.update_price_history(observed_price)
            
            # Calculate the advanced fair value.
            predicted_fair_value = model.calculate_fair_value(order_depth, conversion_obs, current_position)
            if predicted_fair_value is None:
                continue
            
            # Calibrate the model with the prediction error.
            model.calibrate_parameters(observed_price, predicted_fair_value)
            
            print(f"Product: {product} | Observed Price: {observed_price:.2f} | Predicted Fair Value: {predicted_fair_value:.2f} | Current Position: {current_position}")
            
            # Generate BUY orders: if best ask is below fair value.
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_qty = order_depth.sell_orders[best_ask]
                if best_ask < predicted_fair_value:
                    # Limit the trade so as not to breach position limits.
                    max_buy = risk_limits[product] - current_position
                    trade_qty = min(-best_ask_qty, max_buy)
                    if trade_qty > 0:
                        print(f"Placing BUY order for {trade_qty} units at {best_ask} for {product}")
                        orders.append(Order(product, best_ask, trade_qty))
            
            # Generate SELL orders: if best bid is above fair value.
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_qty = order_depth.buy_orders[best_bid]
                if best_bid > predicted_fair_value:
                    # Limit the trade so as not to breach position limits.
                    max_sell = risk_limits[product] + current_position  # current_position is negative for short positions
                    trade_qty = min(best_bid_qty, max_sell)
                    if trade_qty > 0:
                        print(f"Placing SELL order for {trade_qty} units at {best_bid} for {product}")
                        orders.append(Order(product, best_bid, -trade_qty))
            
            result[product] = orders
        
        # Persist model calibration info in traderData for debugging/learning in subsequent rounds.
        traderData = jsonpickle.encode({"models": {p: model.weights for p, model in models.items()}})
        conversions = 1  # Placeholder conversion request as per tutorial instructions.
        return result, conversions, traderData
