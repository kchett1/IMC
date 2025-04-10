import json
from abc import abstractmethod, ABC
from collections import deque
from typing import Any, TypeAlias


class Symbol(str):
    pass

class Listing:
    def __init__(self, symbol: Symbol, product: str, denomination: str):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class OrderDepth:
    def __init__(self):
        self.buy_orders = {}   # {price: volume}
        self.sell_orders = {}  # {price: volume}

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int,
                 buyer: str, seller: str, timestamp: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class Observation:
    def __init__(self):
        self.plainValueObservations = {}
        self.conversionObservations = {}

class ProsperityEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

class TradingState:
    def __init__(self):
        self.timestamp = 0
        self.traderData = ""
        self.listings = {}            # symbol -> Listing
        self.order_depths = {}        # symbol -> OrderDepth
        self.own_trades = {}          # symbol -> list[Trade]
        self.market_trades = {}       # symbol -> list[Trade]
        self.position = {}            # symbol -> int
        self.observations = Observation()


JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to avoid log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, od in order_depths.items():
            compressed[symbol] = [od.buy_orders, od.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([
                    t.symbol,
                    t.price,
                    t.quantity,
                    t.buyer,
                    t.seller,
                    t.timestamp,
                ])
        return compressed

    def compress_observations(self, obs: Observation) -> list[Any]:
        conversion_observations = {}
        for product, v in obs.conversionObservations.items():
            # If these are custom objects, you can add further fields here
            conversion_observations[product] = [
                v.bidPrice,
                v.askPrice,
                v.transportFees,
                v.exportTariff,
                v.importTariff,
                v.sugarPrice,
                v.sunlightIndex,
            ]
        return [obs.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

class Strategy(ABC):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        pass

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        pass

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Track if we keep hitting limit
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        # Liquidation conditions
        soft_liquidate = (
            len(self.window) == self.window_size
            and sum(self.window) >= self.window_size / 2
            and self.window[-1]
        )
        hard_liquidate = (
            len(self.window) == self.window_size
            and all(self.window)
        )

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # Take existing sell orders if cheap
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        # If we still can buy, place a buy order near the best known buy
        if to_buy > 0:
            if buy_orders:
                top_buy = max(buy_orders, key=lambda tup: tup[1])[0]  # price with highest volume
                price = min(max_buy_price, top_buy + 1)
            else:
                price = max_buy_price
            self.buy(price, to_buy)

        # Take existing buy orders if high enough
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        # If we still can sell, place a sell order near the best known sell
        if to_sell > 0:
            if sell_orders:
                top_sell = min(sell_orders, key=lambda tup: tup[1])[0]  # price with smallest volume
                price = max(min_sell_price, top_sell - 1)
            else:
                price = min_sell_price
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        if isinstance(data, list):
            self.window = deque(data)

class RAINFOREST_RESINStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class SQUID_INKStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

    def get_true_value(self, state: TradingState) -> int:
        kelp_depth = state.order_depths["KELP"]
        squid_depth = state.order_depths["SQUID_INK"]

        def mid_price(depth):
            if not depth.buy_orders or not depth.sell_orders:
                return None
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            return (best_bid + best_ask) / 2

        kelp_mid = mid_price(kelp_depth)
        squid_mid = mid_price(squid_depth)

        if kelp_mid is None or squid_mid is None:
            return squid_mid or 100  # fallback value

        # Make squid track kelp, but 1.5x the divergence
        adjusted_value = kelp_mid + (squid_mid - kelp_mid) * 1.5
        return round(adjusted_value)

class KELPStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        od = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = sorted(od.buy_orders.items(), reverse=True)
        sell_orders = sorted(od.sell_orders.items())
        if not buy_orders or not sell_orders:
            return 50
        best_buy = max(buy_orders, key=lambda tup: tup[1])[0]
        best_sell = min(sell_orders, key=lambda tup: tup[1])[0]
        return round((best_buy + best_sell) / 2)

class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
        }
        self.strategies = {
            symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
                "KELP": KELPStrategy,
                "RAINFOREST_RESIN": RAINFOREST_RESINStrategy,
                "SQUID_INK": SQUID_INKStrategy,
            }.items()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            # Load from previous
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])
            # If we have an order book, run strategy
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
    
