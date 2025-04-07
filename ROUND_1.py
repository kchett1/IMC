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
        # buy_orders and sell_orders each hold {price: volume}
        self.buy_orders = {}
        self.sell_orders = {}

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

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

        # We truncate state.traderData, trader_data, and self.logs to fit the log limit
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
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

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
        return value[: max_length - 3] + \"...\"

logger = Logger()

class Strategy(ABC):
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

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

    # Fixed: signature must include 'self'
    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        pass

    def act(self, state: TradingState) -> None:
        # Retrieve a 'fair price' for the asset
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Track how often we're at position limit
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        # Liquidation flags
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
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        # Take existing sell orders if they're cheap enough
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # Hard liquidate
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        # Soft liquidate
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        # Place a buy order near the top buy price if still want more
        if to_buy > 0:
            if buy_orders:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
            else:
                price = max_buy_price
            self.buy(price, to_buy)

        # Take existing buy orders if they're high enough
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        # Hard liquidate
        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        # Soft liquidate
        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        # Place a sell order near the top sell price if we still can
        if to_sell > 0:
            if sell_orders:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
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
        # Hard-coded fair value
        return 10_000

class SQUID_INKStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # Basic approach: use top buy and top sell as a fair price
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return 100  # fallback

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

class KELPStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return 50  # fallback

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return round((popular_buy_price + popular_sell_price) / 2)

class Trader:
    def __init__(self) -> None:
        limits = {
            \"KELP\": 50,
            \"RAINFOREST_RESIN\": 50,
            \"SQUID_INK\": 50,
        }
        # Map each symbol to its strategy class
        self.strategies = {
            symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
                \"KELP\": KELPStrategy,
                \"RAINFOREST_RESIN\": RAINFOREST_RESINStrategy,
                \"SQUID_INK\": SQUID_INKStrategy,
            }.items()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            # Reload saved strategy data
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            # Only run the strategy if we actually have an OrderDepth
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            # Save updated data
            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(\",\", \":\"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


