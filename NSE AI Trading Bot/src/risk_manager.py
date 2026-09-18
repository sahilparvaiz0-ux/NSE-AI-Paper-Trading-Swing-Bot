"""Portfolio cash ledger and risk controls for the swing-trading bot."""

class RiskManager:
    def __init__(self, starting_capital: float = 1_000_000.0,
                 risk_per_trade_pct: float = 0.01,
                 max_concurrent_positions: int = 5,
                 max_single_position_pct: float = 0.25,
                 stop_loss_atr_mult: float = 2.0,
                 take_profit_atr_mult: float = 4.0,
                 brokerage_pct: float = 0.0003,
                 stt_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 annual_risk_free_rate: float = 0.065):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_single_position_pct = max_single_position_pct
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.brokerage_pct = brokerage_pct
        self.stt_pct = stt_pct
        self.slippage_pct = slippage_pct
        self.annual_risk_free_rate = annual_risk_free_rate
        self.daily_rf_rate = annual_risk_free_rate / 252
        self.interest_earned = 0.0

    def position_size(self, entry_price: float, atr_value: float, portfolio_equity: float):
        """Size a trade subject to risk, position-cap and *all-in* cash limits."""
        stop_distance = self.stop_loss_atr_mult * atr_value
        if stop_distance <= 0 or entry_price <= 0 or portfolio_equity <= 0:
            return 0, entry_price, entry_price

        risk_amount = portfolio_equity * self.risk_per_trade_pct
        qty_by_risk = risk_amount / stop_distance
        max_position_value = portfolio_equity * self.max_single_position_pct
        qty_by_cap = max_position_value / entry_price

        # Solve quantity * entry_price + buy_cost(quantity*entry_price) <= cash.
        buy_cost_rate = self.brokerage_pct + self.slippage_pct
        qty_by_cash = self.cash / (entry_price * (1 + buy_cost_rate))
        qty = int(min(qty_by_risk, qty_by_cap, qty_by_cash))

        stop_loss = entry_price - stop_distance
        take_profit = entry_price + self.take_profit_atr_mult * atr_value
        return max(qty, 0), stop_loss, take_profit

    def transaction_cost(self, trade_value: float, side: str) -> float:
        brokerage = trade_value * self.brokerage_pct
        slippage = trade_value * self.slippage_pct
        stt = trade_value * self.stt_pct if side.lower() == "sell" else 0.0
        return brokerage + stt + slippage

    def can_open_new_position(self, open_positions_count: int) -> bool:
        return open_positions_count < self.max_concurrent_positions

    def open_position_cash_outflow(self, qty: int, entry_price: float) -> float:
        notional = qty * entry_price
        cost = self.transaction_cost(notional, "buy")
        self.cash -= notional + cost
        if self.cash < -1e-8:
            raise RuntimeError(f"Cash ledger violation: {self.cash}")
        return cost

    def close_position_cash_inflow(self, qty: int, exit_price: float) -> float:
        notional = qty * exit_price
        cost = self.transaction_cost(notional, "sell")
        self.cash += notional - cost
        return cost

    def accrue_daily_interest(self):
        """Accrue interest only on uninvested cash."""
        interest = self.cash * self.daily_rf_rate
        self.cash += interest
        self.interest_earned += interest
        return interest
