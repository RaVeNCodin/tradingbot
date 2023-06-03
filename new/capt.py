import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException

api_key = 'pfoHrMYxFOzZxIBdBH11dP1BJ7iwuaQfkb88fC5H2bNGFfYBVfjPm1GB8VjyEWIY'
api_secret = 'ksn75QmMjtMHQGU9OgWOFsJdfiURh5DxSDXzqRvVAisUvDMrqiLm0OTYKJsP1Sb4'
client = Client(api_key, api_secret)

# Initialize the open trades dictionary
open_trades = {}

# Fetch the exchange information
exchange_info = client.futures_exchange_info()

# Create a dictionary to save the quantity precision for each symbol
quantity_precision = {}

# Iterate over the symbols
for symbol_info in exchange_info["symbols"]:
    symbol = symbol_info["symbol"]
    # Save the quantity precision in the dictionary
    quantity_precision[symbol] = symbol_info["quantityPrecision"]


# Function to adjust the quantity precision
def adjust_quantity_precision(quantity, symbol):
    # Fetch the maximum allowed precision for the symbol
    precision = quantity_precision[symbol]
    # Adjust the quantity to the maximum allowed precision and return it
    return round(quantity, precision)


# Function to handle selling
def sell(symbol, quantity, is_take_profit=False):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=2)
        # Create a new order
        order = client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=adjust_quantity_precision(quantity, symbol))  # Precision adjustment

        # Set stop loss
        client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            quantity=adjust_quantity_precision(quantity, symbol))  # Precision adjustment

        # Update the open trades dictionary only if it's not a take profit order
        if not is_take_profit:
            if symbol in open_trades:
                open_trades[symbol]["remaining_quantity"] -= quantity
                if open_trades[symbol]["remaining_quantity"] <= 0:
                    del open_trades[symbol]
            else:
                print("No open position found.")

        return order
    except BinanceAPIException as e:
        print(e)
        return None


# Function to handle buying
def buy(symbol, quantity, is_take_profit=False):
    try:
        # Create a new order
        order = client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=adjust_quantity_precision(quantity, symbol))  # Precision adjustment

        client.futures_change_leverage(symbol=symbol, leverage=2)
        # Update the open trades dictionary only if it's not a take profit order
        if not is_take_profit:
            if symbol in open_trades:
                open_trades[symbol]["remaining_quantity"] += quantity
            else:
                open_trades[symbol] = {"remaining_quantity": quantity}

        client.futures_create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            quantity=adjust_quantity_precision(quantity, symbol))  # Precision adjustment

        return order
    except BinanceAPIException as e:
        print(e)
        return None


# Update the open trades dictionary
def update_open_trades():
    try:

        # Fetch the account information
        account_info = client.futures_account()

        # Clear the open_trades dictionary
        open_trades.clear()

        # Iterate over the positions and update the open_trades dictionary
        for position in account_info["positions"]:
            symbol = position["symbol"]
            quantity = float(position["positionAmt"])
            if quantity != 0:
                open_trades[symbol] = {"remaining_quantity": quantity}

    except BinanceAPIException as e:
        print(e)


# Close a trade with a market order
# Updated take_profit function
def take_profit(symbol, percentage, position_type):
    # Update the open trades dictionary
    update_open_trades()

    # If no open position, return None
    if symbol not in open_trades:
        print("No open position found.")
        return None

    # Calculate the quantity to close
    quantity = abs(open_trades[symbol]["remaining_quantity"]) * percentage

    # Double check the remaining quantity just before placing the market order
    update_open_trades()
    if symbol not in open_trades:
        print("No open position found.")
        return None
    real_remaining_quantity = abs(open_trades[symbol]["remaining_quantity"])

    # Adjust the quantity if necessary
    if real_remaining_quantity < quantity:
        print("Adjusting quantity to match the real remaining quantity.")
        quantity = real_remaining_quantity

    # Determine the side (buy/sell) based on the remaining quantity
    side = Client.SIDE_BUY if position_type == "short" else Client.SIDE_SELL

    # Send a market order to close the trade with the specified quantity
    order = client.futures_create_order(
        symbol=symbol,
        side=side,
        type=Client.ORDER_TYPE_MARKET,
        quantity=adjust_quantity_precision(quantity, symbol))  # Precision adjustment

    # Update the open trades dictionary
    open_trades[symbol]["remaining_quantity"] -= quantity
    if open_trades[symbol]["remaining_quantity"] <= 0:
        del open_trades[symbol]

    return order



# Function to calculate ROE from Binance
def calculate_roe(symbol):
    try:
        # Fetch the current position's information
        position_info = client.futures_position_information(symbol=symbol)

        if position_info:
            position = position_info[0]
            unrealized_profit = float(position["unRealizedProfit"])
            entry_price = float(position["entryPrice"])
            position_amt = float(position["positionAmt"])
            leverage = float(position["leverage"])

            if entry_price != 0 and leverage != 0:
                roe = unrealized_profit / (abs(entry_price * position_amt) / leverage) * 100
                # Adjust the sign of roe based on direction of the trade and profit/loss
                roe = roe if (position_amt > 0 and unrealized_profit >= 0) or (position_amt < 0 and unrealized_profit <= 0) else -roe
                return roe

    except BinanceAPIException as e:
        print(e)

    return None


# Your other code for screen capture, OCR, and trade execution based on text signal should follow here...
def capture_screen(bbox=None):
    cap_screen = np.array(ImageGrab.grab(bbox))
    image = process_img(cap_screen)

    return image


# Function to process the image
def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path accordingly

previous_text = None
while True:
    left = 342
    top = 970
    width = 400
    height = 20

    screen = capture_screen(bbox=(left, top, left + width, top + height))
    text = pytesseract.image_to_string(screen)
    text = text.replace('on BICI', '')
    text = text.replace('on BICUSD', '')
    text = text.replace('on BIC', '')
    text = text.replace('on BTC', '')
    text = text.replace('US', '')
    text = text.replace(' ', '')
    text = text.replace('NBD', '')
    text = text.replace('onB', '')
    text = text.replace('onMATD', '')
    text = text.replace('onETHD', '')

    print(text)
    update_open_trades()
    for symbol in list(open_trades.keys()):  # Create a copy of keys with list()
        roe = calculate_roe(symbol)
        if roe is not None:
            print(f"Current ROE for {symbol}: {roe}%")  # Print the current ROE
            if roe < -5:  # I changed this to -5 assuming you want to stop loss when ROE is less than -5%
                print(f"Stop loss reached for {symbol}. ROE is {roe}%. Closing the trade.")
                # Close 100% of the trade
                take_profit(symbol, 1, "short")
                take_profit(symbol, 1, "long")
    if previous_text is not None and text.strip() != previous_text:
        print("Change detected")

        if text.strip() == "Alert:ShortTakeProfit1":
            print("Take profit 1 reached for short")
            # Close 75% of the trade
            take_profit('BTCBUSD', 0.75, "short")

        elif text.strip() == "Alert:LongTakeProfit1":
            print("Take profit 1 reached for long")
            # Close 75% of the trade
            take_profit('BTCBUSD', 0.75, "long")

        elif text.strip() == "Alert:ShortTakeProfit2":
            print("Take profit 2 reached for short")
            # Close 100% of the trade
            take_profit('BTCBUSD', 1, "short")

        elif text.strip() == "Alert:LongTakeProfit2":
            print("Take profit 2 reached for long")
            # Close 100% of the trade
            take_profit('BTCBUSD', 1, "long")

        elif text.strip() == "Alert:LongStopLoss" or text.strip() == "Alert:LongExit":
            print("long stop loss")
            # Close 100% of the trade
            take_profit('BTCBUSD', 1, "long")

        elif text.strip() == "Alert:ShortStopLoss" or text.strip() == "Alert:ShortExit":
            print("short stop loss")
            # Close 100% of the trade
            take_profit('BTCBUSD', 1, "short")

        else:
            print("Unknown signal detected: " + text)

    previous_text = text.strip()
    time.sleep(1)  # Adjust the sleep time as needed