import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException

api_key = 'KHdEb5pUm8NN'
api_secret = 'cRr0mnmt2P6l'
client = Client(api_key, api_secret)

account_info = client.futures_account()

# The account_info['assets'] field contains a list of dictionaries
# Each dictionary contains the asset name and its details
for asset in account_info['assets']:
    asset_name = asset['asset']
    wallet_balance = asset['walletBalance']
    print(f"Asset Name: {asset_name}, Wallet Balance: {wallet_balance}")
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
        client.futures_change_leverage(symbol=symbol, leverage=10)
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

        client.futures_change_leverage(symbol=symbol, leverage=10)
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

    # Check stop loss based on ROE
    roe = calculate_roe(symbol)
    if roe is not None and roe < -2:
        print("Stop loss reached. Closing the trade.")
        # Close 100% of the trade
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
                roe = (unrealized_profit / ((entry_price * position_amt) / leverage)) * 100
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
    left = 356
    top = 955
    width = 482
    height = 19

    screen = capture_screen(bbox=(left, top, left + width, top + height))
    text = pytesseract.image_to_string(screen)
    text = text.replace('on BICI', '')
    text = text.replace('on BICUSD', '')
    text = text.replace('on BIC', '')
    text = text.replace('on BTC', '')
    text = text.replace('US', '')
    text = text.replace(' ', '')
    text = text.replace('onB', '')

    print(text)
    update_open_trades()
    for symbol in list(open_trades.keys()):  # Create a copy of keys with list()
        roe = calculate_roe(symbol)
        if roe is not None:
            print(f"Current ROE for {symbol}: {roe}%")  # Print the current ROE
            if roe < -2:  # I changed this to -5 assuming you want to stop loss when ROE is less than -5%
                print(f"Stop loss reached for {symbol}. ROE is {roe}%. Closing the trade.")
                # Close 100% of the trade
                take_profit(symbol, 1, "short")
                take_profit(symbol, 1, "long")
    if previous_text is not None and text.strip() != previous_text:
        print("Change detected")
        if text.strip() == "Alert:ShortSignal" or text.strip() == "Alert:StrongShortSignal":
            sell('BTCBUSD', 1)  # Sell at market price
            print("Short signal detected. Selling BTCUSDT")

        elif text.strip() == "Alert:LongSignal" or text.strip() == "Alert:StrongLongSignal":
            buy('BTCBUSD', 1)  # Buy at market price
            print("Long signal detected. Buying BTCUSDT")

        elif text.strip() == "Alert:ShortTakeProfit1":
            print("Take profit 1 reached for short")
            # Close 75% of the trade
            take_profit('BTCBUSD', 1, "short")

        elif text.strip() == "Alert:LongTakeProfit1":
            print("Take profit 1 reached for long")
            # Close 75% of the trade
            take_profit('BTCBUSD', 1, "long")

        elif text.strip() == "Alet2":
            print("Take profit 2 reached for short")
            # Close 100% of the trade
            take_profit('BTCBUSD', 1, "short")

        elif text.strip() == "Aleit2":
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







    --------------------------------------------------------------------------
    import cv2
    import numpy as np
    import pyautogui
    import time
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    import threading

    region = (356, 955, 482, 19)
    last_seen_text = None


    def capture_text():
        global last_seen_text


    api_key = 'KHdEb86QlBsqhDX5OxLl0aP2Pdf8WzMAArN6a6s5VhEGcuGWtkLauYaVX5pUm8NN'
    api_secret = 'cRFwYVgCi3ozoqZJgrOVql3wtBHzYooJJE4A9zWOTwQxwQdHRKVr0pgHmnmt2P6l'
    client = Client(api_key, api_secret)

    account_info = client.futures_account()

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
            client.futures_change_leverage(symbol=symbol, leverage=10)
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

            client.futures_change_leverage(symbol=symbol, leverage=10)
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

        # Check stop loss based on ROE
        roe = calculate_roe(symbol)
        if roe is not None and roe < -2:
            print("Stop loss reached. Closing the trade.")
            # Close 100% of the trade
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
                    roe = (unrealized_profit / ((entry_price * position_amt) / leverage)) * 100
                    return roe

        except BinanceAPIException as e:

            print("Binance API Exception:", e)

        return None


    def detect_signals():
        # Define the ROI coordinates (left, top, width, height)
        roi = (810, 90, 100, 800)

        # Define the reference signals
        # These should be numpy arrays representing the images to track
        ref_signal_long = cv2.imread('1.png')
        ref_signal_short = cv2.imread('signal-short.png')

        # Initialize lists to store detected signals
        detected_signals_long = []
        detected_signals_short = []

        while True:
            try:
                # Capture a screenshot within the ROI
                screenshot = pyautogui.screenshot(region=roi)
                screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # Apply template Matching for long signals
                w_long, h_long = ref_signal_long.shape[0], ref_signal_long.shape[1]
                res_long = cv2.matchTemplate(screenshot, ref_signal_long, cv2.TM_CCOEFF_NORMED)
                threshold_long = 0.8
                loc_long = np.where(res_long >= threshold_long)

                for pt in zip(*loc_long[::-1]):
                    detected_signals_long.append(pt)
                    cv2.rectangle(screenshot, pt, (pt[0] + h_long, pt[1] + w_long), (0, 255, 0), 2)

                if len(detected_signals_long) > 0:
                    buy('BTCBUSD', 0.001)
                    print("Found a long signal")
                    # Perform the buy action here

                # Apply template Matching for short signals
                w_short, h_short = ref_signal_short.shape[0], ref_signal_short.shape[1]
                res_short = cv2.matchTemplate(screenshot, ref_signal_short, cv2.TM_CCOEFF_NORMED)
                threshold_short = 0.8
                loc_short = np.where(res_short >= threshold_short)

                for pt in zip(*loc_short[::-1]):
                    detected_signals_short.append(pt)
                    cv2.rectangle(screenshot, pt, (pt[0] + h_short, pt[1] + w_short), (0, 0, 255), 2)

                if len(detected_signals_short) > 0:
                    sell('BTCBUSD', 0.001)
                    print("Found a short signal")
                    # Perform the sell action here

                cv2.imshow("Detected Signals", screenshot)
                cv2.waitKey(1)

                time.sleep(0.5)

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break


    if __name__ == "__main__":
        thread_ds = threading.Thread(target=detect_signals)
        thread_ds.start()

        thread_ct = threading.Thread(target=capture_text)
        thread_ct.start()
