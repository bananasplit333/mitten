import numpy as np
import requests
from langchain.tools import tool
from typing import Literal
#define tools

@tool
def quadratic_calculator(a:float, b:float, c:float) -> str:
    """
    Solves the quadratic equation
    """
    if a == 0: return "DNE";

    #test discriminant 
    discriminant = b**2-4*a*c
    
    if discriminant > 0:
        ans1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        ans2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        return f"({str(ans1)}, {str(ans2)})"
    elif discriminant == 0: 
        ans1 = -b/(2*a)
        return str(ans1)
    else:
        ans1a = -b/(2*a)
        ans1b = np.sqrt(np.abs(b**2 - 4*a*c))/(2*a)
        return f"({str(ans1a)}-{str(ans1b)}i, {str(ans1a)}+{str(ans1b)}i)"


@tool
def fib_list(n: int) -> list[int]:
    """
    Generate the first n fibonacci numbers
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    fibs = [0, 1]
    while len(fibs) < n:
        next_fib = fibs[-1] + fibs[-2]
        fibs.append(next_fib)
    return fibs

CurrencyCode = Literal[
    "AUD", 
    "BGN", 
    "BRL", 
    "CAD", 
    "CHF",
    "CNY",
    "CZK",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "HRK",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "ISK",
    "JPY",
    "KRW",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PHP",
    "PLN",
    "RON",
    "RUB",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "USD",
    "ZAR"
]

@tool
def convert_currency(amount: float, fromCurrency: CurrencyCode, toCurrency: Literal["USD", "CAD", "EUR"]) -> float:
    """
    Convert amount from fromCurrency to either USD, EUR, or CAD.
    """
    base_url = "https://api.freecurrencyapi.com/v1/latest"
    params = {
        "apikey": "fca_live_ntSZWuM334ysr2yg3CZDYslXpjAL62xPNWXdUe53",
        "base_currency": fromCurrency,
        "currencies": toCurrency
    }

    response = requests.get(base_url, params=params).json()
    total = amount * response["data"][toCurrency]
    print(total)
    return total



