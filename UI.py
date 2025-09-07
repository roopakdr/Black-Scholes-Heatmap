import tkinter as tk
from tkinter import messagebox
import math
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call, put

def calculate_option_price():
    try:
        S = float(entry_S.get())
        K = float(entry_K.get())
        T = float(entry_T.get())
        r = float(entry_r.get())
        sigma = float(entry_sigma.get())

        call, put = black_scholes(S, K, T, r, sigma)

        result_var.set(f"Call Price: {call:.4f}\nPut Price:  {put:.4f}")
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")

# GUI setup
root = tk.Tk()
root.title("Black-Scholes Option Pricing")

tk.Label(root, text="Stock Price (S):").grid(row=0, column=0, sticky="e")
tk.Label(root, text="Strike Price (K):").grid(row=1, column=0, sticky="e")
tk.Label(root, text="Time to Maturity (T, in years):").grid(row=2, column=0, sticky="e")
tk.Label(root, text="Risk-Free Rate (r):").grid(row=3, column=0, sticky="e")
tk.Label(root, text="Volatility (σ):").grid(row=4, column=0, sticky="e")

entry_S = tk.Entry(root)
entry_K = tk.Entry(root)
entry_T = tk.Entry(root)
entry_r = tk.Entry(root)
entry_sigma = tk.Entry(root)

entry_S.grid(row=0, column=1)
entry_K.grid(row=1, column=1)
entry_T.grid(row=2, column=1)
entry_r.grid(row=3, column=1)
entry_sigma.grid(row=4, column=1)

tk.Button(root, text="Calculate", command=calculate_option_price).grid(row=5, column=0, columnspan=2, pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Helvetica", 12), fg="blue").grid(row=6, column=0, columnspan=2)

root.mainloop()
