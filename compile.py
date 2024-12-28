import os
import sys 
num = 1 if len(sys.argv) == 1 else int(sys.argv[1])

names = [
    "Ito", 
    "Intro",
    "Arbitrage",
    "BlackScholes",
    "FX",
    "Merton",
    "Exotics",
    "Vol",
    "MC_Binomial",
    "FiniteDifferences",
    "Fourier",
    "FixedIncome",
    "FixedIncomeDerivatives",
    "Vasicek",
    "Survey"
]
chapters = [f"Chapter_{name}.qmd" for name in names]
for _ in range(num):
    for chapter in chapters:
        os.system(
            f"quarto render c:/Users/kerry/repos/book/{chapter} --to html"
        )