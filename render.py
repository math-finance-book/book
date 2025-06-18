import os
os.system("quarto render")
with open("docs/CNAME", "w") as f:
    f.write("https://book.derivative-securities.org")
