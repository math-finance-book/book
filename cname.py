import os 

os.system("copy under_construction.jpg docs")

with open("docs/CNAME", "w") as f:
    f.write("https://book.derivative-securities.org")