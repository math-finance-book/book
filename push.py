import os, sys
msg = sys.argv[1]
os.system("git pull origin main")
os.system("quarto render")
os.system("copy under_construction.jpg docs")
with open("docs/CNAME", "w") as f:
    f.write("https://book.derivative-securities.org")
os.system("git add .")
os.system("git commit -m '" + msg + "'")
os.system("git push origin main")