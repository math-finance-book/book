import os
os.system("quarto render")
os.system("python cname.py")
os.system("git add .")
os.system("git commit -m 'update'")
os.system("git push")