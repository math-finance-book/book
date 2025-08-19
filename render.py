import os
os.system("quarto render")
with open("docs/CNAME", "w") as f:
    f.write("https://book.derivative-securities.org")

# Add git commands to commit and push
os.system("git add -A")
os.system("git commit -m 'AI reorg'")
os.system("git push")
