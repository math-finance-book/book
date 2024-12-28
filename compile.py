import os
import sys 
num = int(sys.argv[1])

chapters = [f"Chapter{i}.qmd" for i in range(1, 16)] + ["ChapterFT.qmd"]
for _ in range(num):
    for chapter in chapters:
        os.system(
            f"quarto render c:/Users/kerry/repos/book/{chapter} --to html"
        )