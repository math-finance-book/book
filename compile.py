import os

chapters = [f"Chapter{i}.qmd" for i in range(1, 16) if i != 10]
for chapter in chapters:
    os.system(
        f"quarto render c:/Users/kerry/repos/book/{chapter} --to html"
    )
for chapter in chapters:
    os.system(
        f"quarto render c:/Users/kerry/repos/book/{chapter} --to html"
    )