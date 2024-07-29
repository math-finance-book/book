import os

for i in range(14, -1, -1):
    os.system(
        f"move Chapter{i}.qmd Chapter{i+1}.qmd"
    )
