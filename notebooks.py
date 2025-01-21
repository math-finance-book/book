import os 
import json
import subprocess

finished = ["Brownian", "Ito"]
finished = ["Chapter_" + f + ".qmd" for f in finished]

nocode = ["Arbitrage"]
nocode = ["Chapter_" + f + ".qmd" for f in nocode]

image_path = "https://www.dropbox.com/scl/fi/6hwvdff7ajaafmkpmnp0o/under_construction.jpg?rlkey=3dex2dx86anniqoutwyqashnu&dl=1"
construction_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        f'<img src="{image_path}" alt="Under Construction" width="400"/>\n'
    ]
}

with open("_quarto.yml", "r") as f:
    lines = [line for line in f.readlines()  if line.strip().startswith("- Chapter")]
chapters = [line.strip()[2:] for line in lines]

numbers = [f"0{n}" for n in range(1, 10)] + [str(n) for n in range(10, len(chapters) + 1)]
names = [c.split("_")[1].replace("qmd", "ipynb") for c in chapters]
notebooks_out = [number + "_" + name for number, name in zip(numbers, names)]

notebooks_in = [c.replace("qmd", "ipynb") for c in chapters]

for chapter, notebook_in, notebook_out in zip(chapters, notebooks_in, notebooks_out):
    if chapter not in nocode:
        subprocess.run("quarto convert " + chapter, shell=True, check=True)
        with open(notebook_in, 'r') as f:
            js = json.load(f)
        
        js['cells'] = [cell for cell in js['cells'] if cell['cell_type'] != 'markdown']
        
        for cell in js['cells']:
            cell['source'] = [line for line in cell['source'] if not line.strip().startswith('#|')]
        
        if chapter not in finished:
            js['cells'].insert(0, construction_cell)

        new_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
                "---\n",
                "\n",
                "Created for [Pricing and Hedging Derivative Securities: Theory and Methods](https://book.derivative-securities.org/)\n",
                "\n",
                "Authored by\n",
                "- Kerry Back, Rice University\n",
                "- Hong Liu, Washington University in St. Louis\n",
                "- Mark Loewenstein, University of Maryland\n",
                " \n",
                "---\n",
                "\n",
                f"<a target=\"_blank\" href=\"https://colab.research.google.com/github/math-finance-book/book-code/blob/main/{notebook_out}\">\n",
                "  <img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/>\n",
                "</a>"
        ]
        }
        js['cells'].insert(0, new_cell)
        with open("../book-code/" + notebook_out, 'w') as f:
            json.dump(js, f, indent=2)    
        subprocess.run("del " + notebook_in, shell=True, check=True)

subprocess.run("git -C ../book-code pull origin main", shell=True, check=True)
subprocess.run("git -C ../book-code add .", shell=True, check=True)
subprocess.run('git -C ../book-code commit -m "update notebooks"', shell=True, check=True)
subprocess.run("git -C ../book-code push origin main", shell=True, check=True)