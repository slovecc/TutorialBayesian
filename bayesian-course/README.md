# Adevinta Bayesian Course

# Tutorial
The link for the final tutorial can be found at this [link](to be included)


### About Jupyter Book

The official documentation page can be found [here](https://jupyterbook.org/intro.html)

A  recommended introduction can be found [here](https://ubc-dsci.github.io/jupyterdays/sessions/beuzen/jupyter_book_tutorial.html)

#### Needed packages:
pip install jupyter-book sphinx-thebe  ghp-import

#### Most used commands:
-  to add a new notebook, include it in the file "_toc.yml"

- to build your book, in the book’s root directory use the command >> jb build ./
- to publish online the book, from the book's root directory >> ghp-import -n -p -f ./_build/html




# Repo structure
The structure of this repository consists:

- datasets-toys : folder with the dataset that are used in the exercise 
    - datasets: raw data
    - description: description about the info into the datasets
    
- tutorial: folder with all the notebooks and file needed for the jupyter book tutorial
