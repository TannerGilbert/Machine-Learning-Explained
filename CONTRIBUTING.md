## Contributing

Thank you for wanting to contribute to Machine-Learning Explained. Machine-Learning Explained is an open-source repository containing explanations and implementations of machine learning algorithms and concepts, and as such any contributions that add to the current explanations or add new ones are more than welcome.

## Setup Machine-Learning-Explained and version control

1. Make a fork of this repository on Github. You will need an account with Github. This will allow you to make pull requests (PRs) later on.
2. Clone your fork.
   ```bash
   git clone <your-fork-url>
   cd Machine-Learning-Explained
   ```
3. Make `git` aware of the Machine-Learning-Explained repo.
   ```bash
   git remote add upstream https://github.com/TannerGilbert/Machine-Learning-Explained.git
   git fetch upstream
   ``` 

## Changing/Adding source code

1. Choose the branch for your changes.
   ```bash
   git checkout -b <new branch name>
   ```
2. Write some awesome code! (Make sure only to write code inside the `code` folders)

## Changing/Adding documentation

1. Choose the branch for your changes.
   ```bash
   git checkout -b <new branch name>
   ```
2. Make changes / add new documentation.
   > Note: Make sure to only work inside the `README.tex.md` files and not inside the `README.md` files.
3. Generate `README.md` file from `README.tex.md`

   1. Install [`readme2tex`](https://github.com/leegao/readme2tex)
      ```bash
      pip install readme2tex
      ```
   2. Convert `README.tex.md` to `README.md`
      ```bash
      python3 -m readme2tex --output README.md README.tex.md --svgdir tex --nocdn
      ```