# OpenGridGym

OpenGridGym is an Open-Source AI-friendly toolkit for distribution market simulation, referenced and described here: https://arxiv.org/abs/2203.04410. Please refer to the paper for details on authors and contact information.



As it is completely Python-based, OpenGridGym can be used on any operating system where Python 3 can be used, including Windows/Mac/Linux. It can also be run online using Google COLAB.



It is currently in its initial stages. Feedback is welcomed, and the developers plan to continue to upload more examples, similar to the ones described in the paper.



### Getting Started

1. Please clone this repository to a local directory.
2. Make sure you have already installed all Python packages listed in `requirements.txt`. The most notable is `dss-python` (see [here](https://github.com/dss-extensions/dss_python) for more), which is used to interface with OpenDSS from Python on any operating system.
3. In the `Examples` folder, you will find a quick tutorial in `demos.ipynb`. Make sure to run the notebook section by section first to ensure no installation errors are encountered.



### Example usage

Once you have created your own, or imported existing, grid, market and agent classes, you may run a simulation as follows:

1. **Instantiate the objects**

    ```python
    grid = CustomGrid(...)

    market = CustomMaraket(...)

    agents = [CustomAgent(...), ..., CustomAgent(...)]

    env = Environment(grid=grid, market=market, agent=agents)
    ```

2. **Run the environment**

   ```python
   env.reset()
   for t in env.iterate():
       ... # customize as needed (e.g. observe measurements)
   ```

