## Nesterov's momentum improves the convergence rate of FW on a class of constraints
This repo contains the code for TSP and AAAI paper
- [A Momentum-Guided Frank-Wolfe Algorithm](https://ieeexplore-ieee-org.ezp1.lib.umn.edu/document/9457128)
- [Enhancing Parameter-Free Frank Wolfe with an Extra Subproblem](https://ojs.aaai.org/index.php/AAAI/article/view/17012)


### Code organization

- [prob.py](https://github.com/BingcongLi/HFW/blob/main/prob.py) defines the loss functions and constraint sets.
- [optimizer.py](https://github.com/BingcongLi/HFW/blob/main/optimizer.py) implements proposed and benchmark algorithms.
- [main.ipynb](https://github.com/BingcongLi/HFW/blob/main/main.ipynb) gives an example of binary classification on dataset *mushroom*.

### Reference
If the code is helpful for your research, please cite the following papers
```
@inproceedings{li2021enhancing,
  title={Enhancing Parameter-Free Frank Wolfe with an Extra Subproblem},
  author={Li, Bingcong and Wang, Lingda and Giannakis, Georgios B and Zhao, Zhizhen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={9},
  pages={8324--8331},
  year={2021}
}

@article{li2021momentum,
  title={A Momentum-Guided Frank-Wolfe Algorithm},
  author={Li, Bingcong and Couti{\~n}o, Mario and Giannakis, Georgios B and Leus, Geert},
  journal={IEEE Transactions on Signal Processing},
  volume={69},
  pages={3597--3611},
  year={2021},
  publisher={IEEE}
}
```
