# EvolutionaryNAS
This repository collects recent Nature-inspired algorithms(Evolutionary and swarm)--based Neural Architecture Search (NAS) optimization and provides a summary (Paper and Code) by year and task. 
# <h1 id='Content'>Content</h1>
NPENAS: Neural Predictor Guided Evolution for Neural Architecture Search, [Paper](https://arxiv.org/abs/2003.12857), [Code](https://github.com/auroua/NPENASv1), Date 2020. 

In this paper, to enhance the exploration ability of EA algorithm, NPENAS to neural predictors are defined. The first predictor a graph-based uncertainty estimation network as a surrogate model. The second predictor is a graph-based neural network that directly outputs the performance prediction of the input neural architecture. An Evolutionary algorithm is applied. 
These predictors are used to rank the candidate architectures. The top performance architectures are selected as offspring and evaluated by training and validation. The procedure repeats a given number of times, and the neural predictor is trained from scratch with all the architectures in the pool at each iteration.

G-EA: Efficient Guided Evolution for Neural Architecture Search,  [Paper](https://arxiv.org/abs/2110.15232), [Code](https://github.com/VascoLopes/GEA), Date 2022. 
Similar to the idea presented by NPENAS, G-EA uses a zero-proxy estimator, as a guiding mechanism to the search method,  to evaluate several architectures in each generation at the initialization stage. To reduce the search space only the Top P scoring networks are trained and kept for the next generation. An Evolutionary algorithm is applied. 

