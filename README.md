The scripts necessary to reproduce all of the results obtained in the manuscript *Integrating infection intensity into within- and between-host pathogen dynamics: implications for invasion and virulence evolution* by Wilber, M., Pfab, F., Ohmer, M. and Briggs, C. in *The American Naturalist*.  The descriptions of the included scripts are given below.  Documentation is also given in the scripts themselves.  The Python environment from which all of the results were generated is given by the `environment.yml` file. The Python environment needed to replicate the analyses can be built from this file (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## Scripts

1. `full_model.py`: The implementation of the full host-parasite IPM used to compare to the reduced host-parasite IPM.

2. `reduced_model.py`: The implementation of the reduced dimension host-parasite IPM models.

3. `test_R0.py`: Script that compares the dynamics of the reduced IPMs and the full IPMs and tests that R0 values for the reduced and full IPM correspond to invasion thresholds in the model dynamics.

4. `manuscript_plots.*`: A Jupyter notebook that contains the code to replicate all of the figures presented in the main text. The html file is a rendering of the .ipynb Jupyter notebook for easy viewing.

5. `continous_time_IPM.*`: Mathematica notebook (*.nb) that describes and validates the derivation of the continuous-time IPM derived in Appendix A.  The pdf file is a rendering of the notebook.