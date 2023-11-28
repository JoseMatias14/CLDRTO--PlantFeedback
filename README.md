# State and Parameter Estimation in Closed-Loop Dynamic Real-time Optimization

Traditional real-time optimization (RTO) utilizes steady-state models, which limits its performance in highly dynamic plant operating environments that are becoming more prevalent with increased globlization and deregulation of energy markets.  This has led to dynamic RTO schemes in which plant dynamics are considered in the predicted plant response.  Inclusion of the impact of the plant control system on its  predicted response gives rise to closed-loop DRTO (CL-DRTO), which has been shown to yield superior performance than its open-loop prediction counterpart, particularly when the controller is detuned, for example to achieve acceptable performance in the face of dead time and inverse response. Plant information is typically incorporated into the CL-DRTO model using an additive noise paradigm by a bias updating strategy. The present work aims to investigate the benefits and challenges of including different model updating strategies in the CL-DRTO cycle, namely: estimating only the states, and combined parameter and state estimation via moving horizon estimation (MHE). We apply the strategies to two case studies, a distillation column and a continuous stirred tank reactor (CSTR). The results indicate that the combined state and parameter strategy outperforms the other two methods when the parametric uncertainty propagates nonlinearly in the system dynamics, but it can suffer significant losses if the parameters are not identifiable in some operating conditions. By contrast, the second case study exemplifies situations in which the ``bias update'' method performance is satisfactory in terms of economics and solution implementation. 

## Case Study 1

The first case study involves a two-product distillation column used for separating an ideal binary mixture.

The case study is simulated in Python 3 using CasADi, a tool for nonlinear optimization and algorithmic differentiation. For solving the dynamic optimization problem (both the MHE and CL-DRTO problems), we discretize the continuous model by applying three point Legendre collocation on finite elements, and solve the resulting nonlinear programming (NLP) problem using IPOPT. When integration is needed (for computing the bias update, the sensitivities, and for simulating the plant), we use the CVODES solver from the SUNDIALS suite. The system parameters are shown in the Supplementary Information. 

Additional result:
1. the control tuning results shown in the Supplementary Information Section can be found here:
   -- \CS1 - Distillation Column\A priori calculations\Control Tuning - 2FO series

## Case Study 2

The second case study considers an exothermic irreversible first-order reaction A to B and is modeled using mass and energy balances balances.

The simulations here are performed in MATLAB 2022 using the CasADi toolbox. For solving the dynamic optimization problem (both at MHE and  CL-DRTO level), we use a multiple shooting approach and the obtained NLP is solved with the IPOPT implementation embedded in the CasADi toolbox. For integrating the system and obtaining the optimization problem gradients, we use CVODES solver from the SUNDIALS suite, which is also embedded in CasADi. The system parameters are presented in the Supplementary Information.

Additional results:
1. the script for carrying out the parameter identifiability test (presented in the supplementary information section) can be found in
   -- CS2 - CSTR\Preliminary\Parameter Identifiability
2. the scripts that generate the results from Section 5.3.2.2 can be found in
   -- \CS2 - CSTR\Comparison\Monte Carlo
3. the results of the Discussion (Section 5.3.3) are obtained using the scripts
   -- Distillation column: \CS2 - CSTR\Comparison\Sensitivities\Distillation Column Sensitivities
   -- CSTR: \CS2 - CSTR\Comparison\Sensitivities\CSTR Sensitivies 

## Installation
CasADi requires manual installation (both in Python and Matlab); however, the solvers do not. 

## License
MIT License

Copyright (c) [2023] [Jose O.A. Matias and Christopher L.E. Swartz]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Contact
Jose Otavio Assumpcao Matias: assumpcj@mcmaster.ca // Christopher L.E. Swartz: swartzc@mcmaster.ca

