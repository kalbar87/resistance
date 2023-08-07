# Fishing Vessel Resistance Prediction

## Table of Contents
- [Overwiev](#overview)
- [Database](#Database)
- [Application Screenshot](#application-screenshot)
- [Tools and Software Requirements](#tools-and-software-requirements)
- [Packages Requirements](#packages-requirements)
- [Instalation](#instalationn)
- [Features and Target](#features-and-target)
- [Models and Algorithms](#model-and-algorithms)
- [Results](#results)
- [Correleation Matrix](#correlation-matrix)
- [Resisual Plot](#resisual-plot)
- [List of Symbols](#list-of-symbols)
- [Reeferences](#references)

## Overview
Machine learning (ML) and artificial intelligence (AI) becomes very popular approach in scientific data analyses recently. This analysis tries to apply this techniques on database of ship resistance evaluated on model tests conducted in the towing tank. The analyses were carried out on the fishing vessel types to maximize accuracy of tested models. The residuary resistance coefficients is predicted basing on presented methods and further evaluated by standard ITTC methodology to achieve main goal which is the total resistance of ship. Accomplished results are evaluated basing on standard metrics such as Pearson correlation coefficient (coefficient of determination 𝑅𝑅 2) and mean square error (MSE). The results of presented study seems to be very practical on early stage design and may be used for primary estimation of engine power based on resistance results.

Those analysis where presented as a paper on 23RD International Symposium on Hydrodynamics in ship design, manoeuvring and operation. 
https://www.prs.pl/uploads/hydronav2023_e_book_web.pdf

## Database
The ML model was built using a database comprises 28 ships that were tested under at least one loading condition. The features (independent variables) were determined through a correlation analysis, and the best ones were selected as inputs for the ML model. The density plots for each of selected features are presented in figure below:


Here is a brief description of the input data:

<b>Length of Waterline LWL [m]</b>: Measurement used to describe the size and design of a ship or boat. It is the length of the hull that is in direct contact with the water when the vessel is afloat and sits at its designed waterline. In other words, it is the length of the portion of the hull that is submerged when the ship is at its operational draft.

<b>Breadth Moulded B [m]</b>: It refers to the maximum width of the ship's hull at a particular cross-section, typically measured from the outermost points on the ship's hull. The breadth moulded measurement is taken at a specific location, usually amidships, which is the midpoint of the ship's length.

<b>Displacement Volume &#8711; [m&#179;]</b>: It represents the weight of water displaced by a ship when it is floating at a specific draft or immersion in the water. In other words, it is the mass of water that is "pushed aside" by the hull of the ship to make room for the vessel to float.

<b>Area of Wetted Surface S [m&#179;]</b>: It refers to the total surface area of the ship's hull that is in direct contact with the water when the vessel is afloat.

<b>Draught TA/TF [m]</b>: It represents the vertical distance from the waterline to the deepest point of the ship's hull, both at the fore perpendicular (TF) and at the aft perpendicular (TA). In other words, it is the portion of the ship's hull that is submerged below the water surface when the vessel is afloat.

<b>Longitudinal Centre of Buoyancy LCB [m]</b>:  It represents the longitudinal position, or location, of the center of buoyancy along the length of the ship's hull. The center of buoyancy refers to the point where the resultant buoyant force, or the upward force exerted by the water on the submerged part of the hull, acts.

<b> Transverse Projected Area of Ship above Waterline AT [m&#179;]</b>: The area above waterline is a two-dimensional measurement and is typically expressed in square meters or square feet. It is calculated by considering the entire cross-sectional area of the ship's hull that is exposed to the air above the waterline. This includes the sides (port and starboard) and any superstructures, deckhouses, or other structures that protrude above the waterline.

<b>Longitudinal Prismatic Coefficient CP</b>: The ratio of the volume of displacement to the volume of a prism having a length equal to the length between perpendiculars and a cross – sectional area equal to the midship sectional area. Mathematically, it can be expressed as: CP = &#8711; / (AW * T)


![Screenshot](/Images/CP.png)

<b>Midship Seaction Coefficient CM</b>: The ratio the midship section area to the area of a rectangle whose sides are equal to the draught and breadth extreme amidships. Mathematically, it can be expressed as: CM = AM / (B * T)

![Screenshot](/Images/CM.png)

<b>Speed Range V [knots]</b>: Minimal, maximal speed for which resistance prediction will be made 

<b>Form Factor k [-]</b>: It is one of the factors used in the calculation of the ship's skin friction resistance, which is the resistance due to the frictional drag experienced by the wetted surface of the hull as it moves through the water. A lower form factor value indicates a more streamlined hull shape, which results in reduced skin friction resistance and, therefore, better fuel efficiency and higher speeds. When the form factor is unknown it could be calculated based on ITTC formula k = 0.017 + 20 * CB / (LWL * B)^2 * sqrt(B / T)

<b>Design Speed [knots]</b>: The design speed is determined based on the ship's intended purpose, operational requirements, and the type of vessel being designed. 

## Application Screenshot
![Screenshot](/Images/app.png)


## Tools and Software Requirements
- [Github Account](https://github.com/)
- [Github CLI](https://cli.github.com/)
- [VS Code IDE](https://code.visualstudio.com/)
- [Jupyter](https://jupyter.org/)
- [Python3.8](https://www.python.org/downloads/release/python-380/)

## Packages Requirements
- Numpy
- Pandas
- Matplotlib
- Joblib
- Keras
- Scipy
- Tensorflow

## Instalation
1. Clone repository git clone https://github.com/kalbar87/resistance.git
2. Navigate to the project directory: cd resistance
3. Install dependencies: pip install -r requirements.txt
4. Run the project: streamlit run app.py

## Features and Target
The machine learning and deep neural network model have 8 input features (k, CB, LCB, CM, CP, B/T, Fr, Trim/T), which are calculated from hydrostatic input data. Residual resistance (CR) represents mostly the wave resistance and is the model target. The histogram plot for each feature adn target is shown below:

![Screenshot](/Images/features_hist.png) 


## Models and Algorithms
<b>Gradient boosting</b> is a powerful machine learning method used for regression and classification tasks, among other applications. It creates a predictive model by combining multiple weak prediction models, often represented as decision trees. When using decision trees as weak learners, the resulting algorithm is called gradient-boosted trees, which has shown to outperform the popular random forest algorithm.

Algorithm for transforming features and target is shown in below figure

![Screenshot](/Images/gbr_diagram.png) 

<b>Deep Neural Network (DNN)</b> is a type of artificial neural network that is composed of multiple layers of interconnected nodes (neurons) arranged in a hierarchical fashion. These networks are designed to model complex relationships and patterns in data by leveraging multiple levels of abstraction.

Graph representing neural network used in this project is shown below:

![Screenshot](/Images/dnn_plot.png) ![Screenshot](/Images/dnn_model2.png) 

## Results
During the preliminary analysis, five machine learning models were scored: Linear SVR, Lasso, Ridge, Extreme Gradient Boosting, Random Forest, and Gradient Boosting. Metrics such as Root Mean Square Error (RMSE), R2-value, etc., are shown in the table below:

Metrics Report

|                         |     RMSE|      MAE|       R2|   p-value|
|------------------------:|--------:|--------:|--------:|---------:|
|               Linear SVR|     0.58|     0.34|     0.72|   8.9e-25|
|                    Lasso|     0.38|     0.34|     0.73|   5.3e-25|
|                    Ridge|     0.51|     0.34|     0.73|   5.2e-25|
| Extreme Gradine Boosting|     0.18|     0.16|     0.91|   1.2e-41|
|            Random Forest|     0.24|     0.23|     0.85|   6.7e-33|
|        Gradient Boosting|     0.14|     0.13|     0.95|   4.5e-50|
|                         |         |         |         |          |
|      Deep Neural Network|         |         |     0.92|          |
|                         |         |         |         |          |

Gradient Boosting Regressor was selected as the machine learning model for ship resistance predictions.

## Correlation Matrix
![Screenshot](/Images/heatmap.png) 

## Residual Plot
![Screenshot](/Images/residuals.png) 

## List of Symbols
AT - Transverse Projected Area of Ship Above Waterline [m2]

B - Breadth moulded [m]

CB - Block Coefficient [-]

CF - Frictional resistance coefficient of a corresponding plate [-]

CM - Midship Section Coefficient [-]

CP = Longitudinal Prismatic Coefficient [-]

CR - Residuary resistance coefficient [-]

CT - Total resistance coefficient [-]

Fr - Froude Number [-]

k - Hull Form Factor [-]

LCB - Longitudinal Centre of Buoyancy [m]

LWL - Length of waterline [m]

PE - Effective Power [kW]

Re - Raynolds Number [-]

RT - Total Resistance [N]

S - Area of wetted surface [m2]

T - Draught even keel or mean draught at midship section [m]

TA - Draught aft (at aft perpendicular) [m]

TF - Draught fore (at fore perpendicular) [m]

V - Vessel speed [knots]

## References
1. M. G. Holtrop J., “An approximate power prediction method,” International Shipbuilding Progress, vol. 29, pp. 166–170, 1978.
2. L. Nikolopoulos and E. Boulougouris, “A Study on the Statistical Calibration of the Holtrop and Mennen Approximate Power Prediction Method for Full Hull Form, Low Froude Number Vessels,” Journal of Ship Production and Design, vol. 35, no. 1, pp. 41–68, Feb. 2019.
3. H. Khan, “Artificial neural networks based prediction and validation tool (annpvt) for ship model powering tests in towing tanks,” in 2021 international conference on artificial intelligence (icai), 2021, pp. 113–118.
4. K. Grabowska and P. Szczuko, “Ship resistance prediction with artificial neural networks,” in 2015 signal processing: Algorithms, architectures, arrangements, and applications (spa), 2015, pp. 168–173.
5. Q. Sun, M. Zhang, L. Zhou, K. Garme, and M. Burman, “A machine learning-based method for prediction of ship performance in ice: Part i. Ice resistance,” Marine Structures, 
vol. 83, p.103181, 2022.
6. A.-S. Milaković, F. Li, M. Marouf, and S. Ehlers, “A machine learning-based method for simulation of ship speed profile in a complex ice field,” Ships and Offshore Structures, vol. 15, no. 9, pp. 974–980, 2020.
7. Y. Ao, Y. Li, J. Gong, and S. Li, “Artificial Intelligence Design for Ship Structures: A Variant Multiple-Input Neural Network-Based Ship Resistance Prediction,” Journal of Mechanical Design, vol. 144, no. 9, Jun. 2022.
8. W. Tarelko and K. Rudzki, “Applying artificial neural networks for modelling ship speed and fuel consumption,” Neural Computing and Applications, vol. 32, no. 23, pp. 17379–17395, Dec. 2020.
9. Y. Yang, H. Tu, L. Song, L. Chen, D. Xie, and J. Sun, “Research on accurate prediction of the container ship resistance by rbfnn and other machine learning algorithms,” Journal of
Marine Science and Engineering, vol. 9, no. 4, 2021.
10. T. Uyanık, Ç. Karatuğ, and Y. Arslanoğlu, “Machine learning approach to ship fuel consumption: A case of container vessel,” Transportation Research Part D: Transport and Environment, vol. 84, p. 102389, 2020.
11. A. J. Smola and B. Schölkopf, “A tutorial on support vector regression,” Statistics and Computing, vol. 14, no. 3, pp. 199–222, Aug. 2004.
12. R. Tibshirani, T. Hastie, and J. Friedman, “Regularized paths for generalized linear models via coordinate descent,” Journal of Statistical Software, vol. 33, Feb. 2010.
13. A. E. Hoerl and R. W. Kennard, “Ridge regression: Biased estimation for nonorthogonal problems,”Technometrics, vol. 12, no. 1, pp. 55–67, 1970.
14. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proceedings of the 22Nd acm sigkdd international conference on knowledge discovery and data mining, in KDD
’16. New York, NY, USA: ACM, 2016, pp. 785–794. 
15. J. H. Friedman, “Greedy function approximation: A gradient boosting machine.” The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.
16. L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
17. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” CoRR, vol. abs/1412.6980, 2014.
18. ITTC quality system manual: Recommended procedures and guidelines: Procedure 1978 ittc performance prediction method. International Towing Tank Conference, 1978.
19. B. Bowden and N. Davison, “Resistance increments due to hull roughness associated with form factor extrapolation methods,” NPL Ship TM 3800, 1974.
20. F. Pedregosa et al., “Scikit-learn: Machine learning in Python.” https://scikit-learn.org, 2011.