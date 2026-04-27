# Presentation Guide

## Final conclusion

Our project does not claim that historic status directly causes higher prices. It shows that architectural heritage carries predictive signal for Manhattan sale prices, but much of that signal is already absorbed by standard real-estate variables, especially tax assessment, building class, location-related controls, and property scale.

The strongest conclusion is the "vanishing heritage premium": heritage matters, but once the model knows assessment value and structural variables, the extra R2 improvement from explicit heritage features becomes modest. That is an insight, not a failure. It suggests the market and the city's assessment process have already priced part of the heritage value in.

## Eight-minute speaking plan

### Speaker 1: business case and data, about 2.5 minutes

Open Page 1.

Say that the app answers one question: can architectural and preservation features make Manhattan property price prediction more accurate?

Explain the business and society angle: buyers, real-estate analysts, urban planners, and preservation groups all care about whether heritage is measurable in market value.

Explain the data merge: NYC Rolling Sales gives sale price, PLUTO gives physical and tax assessment variables, and the Landmark Database gives architect, style, facade material, alteration history, and preservation status. The join key is BBL, the Borough-Block-Lot parcel identifier.

Important caveat: this is predictive, not causal. We are not proving landmark status causes higher prices.

### Speaker 2: visual insights, about 2.5 minutes

Open Page 2.

Use the 3D map first. Explain that each building is a property, color can represent price or price per square foot, and height can represent size, age, or price. This makes the geography of value visible.

Use the construction era and style/material charts to show that heritage is not one single variable. Age, style, material, and architect identity point to different kinds of market signal.

Use the architect leaderboard to explain heritage as brand value. Some architect portfolios sell above the Manhattan median price per square foot, which supports the idea that architecture can carry market reputation.

Use the correlation heatmap to set up the modeling result: assessment value is strongly related to sale price, so some heritage value may already be embedded in assessment.

### Speaker 3: models, explainability, and tuning, about 3 minutes

Open Page 3.

Explain that every model is trained twice: once with baseline real-estate variables and once with baseline plus heritage variables. The target is log sale price, because Manhattan prices have extreme outliers.

Linear regression is the required interpretable baseline. It is easy to explain but too simple for this dataset because price depends on nonlinear interactions between size, class, assessment, era, and location proxies.

Tree and boosting models do better because they capture thresholds and interactions. Random Forest is robust; Gradient Boosting, LightGBM, and CatBoost usually perform best because they build many small corrections.

Open Page 4.

Use feature importance and SHAP to show that the model is not a black box. Structural variables dominate, but heritage variables such as architect, building age, style, material, and alteration history still contribute signal.

Open Page 5.

Explain that W&B tracks the hyperparameter experiments and helps select the best-performing model configuration.

Final line: heritage improves prediction modestly, and the modest uplift is the story. Heritage value exists, but much of it is already priced into assessment and market structure.

## How to explain each chart

3D property map: where high-value or large properties cluster spatially. It is the opening visual.

Construction era map: how Manhattan's built history is distributed geographically.

Median price by era: why age is not a simple linear premium. Very old buildings can be scarce, while modern buildings can reflect luxury construction.

Style and material charts: architectural aesthetics can become market signals, but they should be interpreted as correlations rather than causal effects.

Architect leaderboard: the cleanest way to explain heritage as brand value. Some architects' portfolios sell above the city baseline.

Sankey diagram: shows how construction era flows into architectural style and then into price tiers.

Monthly sale chart: useful caveat. The dataset is one year of transactions, so it supports price-level modeling better than long-run appreciation claims.

Correlation heatmap: assessment value is strongly related to sale price, which helps explain the vanishing heritage premium.

Baseline vs heritage model chart: directly answers the research question. The heritage model should improve performance if heritage variables add signal.

Actual vs predicted: checks whether the model is calibrated overall. Points closer to the diagonal are better.

Residual map: identifies where the market paid more or less than the model expected. This turns errors into interpretable real-estate stories.

Feature importance and SHAP: explains which variables drive predictions and whether heritage variables matter after structural controls.

W&B tuning results: proves the model choice was tested systematically rather than chosen by intuition.

## Model-by-model interpretation

Linear Regression: simple and required by the course. It assumes each variable has a straight-line effect on log price. It underperforms because Manhattan property value depends on nonlinear thresholds and interactions.

Ridge Regression: similar to linear regression but shrinks coefficients to reduce overfitting. It can be more stable than plain linear regression, but it still cannot capture nonlinear relationships.

Lasso Regression: can shrink weak coefficients to zero, which helps feature selection. On this dataset, the limiting factor is not too many features; it is nonlinear structure, so Lasso does not solve the core problem.

Elastic Net: combines Ridge and Lasso. It is useful when many features are correlated, but it remains a linear model and therefore has a low ceiling here.

Decision Tree: easy to explain because it splits the data into rules. It can capture nonlinear effects, but a single tree is unstable and can overfit or miss smooth patterns.

Random Forest: averages many trees, making it more stable than one decision tree. It handles interactions well and is a strong general-purpose model, but it may smooth out rare luxury/trophy-property signals.

Gradient Boosting: builds trees sequentially, each correcting previous mistakes. It usually improves over Random Forest when tuned well, but it is more sensitive to learning rate, depth, and number of trees.

LightGBM: a fast gradient boosting implementation. It tends to perform very well on tabular data because it captures nonlinear interactions efficiently.

CatBoost: another boosting model that is especially strong with categorical-style signals. In this app, categories are label-encoded before modeling, so CatBoost still performs well, but it is not using its full native categorical advantage.

## Why the final result happens

Boosting models win because the data is tabular, nonlinear, and interaction-heavy.

Linear models lag because the relationship between price and features is not a single straight line.

Heritage uplift is modest because the baseline already includes powerful variables such as assessment value, building class, zoning, size, and FAR. These variables already contain some of the market's judgment about heritage.

Dollar errors remain large because Manhattan prices span from ordinary transactions to extreme trophy assets. The model predicts log price to reduce outlier dominance, but converting errors back into dollars makes misses at the high end look very large.

Trophy properties are hardest to predict because their prices depend on buyer psychology, prestige, view, scarcity, interior condition, and negotiation details that public datasets do not fully capture.
