# Public-data

| Dataset | Exact loader/source | Response | Informed predictors |
|---|---|---|---|
| Diabetes | `sklearn.datasets.load_diabetes(as_frame=True)` | `target` | bmi(+), bp(+), s5(+), s3(-) |
| Engel | `statsmodels.datasets.engel.load_pandas().data` | foodexp | income(+) |
| Concrete | OpenML **data_id=44959** | strength | cement(+), age(+) |
| Abalone | OpenML **data_id=183** | Class_number_of_rings | length(+), diameter(+), height(+), shell_weight(+) |
| Red wine | OpenML **data_id=42184** | quality | volatile acidity(-), citric acid(+), sulphates(+), alcohol(+) |
| White wine | OpenML **data_id=40498** | Class | V2(-), V3(+), V10(+), V11(+) |
| Auto MPG | [Rdatasets/ISLR Auto.csv](https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Auto.csv) | mpg | displacement(-), horsepower(-), weight(-) |
| Windsor | [Rdatasets/Ecdat Housing.csv](https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Housing.csv) | price | lotsize(+) |

OpenML source records: https://www.openml.org/d/44959 ; https://www.openml.org/d/183 ; https://www.openml.org/d/42184 ; https://www.openml.org/d/40498 .
