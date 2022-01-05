import warnings
from typing import Dict, Tuple

import lightgbm as lgb
import numpy
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.model_selection import GridSearchCV, train_test_split

from dasher_fms.objects import HTEModelConfig
from dasher_fms.utils.feature_generation import parse_time, save_float_mem
from dasher_fms.utils.logging_utils import logger

warnings.filterwarnings("ignore")


class HTEModel:
    def __init__(self, config: HTEModelConfig):
        """Initialize class for training hte models.

        Args:
            config (ModelConfig): A custom model configuration. Please see the model configuration definition case
            class.
        """
        self.primary_timestamp = config.primary_timestamp
        self.primary_id = config.primary_id
        self.label = config.label
        self.treatment = config.treatment
        self.schema = config.data_schema
        self.weights = config.weights

        self.dtype_cols = self._set_dtypes()

        self.model = None
        self.control_vars = None
        self.control_dtypes = None

        self.model_y_best_params = None
        self.model_t_best_params = None

        self.model_params = config.param_grid

    def _set_dtypes(self) -> Dict:
        """Identify numerical and string columns.

        Returns:
            Dict: A dictionary of column names and clumn types.
        """
        dtype_cols = {}
        dtype_cols["cat"] = [col for col, types in self.schema.items() if types == "str"]
        dtype_cols["num"] = [col for col, types in self.schema.items() if types == "num"]
        return dtype_cols

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing steps on input dataframe.

        1. Assigns the right data type
        2. Outlier handling on target variable

        Args:
            df: Input data frame for prediction
        Returns: Processed data frame
        """
        for col in self.dtype_cols["cat"]:
            df[col] = df[col].astype("str")
            df[col] = df[col].astype("category")

        for col in self.dtype_cols["num"]:
            df[col] = df[col].astype("float")

        df.loc[df[self.label] > df[self.label].quantile(0.99), self.label] = df[self.label].quantile(0.99)
        df.loc[df[self.label] < df[self.label].quantile(0.01), self.label] = df[self.label].quantile(0.01)

        return df

    def feature_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features for input dataframe.

        Args:
            df: Input dataframe for prediction

        Returns: Input dataframe with new features added

        """
        df = df[list(self.schema.keys())]

        time_fea = parse_time(df[self.primary_timestamp])

        timestamp = df[self.primary_timestamp].values
        df = df.drop(self.primary_timestamp, axis=1)
        df = pd.concat([df, time_fea], axis=1)
        df = df.reset_index(drop=True)

        y = df[self.label]
        t = df[self.treatment]

        df = df.drop([self.label, self.treatment], axis=1)

        if self.weights is not None:
            wt = df[self.weights]
            df = df.drop([self.weights], axis=1)

        self.control_vars = [col for col in df.columns if col not in self.primary_id]
        self.control_dtypes = df[self.control_vars].dtypes

        df = save_float_mem(df)
        # add timestamp column back
        df["timestamp"] = timestamp
        df[self.label] = y
        df[self.treatment] = t

        if self.weights is not None:
            df[self.weights] = wt

        logger.info(f"Data shape after feature generation: {df.shape}")

        return df

    def determine_best_params(self, df: pd.DataFrame) -> float:
        """Determine best parameters for first stage models.

        Args:
            df: Input data frame that will serve as training data

        Returns: MSE score on validation data by best performing model

        """
        train, test = train_test_split(df, train_size=0.8, random_state=123)
        train = train.reset_index()
        test = test.reset_index()

        if self.model_params is None:
            self.model_params = {"max_depth": [3, 5, 10], "n_estimators": (50, 100, 200, 500)}

        first_stage_y = GridSearchCV(
            estimator=lgb.LGBMRegressor(),
            param_grid=self.model_params,
            cv=3,
            n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )

        first_stage_t = GridSearchCV(
            estimator=lgb.LGBMClassifier(),
            param_grid=self.model_params,
            cv=3,
            n_jobs=-1,
            scoring="neg_log_loss",
        )

        model_y = first_stage_y.fit(train[self.control_vars + self.primary_id], train[self.label].values)
        model_t = first_stage_t.fit(train[self.control_vars + self.primary_id], train[self.treatment].values)

        self.model_y_best_params = model_y.best_params_
        self.model_t_best_params = model_t.best_params_

        dml = CausalForestDML(
            model_y=lgb.LGBMRegressor(**model_y.best_params_),
            model_t=lgb.LGBMClassifier(**model_t.best_params_),
            n_estimators=100,
            discrete_treatment=True,
            random_state=123,
        )
        dml.fit(
            Y=train[self.label],
            T=train[self.treatment],
            X=train[self.primary_id],
            W=train[self.control_vars],
            sample_weight=train[self.weights],
        )

        model_score = dml.score(
            Y=test[self.label], T=test[self.treatment], X=test[self.primary_id], W=test[self.control_vars]
        )
        logger.info(f"Model MSE: {model_score}")

        return model_score

    def train(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Train DML model for estimating HTE.

        Args:
            df: Input data frame for training HTE model

        Returns: Dataframe used for training, model mse score on validation data

        """
        df = self.preprocessing(df)
        df = self.feature_generation(df)
        logger.info("Determining optimal parameters for first stage models")
        model_score = self.determine_best_params(df)
        logger.info("Fitting on full data with optimal parameters")
        self.model = CausalForestDML(
            model_y=lgb.LGBMRegressor(**self.model_y_best_params),
            model_t=lgb.LGBMClassifier(**self.model_t_best_params),
            n_estimators=100,
            discrete_treatment=True,
            random_state=123,
        )
        self.model.fit(
            Y=df[self.label],
            T=df[self.treatment],
            X=df[self.primary_id],
            W=df[self.control_vars],
            sample_weight=df[self.weights],
        )
        return df, model_score

    def random_covariate_check(self, df: pd.DataFrame) -> Tuple[float, numpy.ndarray, numpy.ndarray]:
        """Compare deviation of HTE estimates with addition of random covariate.

        Args:
            df: Inout data frame for training HTE model with random covariate

        Returns:
            1. Mean squared deviation in estimates with and without a random covariate
            2. HTE estimate without random covariate
            3. HTE estimate with random covariate

        """
        df["random_var"] = np.random.normal(0, 1, len(df))

        dml_random = CausalForestDML(
            model_y=lgb.LGBMRegressor(**self.model_y_best_params),
            model_t=lgb.LGBMClassifier(**self.model_t_best_params),
            n_estimators=100,
            discrete_treatment=True,
            random_state=123,
        )
        dml_random.fit(
            Y=df[self.label],
            T=df[self.treatment],
            X=df[self.primary_id],
            W=df[self.control_vars + ["random_var"]],
            sample_weight=df[self.weights],
        )

        cate_original = self.model.const_marginal_effect(df[self.primary_id])
        cate_random = dml_random.const_marginal_effect(df[self.primary_id])

        wts = df[self.weights].values
        sq_error = (cate_original - cate_random) ** 2
        mean_sq_deviation = np.sum(sq_error * wts[:, None]) / (sq_error.shape[1] * np.sum(wts[:, None]))

        logger.info(f"Mean Squared Deviation after adding Random Covariate:{mean_sq_deviation}")

        return mean_sq_deviation, cate_original, cate_random

    def cate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return estimated HTE for all unique values of input variables.

        Args:
            df: Input data frame for estimating HTE

        Returns: Dataframe with estimated HTE values with original input variable

        """
        cate = self.model.const_marginal_effect(df[self.primary_id])
        cate_df = pd.concat([df[self.primary_id], pd.DataFrame(cate)], axis=1)
        return cate_df
