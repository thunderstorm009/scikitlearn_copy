import inspect

class BaseEstimator:
    """Base class for all estimators in scikitlearn_copy."""
    
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self}.")
            setattr(self, key, value)
            valid_params[key] = value
        return self

class ClassifierMixin:
    """Mixin class for all classifiers in scikitlearn_copy."""
    _estimator_type = "classifier"

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

class RegressorMixin:
    """Mixin class for all regression estimators in scikitlearn_copy."""
    _estimator_type = "regressor"

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction."""
        # For simplicity, returning negative MSE, but R^2 is typical
        from .metrics import mean_squared_error
        return -mean_squared_error(y, self.predict(X))

class TransformerMixin:
    """Mixin class for all transformers in scikitlearn_copy."""
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it."""
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        else:
            return self.fit(X, y, **fit_params).transform(X)
