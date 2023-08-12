from typing import List, Tuple, NewType, Union, Any, TypeAlias

Number = Union[int, float]
Numbers = List[Number]
TimeSeriesObservation: TypeAlias = Number
TimeSeriesRealization = NewType("TimeSeriesRealization", List[TimeSeriesObservation])
UnivariateTimeSeriesData = NewType("UnivariateTimeSeriesData", TimeSeriesRealization)
MultivariateTimeSeriesData = NewType("MultivariateTimeSeriesData", List[UnivariateTimeSeriesData])
TimeSeriesData = Union[TimeSeriesRealization, MultivariateTimeSeriesData]
