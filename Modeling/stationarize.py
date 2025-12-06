import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from mlfinpy.util.frac_diff import frac_diff_ffd, plot_min_ffd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class STLModel:
  def __init__(self,
               data: pd.Series,
               log_smooth: bool = True,
               seasonal_smooth: int = 23,
               seasonal_periods: int = 15
               ):
    '''data(pd.Series): Временной ряд. Индекс обязательно должен быть в формате datetime!!!!!!!!!'''
    self.data = np.log(data) if log_smooth else data
    self.original_data = data
    self.log_smooth = log_smooth


    '''
      seasonal_smooth(int): Length of the seasonal smoother. Must be an odd integer, and should
      normally be >= 7 (default).
    '''
    self.seasonal_smooth = seasonal_smooth

    '''
      seasonal_periods(int): Количество периодов в полном цикле сезонности.
    '''
    self.seasonal_periods = seasonal_periods

    self.model = STL(
          self.data,
          period=self.seasonal_periods,
          seasonal=seasonal_smooth,
          robust=True
      )

  def fit(self):
    try:
        self.model = self.model.fit()
        return self

    except (RuntimeWarning, ValueError, FloatingPointError) as e:
        return None
    except Exception as e:
        return None

  def get_components(self, make_plot=False, last_ticks_plot=100, pic_file='STL_tsr.png'):
      if last_ticks_plot is None:
          last_ticks_plot = len(self.data)

      trend = self.model.trend
      seasonal = self.model.seasonal
      residual = self.model.resid

      components = pd.DataFrame(
          np.c_[trend, seasonal, residual],
          columns=["trend", "seasonal", "residual"],
          index=self.data.index,
      )

      if make_plot:
          self.plot_trend_seasonal_residual(components, is_log=self.log_smooth, last_ticks_plot=last_ticks_plot, pic_file=pic_file)

      return components


  @staticmethod
  def plot_trend_seasonal_residual(components, is_log=False, last_ticks_plot=100, pic_file='STL_tsr.png'):
    dates = components.index

    if is_log:
      trend = np.exp(components['trend'])
      residual = np.exp(components['residual'])
      seasonal_multiplier= np.exp(components['seasonal'])
      seasonal_pct = (seasonal_multiplier - 1) * 100
    else:
      trend = components['trend']
      residual = components['residual']
      seasonal_pct = (components['seasonal'] / trend) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

    ax1.plot(dates[-last_ticks_plot:], trend[-last_ticks_plot:], ls='-.')
    ax1.set_title('Trend')

    ax2.plot(dates[-last_ticks_plot:], seasonal_pct[-last_ticks_plot:], ls='--')
    ax2.set_title('Seasonal')

    ax3.plot(dates[-last_ticks_plot:], residual[-last_ticks_plot:], ls='--')
    ax3.set_title('Residual')

    plt.suptitle('Trend, seasonal and residual')
    plt.savefig('Modeling/images/' + pic_file)

  def reconstruct(self, components, last_ticks_plot=100, original=None, make_compare_plot=False, pic_file='STL_compare.png'):
    if components is None:
        components = self.get_components(make_plot=False)
    dates = components.index

    trend = components['trend']
    seasonal = components['seasonal']
    residual = components['residual']

    if self.log_smooth:
        reconstructed = np.exp(trend + seasonal + residual)
    else:
        reconstructed = trend + seasonal + residual


    reconstructed_series = pd.Series(
        reconstructed.values.flatten() if hasattr(reconstructed, 'values') else reconstructed,
        index=dates,
        name='reconstructed'
    )

    if make_compare_plot:
      self.plot_prices_compare(dates, reconstructed, original, period=last_ticks_plot, pic_file=pic_file)

    return reconstructed_series


  @staticmethod
  def plot_prices_compare(dates, reconstructed, original, period=100, pic_file='STL_compare.png'):
    plt.figure(figsize=(15, 10))
    plt.plot(dates[-period:], reconstructed[-period:], label='reconstructed',ls='-.')
    plt.plot(dates[-period:], original[-period:], label='original', ls='--')
    plt.legend()
    plt.title('Compare price reconstruction')
    plt.savefig('Modeling/images/' + pic_file)




class HolterWintersModel:
  def __init__(self,
               data: pd.Series,
               log_smooth: bool = True,
               seasonal_periods: int = 15,
               trend_mode: str = 'add',
               seasonal_mode: str = 'mul',
               use_boxcox: bool = False,
               ):
    '''data(pd.Series): Индекс обязательно должен быть в формате datetime!!!!!!!!!'''

    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame должен содержать ровно одну колонку для Holt-Winters.")
        data = data.iloc[:, 0]

    self.data = np.log(data) if log_smooth else data
    self.original_data = data
    self.log_smooth = log_smooth


    '''
      seasonal_periods(int): Количество периодов в полном цикле сезонности.
    '''
    self.seasonal_periods = seasonal_periods

    '''
      trend_mode(str): Тип трендовой компоненты. Возможные значения: "add", "mul",
      seasonal_mode(str): Тип сезонной компоненты. Возможные значения: "add", "mul",
      use_boxcox(bool): Применять ли преобразование Boxcox. При log_smooth = True, use_boxcox автоматиечски False.
    '''
    self.trend_mode = trend_mode
    self.seasonal_mode = seasonal_mode
    self.use_boxcox = False if self.log_smooth else use_boxcox



    self.model = ExponentialSmoothing(
      self.data,
      seasonal_periods=seasonal_periods,
      trend=trend_mode,
      seasonal=seasonal_mode,
      use_boxcox=use_boxcox,
      initialization_method="estimated",
    )

  def fit(self):
    try:
        self.model = self.model.fit()
        return self

    except (RuntimeWarning, ValueError, FloatingPointError) as e:
        raise e
        return None
    except Exception as e:
        raise e
        return None

  def get_components(self, make_plot=False, last_ticks_plot=100, pic_file='HW_tsr.png'):
      if last_ticks_plot is None:
          last_ticks_plot = len(self.data)

      trend = self.model.level
      if self.trend_mode == 'add':
          trend += self.model.trend
      else:
          trend *= self.model.trend

      seasonal = self.model.season

      if self.log_smooth:
          if self.seasonal_mode == 'add':
              residual = self.data - (trend + seasonal)
          else:  
              residual = self.data - (trend + seasonal) 
      else:
          if self.seasonal_mode == 'add':
              residual = self.data - (trend + seasonal)
          else:
              residual = self.data - (trend * seasonal)


      components = pd.DataFrame(
          np.c_[trend, seasonal, residual],
          columns=["trend", "seasonal", "residual"],
          index=self.data.index,
      )

      if make_plot:
          self.plot_trend_seasonal_residual(components, is_log=self.log_smooth, last_ticks_plot=last_ticks_plot, pic_file=pic_file)

      return components


  @staticmethod
  def plot_trend_seasonal_residual(components, is_log=False, last_ticks_plot=100, pic_file='HW_tsr.png'):
    dates = components.index

    if is_log:
      trend = np.exp(components['trend'])
      residual = np.exp(components['residual'])
      seasonal_multiplier= np.exp(components['seasonal'])
      seasonal_pct = (seasonal_multiplier - 1) * 100
    else:
      trend = components['trend']
      residual = components['residual']
      seasonal_pct = (components['seasonal'] / trend) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

    ax1.plot(dates[-last_ticks_plot:], trend[-last_ticks_plot:], ls='-.')
    ax1.set_title('Trend')

    ax2.plot(dates[-last_ticks_plot:], seasonal_pct[-last_ticks_plot:], ls='--')
    ax2.set_title('Seasonal')

    ax3.plot(dates[-last_ticks_plot:], residual[-last_ticks_plot:], ls='--')
    ax3.set_title('Residual')

    plt.suptitle('Trend, seasonal and residual')
    plt.savefig('Modeling/images/' + pic_file)

  def reconstruct(self, components, last_ticks_plot=100, original=None, make_compare_plot=False, pic_file='HW_compare.png'):
    if components is None:
        components = self.get_components(make_plot=False)
    dates = components.index

    trend = components['trend']
    seasonal = components['seasonal']
    residual = components['residual']


    reconstructed = self._reconstruct_holt_winters(trend, seasonal, residual)

    reconstructed_series = pd.Series(
        reconstructed.values.flatten() if hasattr(reconstructed, 'values') else reconstructed,
        index=dates,
        name='reconstructed'
    )

    if make_compare_plot:
      self.plot_prices_compare(dates, reconstructed, original, period=last_ticks_plot, pic_file=pic_file)

    return reconstructed_series


  def _reconstruct_holt_winters(self, trend, seasonal, residual):
      if self.log_smooth:
          return np.exp(trend + seasonal + residual)

      base = trend

      if self.seasonal_mode == 'add':
          model_value = base + seasonal
      else:
          model_value = base * seasonal

      return model_value + residual

  @staticmethod
  def plot_prices_compare(dates, reconstructed, original, period=100, pic_file='HW_compare.png'):
    plt.figure(figsize=(15, 10))
    plt.plot(dates[-period:], reconstructed[-period:], label='reconstructed',ls='-.')
    plt.plot(dates[-period:], original[-period:], label='original', ls='--')
    plt.legend()
    plt.title('Compare price reconstruction')
    plt.savefig('Modeling/images/' + pic_file)



class FracDiffModel:
  def __init__(self,
               data: pd.DataFrame,
               log_smooth: bool = True,
               diff_amt: float = 0.5,
               ):
    '''data(pd.DataFrame): Индекс обязательно должен быть в формате datetime!!!!!!!!!'''

    if isinstance(data, pd.Series):
        data = data.to_frame(name='close')  
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame должен содержать ровно одну колонку")
    else:
        raise ValueError("data должен быть Series или DataFrame")

    

    self.data = np.log(data) if log_smooth else data
    self.original_data = data
    self.log_smooth = log_smooth

    '''
      diff_amt(float): Differencing amount.
    '''
    self.diff_amt = diff_amt

  
  def fit(self):
    try:
        self.frac_diff_series = frac_diff_ffd(self.data, self.diff_amt)
        return self

    except (RuntimeWarning, ValueError, FloatingPointError) as e:
        raise e
        return None
    except Exception as e:
        raise e
        return None

  def get_components(self, make_plot=False, last_ticks_plot=100, pic_file='FD_tsr.png'):
      if last_ticks_plot is None:
          last_ticks_plot = len(self.data)

      residual = self.frac_diff_series.iloc[:, 0]
      components = pd.DataFrame({'residual': residual}, index=self.data.index)

      if make_plot:
          plt.figure(figsize=(10, 8))
          plt.plot(components.index[-last_ticks_plot:], components[-last_ticks_plot:]['residual'], label='residual')
          plt.suptitle('Fractional diff result')
          plt.savefig('Modeling/images/' + pic_file)
      return components

  def plot_min_ffd(self, pic_file='plot_min_ffd.png'):
    ax = plot_min_ffd(self.data)
    fig = ax.get_figure()
    fig.savefig('Modeling/images/' + pic_file)
    plt.close(fig)





if __name__ == "__main__":
    data = pd.read_csv('DataBase/CLOSE.csv')
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
    # data1 = pd.DataFrame({'close' : np.array(data['SBER'])}, index=data['datetime']) 
      # ИЛИ
    data1 = pd.Series(np.array(data['SBER']), index=data['datetime'], name='close')
    print(data1.head())


    print('Start STL..')
    stat = STLModel(data=data1, log_smooth=False, seasonal_periods=15).fit()
    components = stat.get_components(make_plot=True)
    reconstructed = stat.reconstruct(components, original=data1, make_compare_plot=True)
    print('STL complete')

    print('Start HW...')
    stat = HolterWintersModel(data=data1, log_smooth=True, trend_mode='add', seasonal_mode='add').fit()
    components = stat.get_components(make_plot=True, last_ticks_plot=200)
    reconstructed = stat.reconstruct(components, original=data1, make_compare_plot=True, last_ticks_plot=200)
    print('HW complete')

    print('Start FRACDIFF')
    stat = FracDiffModel(data=data1, log_smooth=False, diff_amt = 0.3).fit()
    components = stat.get_components(make_plot=True, last_ticks_plot=300)
    stat.plot_min_ffd()
    print('FRACDIFF complete')


