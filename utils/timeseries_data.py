import os
from abc import ABC
from abc import abstractmethod
from datetime import datetime, timedelta

import pandas as pd
import timeseries as tss

default_dir_path = "data"


class TimeSeriesData(ABC):

    def __init__(self, data_type, data_name, dirpath=default_dir_path):
        self.dirpath = dirpath
        self.data_type = data_type
        self.data_name = data_name
        self.ts = None
        self.train_interval = None
        self.val_interval = None
        self.test_interval = None
        self.pred_steps = None
        self.pred_jump = None
        self.load()

    def load(self):
        if not os.path.isdir(self.dirpath):
            raise Exception(f"Dictionary '{self.dirpath}' does not exist")
        self.__load__()
        assert self.ts is not None
        self.__set_competition_params__()
        assert self.train_interval is not None
        assert self.val_interval is not None
        assert self.test_interval is not None
        assert self.pred_steps is not None
        assert self.pred_jump is not None

    @abstractmethod
    def __load__(self):
        ...

    @abstractmethod
    def __set_competition_params__(self):
        ...


class ForexGBPUSDData(TimeSeriesData):

    def __init__(self, dirpath=default_dir_path):
        super().__init__("Forex Currencies Rates", "GBP/USD", dirpath)

    def __load__(self):
        forex_dir_path = os.path.join(self.dirpath,
                                      "gbpusd_one-minute_forex_intraday_data")
        if not os.path.isdir(forex_dir_path):
            raise Exception(f"Dictionary '{forex_dir_path}' does not exist")

        forex_prefix = "gbpusd_one-minute_forex_intraday_data_"

        def forex_gbpusd_path(date):
            return os.path.join(forex_dir_path,
                                f"{forex_prefix}{str(date)[:-9]}.csv")

        def read_forex_day(date):
            df = pd.read_csv(forex_gbpusd_path(date), sep=";")
            float_cols = ["Close", "Open", "High", "Low"]
            df.columns = ["Date"] + float_cols + ["Volume", "Ticks"]
            for col in float_cols:
                df[col] = df[col].apply(lambda s: s.replace(",", ".")).astype(
                    float)
            df.Date = pd.to_datetime(df.Date, format="%d.%m.%Y %H:%M:%S")
            df.set_index("Date", inplace=True)
            df.sort_index(ascending=True, inplace=True)
            return df

        forex_begin_date = datetime(2021, 7, 25)
        forex_end_date = datetime(2021, 8, 16)

        forex_gpdusd_data = pd.DataFrame([])

        date = forex_begin_date

        while date <= forex_end_date:
            try:
                day_data = read_forex_day(date)
                forex_gpdusd_data = forex_gpdusd_data.append(day_data)
            except:
                pass
            date += timedelta(days=1)

        self.ts = forex_gpdusd_data["Close"].copy()[datetime(2021, 7, 25, 23):]
        del forex_gpdusd_data

    def __set_competition_params__(self):
        train_datetime = datetime(2021, 7, 26)
        val_datetime = datetime(2021, 8, 13)
        test_datetime = datetime(2021, 8, 13, 21, 59)
        test_end_datetime = self.ts.index[-1] + timedelta(minutes=1)
        self.train_interval = tss.Interval(self.ts, train_datetime,
                                           val_datetime)
        self.val_interval = tss.Interval(self.ts, val_datetime, test_datetime)
        self.test_interval = tss.Interval(self.ts, test_datetime,
                                          test_end_datetime)
        self.pred_steps = 60  # 1 hour
        self.pred_jump = 10  # ten minutes


class EEGData(TimeSeriesData):

    def __init__(self, dirpath=default_dir_path):
        super().__init__("EEG", "Fp1", dirpath)

    def __load__(self):
        eeg_data = pd.read_csv(
            os.path.join(self.dirpath, "eeg_features_raw.csv")
        )
        signal = "Fp1"
        self.ts = eeg_data[signal].copy()
        del eeg_data

    def __set_competition_params__(self):
        train_point = 500
        val_point = 6000
        test_point = 7000
        test_end_point = self.ts.index[-1] + 1
        self.train_interval = tss.Interval(self.ts, train_point, val_point)
        self.val_interval = tss.Interval(self.ts, val_point, test_point)
        self.test_interval = tss.Interval(self.ts, test_point, test_end_point)
        self.pred_steps = 30
        self.pred_jump = 7


class WebsiteVisitsData(TimeSeriesData):

    def __init__(self, dirpath=default_dir_path):
        super().__init__("Website Visits", "Page Loads", dirpath)

    def __load__(self):
        website_data = pd.read_csv(
            os.path.join(self.dirpath, "daily-website-visitors.csv")
        )
        website_data["Date"] = pd.to_datetime(website_data["Date"],
                                              format="%m/%d/%Y")
        website_data.set_index("Date", inplace=True)
        website_data.sort_index(ascending=True, inplace=True)
        for col in website_data.columns:
            if col not in {"Row", "Day", "Day.Of.Week"}:
                website_data[col] = website_data[col].apply(
                    lambda s: int(s.replace(",", "")))
        self.ts = website_data["Page.Loads"].copy()
        del website_data

    def __set_competition_params__(self):
        train_datetime = self.ts.index[150]
        val_datetime = datetime(2018, 9, 1)  # Monday
        test_datetime = datetime(2019, 5, 7)  # Monday
        test_end_datetime = self.ts.index[-1] + timedelta(days=1)
        self.train_interval = tss.Interval(self.ts, train_datetime,
                                           val_datetime)
        self.val_interval = tss.Interval(self.ts, val_datetime, test_datetime)
        self.test_interval = tss.Interval(self.ts, test_datetime,
                                          test_end_datetime)
        self.pred_steps = 7  # week
        self.pred_jump = 4  # 4 days


class Covid19Data(TimeSeriesData):

    def __init__(self, dirpath=default_dir_path):
        super().__init__("Covid-19", "Argentina", dirpath)

    def __load__(self):
        covid_data = pd.read_csv(
            os.path.join(self.dirpath, "covid-data.csv")
        )
        covid_data["date"] = pd.to_datetime(covid_data["date"],
                                            format="%Y-%m-%d")
        covid_data.set_index("date", inplace=True)
        covid_data.sort_index(ascending=True, inplace=True)

        loc = "Argentina"
        self.ts = covid_data[covid_data.location == loc]["new_cases"][
            datetime(2020,3,3):].copy()
        del covid_data

    def __set_competition_params__(self):
        train_datetime = datetime(2020,4,1)
        val_datetime = datetime(2021, 4, 1)
        test_datetime = datetime(2021, 5, 1)
        test_end_datetime = self.ts.index[-1] + timedelta(days=1)
        self.train_interval = tss.Interval(self.ts, train_datetime,
                                           val_datetime)
        self.val_interval = tss.Interval(self.ts, val_datetime, test_datetime)
        self.test_interval = tss.Interval(self.ts, test_datetime,
                                          test_end_datetime)
        self.pred_steps = 14  # 2 weeks
        self.pred_jump = 1  # 1 day

# class (TimeSeriesData):

#     def __init__(self, dirpath=default_dir_path):
#         super().__init__("", "", dirpath)

#     def __load__(self):


#     def __set_competition_params__(self):
