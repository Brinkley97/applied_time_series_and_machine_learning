# TSLearn App

# Folders + Files
```
tslearn_app/
├── manage.py                   # Django's command-line utility
|
├── finance_project/            # The main project configuration folder
│   ├── __init__.py
│   ├── settings.py             # Project settings (INSTALLED_APPS go here)
│   ├── urls.py                 # Main URL router (points to app-level urls.py)
│   ├── wsgi.py
│   └── asgi.py
|
├── core/                       # For shared code and utilities
│   ├── __init__.py
│   ├── models.py               # (e.g., an abstract BaseTimestampModel)
│   └── utils.py                # (e.g., a shared plotting function)
|
├── users/                      # For authentication and user profiles
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py               # (e.g., Profile model)
│   ├── urls.py
│   ├── views.py                # (e.g., login, logout, register, profile_edit)
│   └── templates/
│       └── users/
│           ├── login.html
│           └── profile.html
|
├── forecasts/                  # The "Forecasting Engine"
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py               # (ForecastModel)
│   ├── services.py             # <<< Your core forecasting logic goes here
│   ├── urls.py
│   └── views.py
|
├── stocks/                     # For all stock-related functionality
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py               # (e.g., Watchlist model)
│   ├── urls.py
│   ├── views.py                # (e.g., stock_chart_view, stock_forecast_view)
│   └── templates/
│       └── stocks/
│           ├── stock_chart.html
│           └── stock_forecast.html
|
├── weather/                    # For all weather-related functionality
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── views.py                # (e.g., weather_chart_view, weather_forecast_view)
│   └── templates/
│       └── weather/
│           ├── weather_chart.html
│           └── weather_forecast.html
|
├── dashboard/                  # For the main user dashboard
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   └── views.py                # (The view that aggregates data from other apps)
|
├── static/                     # For GLOBAL static files (CSS, JS, Images)
│   └── css/
│       └── style.css
│   └── js/
│       └── main.js
|
└── templates/                  # For GLOBAL base templates
    └── base.html               # The main site template that other templates extend

```