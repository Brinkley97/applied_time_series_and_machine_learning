import yfinance as yf
import matplotlib.pyplot as plt
from django.shortcuts import render
from io import BytesIO
from django.http import HttpResponse

def index(request):
    return render(request, 'charts/index.html')

def get_chart(request):
    if request.method == 'POST':
        selected_stocks = request.POST.getlist('stocks')

        # Plot the selected stocks
        plt.figure(figsize=(10, 6))

        for stock in selected_stocks:
            stock_data = yf.Ticker(stock)
            hist = stock_data.history(period="1y")
            plt.plot(hist.index, hist['Close'], label=stock)

        plt.title("Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Return the image as a response
        return HttpResponse(buffer, content_type='image/png')

    return HttpResponse("Invalid Request", status=400)
