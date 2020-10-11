'''

Aru Raghuvanshi
12-10-2020

'''


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import warnings
from datetime import date, timedelta, datetime
import tkinter as tk
from tkinter import *
from PIL import ImageTk

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def run_predictor():
    COMPANY = get_company()  # 'BHARTIARTL.NS'
    TRAINING_END = get_date()  # '2020-10-05'
    TRAINING_START = '2012-01-01'

    try:

        d = datetime.strptime(TRAINING_END, '%Y-%m-%d')  # string to dt

    except (ValueError, UnboundLocalError):
        print('Invalid Date Format (yyyy-mm-dd). Use Hyphens with Date Format.')
        tk.Label(window,
                 text=f'Check yyyy-mm-dd format. Use hyphens as date seperator.',
                 fg="white", bg="firebrick",
                 width=60, height=2,
                 activebackground="indianred",
                 font=('arial', 10, ' bold ')).place(x=860, y=370)
    finally:
        weekno = d.weekday()

    if weekno < 5:

        # Get the stock quote
        df = web.DataReader(COMPANY, data_source='yahoo',
                            start=TRAINING_START, end=TRAINING_END)

        # Create a new dataframe with only the 'Close' column
        data = df.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)

        # Scale the all of the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Create the scaled training data set
        train_data = scaled_data[0:training_data_len, :]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []
        for i in range(120, len(train_data)):
            x_train.append(train_data[i - 120:i, 0])
            y_train.append(train_data[i, 0])

        # Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data into the shape accepted by the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # ------------------------------------------ NEURAL NETWORK MODEL ----------------------------------- ]

        # #Build the LSTM network model
        # model = Sequential()
        # model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
        # model.add(LSTM(units=50, return_sequences=False))
        # model.add(Dense(units=25))
        # model.add(Dense(units=1))

        # #Compile the model
        # model.compile(optimizer='adam', loss='mean_squared_error')

        # #Train the model
        # model.fit(x_train, y_train, batch_size=1, epochs=1)

        # model.save('stockprice_new.h5')

        model = load_model('stockprice_new.h5')

        # ---------------------------------------------- PREDICTION SIDE --------------------------------- ]

        # converting string to datetime
        day = datetime.strptime(TRAINING_END, '%Y-%m-%d')
        tar = day + timedelta(days=1)
        # converting datetime to string
        TARGET = tar.strftime('%Y-%m-%d')  # dt to string

        # Get the quote
        apple_quote = web.DataReader(
            COMPANY, data_source='yahoo', start=TRAINING_START, end=TRAINING_END)
        # Create a new dataframe
        new_df = apple_quote.filter(['Close'])
        # Get teh last 120 day closing price
        last_120_days = new_df[-120:].values
        # Scale the data to be values between 0 and 1
        last_120_days_scaled = scaler.transform(last_120_days)
        # Create an empty list
        X_test = []
        # Append teh past 120 days
        X_test.append(last_120_days_scaled)
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Get the predicted scaled price
        pred_price = model.predict(X_test)
        # undo the scaling
        pred_price = scaler.inverse_transform(pred_price)[0][0]
        pred_price = np.ceil(pred_price)
        print(pred_price)
        print(f"PREDICTED PRICE FOR {COMPANY} ON {TARGET}: ₹{pred_price}")
        tk.Label(window,
                 text=f"PREDICTED STOCK PRICE ON {TARGET}: ₹{pred_price}",
                 fg="darkorchid", bg="lawngreen",
                 width=50, height=2,
                 activebackground="indianred",
                 font=('arial', 15, ' bold ')).place(x=860, y=370)

        # Get the actual quote from yahoo finance
        try:
            apple_quote2 = web.DataReader(
                COMPANY, data_source='yahoo', start=TARGET, end=TARGET)
            print(f"ACTUAL PRICE OF {COMPANY} ON {TARGET}: ₹{round(apple_quote2['Close'][0], 1)}")
            tk.Label(window,
                     text=f"ACTUAL STOCK PRICE ON {TARGET}: ₹{round(apple_quote2['Close'][0], 1)}",
                     fg="lawngreen", bg="darkorchid",
                     width=50, height=2,
                     activebackground="indianred",
                     font=('arial', 15, ' bold ')).place(x=860, y=430)

        except:
            print('No Such Date or a Weekend')
            tk.Label(window,
                     text=f"Can't fetch actual data from weekend or future date",
                     fg="white", bg="firebrick",
                     width=50, height=2,
                     activebackground="indianred",
                     font=('arial', 10, ' bold ')).place(x=860, y=370)

        diff = abs(pred_price - round(apple_quote2['Close'][0], 1))
        tk.Label(window,
                 text=f"Diff in Prediction: ₹{diff}",
                 fg="white", bg="cadetblue",
                 width=50, height=2,
                 activebackground="indianred",
                 font=('arial', 10, ' bold ')).place(x=970, y=480)


    else:
        print(f"{TRAINING_END} is a \033[1;31mWeekend\033[0m.")
        print(f"The Stock Exchange doesn't operate on Weekends.")
        tk.Label(window,
                 text=f'{TRAINING_END} is Weekend',
                 fg="white", bg="tomato",
                 width=30, height=2,
                 activebackground="indianred",
                 font=('arial', 10, ' bold ')).place(x=860, y=370)


# ---------------------------------------------  G U I  -------------------------------------- ]

window = tk.Tk()

image = ImageTk.PhotoImage(file="bk2.jpg")
canvas = Canvas(width=image.width(), height=image.height(), bg='lightgreen')
canvas.create_image(0, 0, image=image, anchor='nw')
canvas.pack(expand='yes', fill='both')

window.attributes('-fullscreen', True)
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

lbl = tk.Label(window, text='STOCK MARKET PRICE PREDICTION USING AI',
               fg="white", bg="indianred",
               width=60, height=3,
               activebackground="Red",
               font=('arial', 25, ' bold ')).place(x=200, y=30)

# ----------------------------- CHOOSE COMPANY -------------------------------------------- ]

comp = tk.Label(window, text='CHOOSE COMPANY: ',
                fg="white", bg="dodgerblue",
                width=20, height=2,
                activebackground="firebrick",
                font=('arial', 12, ' bold ')).place(x=200, y=270)
variable = StringVar(window)
variable.set('BHARTIARTL.NS')  # default value
w = OptionMenu(window, variable, 'BHARTIAIRTL.NS', 'RELIANCE.NS').place(x=420, y=275)


def get_company():
    comp = []
    co = variable.get()
    comp.append(co)
    return comp[0]


gcobtn = tk.Button(window, command=get_company, text='OK',
                   fg="white", bg="yellowgreen",
                   width=10, height=1,
                   activebackground="firebrick",
                   font=('arial', 10, ' bold ')).place(x=560, y=275)

# ----------------------------- CHOOSE DATE -------------------------------------------- ]

ent1 = tk.Label(window, text='ENTER DATE IN YYYY-MM-DD: ',
                fg="white", bg="orchid",
                width=30, height=2,
                activebackground="skyblue",
                font=('arial', 12, ' bold ')).place(x=200, y=370)

ent = StringVar(window)
ent.set('')  # default value
e = tk.Entry(window, textvariable=ent,
             fg="white", bg="dodgerblue",
             width=20,
             font=('arial', 12, ' bold ')).place(x=520, y=380)


def get_date():
    date = []
    k = ent.get()
    date.append(k)
    return date[0]


gdbtn = tk.Button(window, command=get_date, text='Submit',
                  fg="white", bg="yellowgreen",
                  width=10, height=1,
                  activebackground="skyblue",
                  font=('arial', 10, ' bold ')).place(x=720, y=380)

# ----------------------------- PREDICTOR -------------------------------------------  ]

gcobtn = tk.Button(window, command=run_predictor, text='RUN PREDICTOR',
                   fg="white", bg="mediumslateblue",
                   width=20, height=3,
                   activebackground="indianred",
                   font=('arial', 20, ' bold ')).place(x=600, y=550)

# ----------------------------- PREDICTION -------------------------------------------  ]

ent1 = tk.Label(window, text='ENTER DATE IN YYYY-MM-DD: ',
                fg="white", bg="orchid",
                width=30, height=2,
                activebackground="skyblue",
                font=('arial', 12, ' bold ')).place(x=200, y=370)

# ------------------------------------------------------------------------------------  ]
btn2 = tk.Button(window, text='QUIT', command=window.destroy,
                 fg="white", bg="purple",
                 width=20, height=3,
                 activebackground="Red",
                 font=('arial', 15, ' bold ')).place(x=650, y=700)

window.mainloop()