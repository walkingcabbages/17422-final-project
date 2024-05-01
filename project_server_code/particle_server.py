from flask import Flask, render_template, request
import socket
import numpy as np 
import scipy as sp
from sklearn.svm import SVC
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from threading import Thread

frame_rate = 50 # accelerometer sampling rate
duration_s = 2.5 # sample duration in seconds
total_samples = int(frame_rate * duration_s)
window_size = 5

gesture_dict = {}
clf = None

prediction = ""
running = True

def median_filter(a):
    window_medians = sp.signal.medfilt(a, window_size) # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.medfilt.html
    return window_medians

def extract_ax(arr):
    return arr[:, 0:1].flatten()
def extract_ay(arr):
    return arr[:, 1:2].flatten()
def extract_az(arr):
    return arr[:, 2:3].flatten()

# Returns tuple of arrays of x, y peaks
def get_extrema(ax, ay):
    neg_ax = np.negative(ax)
    neg_ay = np.negative(ay)
    ax_peaks = sp.signal.find_peaks(ax, prominence=0.4)
    ay_peaks = sp.signal.find_peaks(ay, prominence=0.4)
    ax_min = sp.signal.find_peaks(neg_ax, prominence=0.4)
    ay_min = sp.signal.find_peaks(neg_ay, prominence=0.4)
    # plot_peaks(ax, ax_peaks, ax_min, "Ax peaks")
    # plot_peaks(ay, ay_peaks, ay_min, "Ay peaks")
    return (ax_peaks, ax_min, ay_peaks, ay_min)

def get_prominence(ax, ax_peaks, ax_mins, ay, ay_peaks, ay_mins):
    neg_ax = np.negative(ax)
    neg_ay = np.negative(ay)
    ax_peak_prominence = sp.signal.peak_prominences(ax, ax_peaks[0]) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences
    ay_peak_prominence = sp.signal.peak_prominences(ay, ay_peaks[0])
    ax_min_prominence = sp.signal.peak_prominences(neg_ax, ax_mins[0])
    ay_min_prominence = sp.signal.peak_prominences(neg_ay, ay_mins[0])
    return (ax_peak_prominence, ax_min_prominence, ay_peak_prominence, ay_min_prominence)

def get_feature_conditional(peaks, prominences, mode):
    if peaks[0].shape[0] == 0:
        return 0
    elif mode == 'AVG':
        return np.average(prominences[0])
    elif mode == 'SUM':
        return np.sum(prominences[0])
    elif peaks[0].shape[0] == 1 and mode == 'AVG-GAP':
        return 0
    elif mode == 'AVG-GAP':
        return np.average(np.diff(prominences[0]))

def featurize(arr):
    fvec = np.std(arr,axis=0)
    return fvec

	### amy
	# ax = median_filter(extract_ax(arr))
	# ay = median_filter(extract_ay(arr))
	# (ax_peaks, ax_mins, ay_peaks, ay_mins) = get_extrema(ax, ay)
	# (ax_peak_prominence, ax_min_prominence, ay_peak_prominence, ay_min_prominence) = get_prominence(ax, ax_peaks, ax_mins, ay, ay_peaks, ay_mins)
	# fvec = []
	# fvec.append(ax_peaks[0].shape[0]) # number of ax peaks
	# fvec.append(ay_peaks[0].shape[0]) # number of ay peaks
	# fvec.append(get_feature_conditional(ax_peaks, ax_peak_prominence, 'AVG')) # average prominence of ax maxima
	# fvec.append(get_feature_conditional(ay_peaks, ay_peak_prominence, 'AVG')) # average prominence of ay maxima
	
	# ### nandini
	# X_cord = arr[:, 0]
	# Y_cord = arr[:, 1]
	# Z_cord = arr[:, 2]
	# X_fft = np.abs(fft(X_cord))
	# Y_fft = np.abs(fft(Y_cord))
	# Z_fft = np.abs(fft(Z_cord))
	# fv = []
	# fv.append(np.mean(X_fft))
	# fv.append(np.mean(Y_fft))
	# fv.append(np.mean(Z_fft))
	# fv.append(np.max(X_fft))
	# fv.append(np.max(Y_fft))
	
	# ### alejandro
	# features = []
	# # Standard Deviation
	# features.append(np.std(arr, axis=0))
	# # Mean of absolute values
	# features.append(np.mean(np.abs(arr), axis=0))
	# # Max-Min difference
	# features.append(np.max(arr, axis=0) - np.min(arr, axis=0))
	# # Sum of absolute values
	# features.append(np.sum(np.abs(arr), axis=0))
	# # Sum of squares
	# features.append(np.sum(np.square(arr), axis=0))
	# features = np.concatenate(features)

	# return np.concatenate((fvec,fv,features))

# This is the part that trains a classifier
def train_ml_classifier(): 
   clf_model = RandomForestClassifier()
   X = []
   Y = []
   for key in gesture_dict.keys():
      gest_trials = gesture_dict[key]
      for trial in gest_trials:
         X.append(featurize(trial))
         Y.append(key)
   X = np.array(X)
   Y = np.array(Y)
   clf_model.fit(X, Y)
   return clf_model

# Parse the data stream to get the values from it
def parse_data_stream(data_stream):
   data = data_stream.split("\n")
   data_vals = []

   # scan tokens
   for i in range(len(data)):
      imu_vals = data[i]
      
      # if EOD, return
      if "EOD" in imu_vals:
         break
      
      # 3 axes per line
      values = []
      for val in imu_vals.split(" "):
         try:
            values.append(float(val))
         except ValueError:
            break
      if (len(values) == 21):
         data_vals.append(values)
   
   data_vals = np.array(data_vals)
   print(data_vals)
   return data_vals

app = Flask(__name__)

# start TCP server
port = 8080
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        
print ("Socket successfully created") 

# TODO: update with your IP here
s.bind(('10.0.0.10', port))
print ("socket binded to %s" %(port))

s.listen(5)      
print ("socket is listening")

newSocket, address = s.accept()
print ("Connection from ", address)

# ask particle for new sample
def collectTest():

   # send new command
   cmd = str(total_samples)
   cmd_b = bytearray()
   cmd_b.extend(map(ord, cmd))
   print ("Requesting " + cmd + " samples")
   newSocket.send(cmd_b)

   # receive response
   data_stream = ""
   while 1:
      receivedData = newSocket.recv(1024)
      if not receivedData:
         break
      else:
         data = receivedData.decode('utf-8')
         data_stream = data_stream + data
         
         # Continously receive the data until you hit EOD
         if "EOD\n" in data:
            return data_stream

# send prediction to particle
def sendPrediction(prediction):
   cmd = "trigger_function {pred}".format(pred=prediction)
   cmd_b = bytearray()
   cmd_b.extend(map(ord, cmd))
   print ("Sending prediction " + prediction)
   newSocket.send(cmd_b)
   
# collect tests until stopped
def collectTests():
   global running, prediction
   while running:
      stream = collectTest()
      X_in = featurize(parse_data_stream(stream)) # Featurization happens here
      curr_prediction = clf.predict([X_in])[0]
      sendPrediction(curr_prediction)
      prediction = prediction + "<br>" + curr_prediction # Collect prediction history
        
@app.route("/", methods = ['POST', 'GET'])
def index():
   global prediction, clf, gesture_dict, running
   gesture_dict = np.load("combined_gesture_dict.npy", allow_pickle=True).item()
   if clf is None: # If not trained, train a new model
      clf = train_ml_classifier()
   
   # check for post request
   if request.method == "POST":
      # print(request.form)
      # if beginning identifying data, receive new data stream and make inference
      if request.form["submit"] == "Start":
         prediction = "" # Reset prediction history
         running = True
         thread = Thread(target=collectTests)
         thread.start()
      # if reset, delete gesture dict and reset classifier
      elif request.form["submit"] == "Stop":
         running = False
   
   # if prediction history present, update page
   if (prediction != ""): 
      return render_template("index.html", pred=prediction)
   
   # else, only output trial count
   return render_template("index.html")

app.run(debug=True, use_reloader=False)