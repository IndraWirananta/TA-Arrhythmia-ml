from __future__ import division, print_function
import os
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from scipy.signal import find_peaks
from wfdb import rdrecord, rdann
from utils import *
from filtering import butter_filter
from filtering import simple_moving_average
from config import get_config
import deepdish as dd
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from ecgdetectors import Detectors
from bisect import bisect_left
import statistics
from pylab import *
import pywt
"""
This database includes 25 long-term ECG recordings of human subjects with atrial fibrillation (mostly paroxysmal).

Of these, 23 records include the two ECG signals (in the .dat files);
records 00735 and 03665 are represented only by the rhythm (.atr) and unaudited beat (.qrs) annotation files.

The individual recordings are each 10 hours in duration, and contain two ECG signals each sampled at 250 samples
per second with 12-bit resolution over a range of ±10 millivolts. The original analog recordings were made at 
Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center) using ambulatory ECG recorders with
a typical recording bandwidth of approximately 0.1 Hz to 40 Hz. The rhythm annotation files (with the suffix .atr)
were prepared manually;

these contain rhythm annotations of types :
1. (AFIB (atrial fibrillation), 
2. (AFL (atrial flutter), 
3. (J (AV junctional rhythm), 
4. and (N (used to indicate all other rhythms).

(The original rhythm annotation files, still available in the old directory,
used AF, AFL, J, and N to mark these rhythms; the atr annotations in this
directory have been revised for consistency with those used for the MIT-BIH
Arrhythmia Database.) Beat annotation files (with the suffix .qrs) were 
prepared using an automated detector and have not been corrected manually.

For some records, manually corrected beat annotation files (with the suffix .qrsc)
are available. (The .qrs annotations may be useful for studies of methods for automated
AF detection, where such methods must be robust with respect to typical QRS detection
errors. The .qrsc annotations may be preferred for basic studies of AF itself, where
QRS detection errors would be confounding.) Note that in both .qrs and .qrsc files,
no distinction is made among beat types (all beats are labelled as if normal).
"""

"""
.dat contains sample #, ECG1, ECG2
               75000     45    67
               75001     48    69
               75002     44    80

.atr contains time,         sample #,  Type, Sub, Chan, Num, Aux
              12:00:00.028   1381607    +     0    0     0   (AFIB
              12:20:00.028   1381634    +     0    0     0   (N
              12:40:00.028   1381654    +     0    0     0   (AFL
"""

def preprocess( split ):
    
    #00735 and 03665 are represented only by the rhythm (.atr) and unaudited beat (.qrs annotation files, thus wont be used
    # nums = ['04746']
    nums = ['04015','04043','04048','04126','04746','04908','04936','05091','05121','06426','06453','06995','07162','07859','07879','07910','08215','08219','08378','08405','08434','08455']
    # nums = ['100','101','102','103','104','105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124']
    features = ['ECG1', 'ECG2'] 

    def dataSaver(dataSet, datasetname, labelsname):
        classes = ['(AFIB','(N','(AFL',"(J"]
        Nclass = len(classes)
        datadict, datalabel= dict(), dict()

        final_features = []
       

        for feature in features:
            datadict[feature] = list()
            datalabel[feature] = list()


        def dataprocess():
          input_size = config.input_size 
          for num in tqdm(dataSet):
            
            record = rdrecord('mitdb/'+ num, smooth_frames= True)
            
            signals0 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0])).tolist() #ECG1
            signals1 = preprocessing.scale(np.nan_to_num(record.p_signal[:,1])).tolist() #ECG2

            rawSignals0 = signals0
    
            signalMean = mean(signals0)
            signals0 = [x - signalMean for x in signals0]

            signalMean = mean(signals1)
            signals0 = [x - signalMean for x in signals1]

            signals0 = simple_moving_average(signals0, 10)
            signals1 = simple_moving_average(signals1, 10)

            if config.use_filter:
            
                filter_01 = butter_filter(signals0, filter_type="highpass", order=3, cutoff_freqs=[1], sampling_freq=250)
                filter_02 = butter_filter(filter_01, filter_type="bandstop", order=3, cutoff_freqs=[58, 62], sampling_freq=250)
                signals0 = butter_filter(filter_02, filter_type="lowpass", order=4, cutoff_freqs=[25], sampling_freq=250)
                

                filter_11 = butter_filter(signals1, filter_type="highpass", order=3, cutoff_freqs=[1], sampling_freq=250)
                filter_12 = butter_filter(filter_11, filter_type="bandstop", order=3, cutoff_freqs=[58, 62], sampling_freq=250)
                signals1 = butter_filter(filter_12, filter_type="lowpass", order=4, cutoff_freqs=[25], sampling_freq=250)

            feature0, feature1 = record.sig_name[0], record.sig_name[1]

            global lappend0, lappend1, dappend0, dappend1 
            lappend0 = datalabel[feature0].append
            lappend1 = datalabel[feature1].append
            dappend0 = datadict[feature0].append
            dappend1 = datadict[feature1].append
                        
            ann = rdann('mitdb/'+ num, extension='atr', sampfrom = 0, sampto = record.sig_len)
            peaks = ann.sample

            for peak in tqdm(peaks[1:]):
              start, end =  peak-input_size//2 , peak+input_size//2
              ann = rdann('mitdb/'+ num, extension='atr', sampfrom = start, sampto = end)



            #  Wavelet features
            #   from collections import Counter
            #   import scipy.stats
              def interquartile_range(data):
                quartile_1, quartile_3 = np.percentile(data, [25, 75])
                return quartile_3 - quartile_1

            #   def calculate_entropy(list_values):
            #     counter_values = Counter(list_values).most_common()
            #     probabilities = [elem[1]/len(list_values) for elem in counter_values]
            #     entropy=scipy.stats.entropy(probabilities)
            #     print("entropy")
            #     print(entropy)
            #     return entropy

            #   def calculate_statistics(list_values):
            #     n5 = np.nanpercentile(list_values, 5)
            #     n25 = np.nanpercentile(list_values, 25)
            #     n75 = np.nanpercentile(list_values, 75)
            #     n95 = np.nanpercentile(list_values, 95)
            #     mav = np.mean(np.abs(list_values))
            #     median = np.nanpercentile(list_values, 50)
            #     avp = np.mean(list_values**2)
            #     mean = np.nanmean(list_values)
            #     std = np.nanstd(list_values)
            #     var = np.nanvar(list_values)
            #     rms = np.nanmean(np.sqrt(list_values**2))

            #     # max_wv = max(list_values)
            #     # min_wv  = min(list_values)
            #     # # mode_wv  = statistics.mode(list_values)
            #     # range_wv  = max_wv-min_wv
            #     # quartile_wv  = interquartile_range(list_values)
            #     # skew_wv  = scipy.stats.skew(list_values)
            #     # return [n5, n25, n75, n95, median, mean, std, var, rms, mav, avp,max_wv, min_wv, mode_wv, range_wv, quartile_wv, skew_wv]
            #     return [n5, n25, n75, n95, mean, std, var, rms, mav, avp]

            #   def calculate_crossings(list_values):
            #     zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
            #     no_zero_crossings = len(zero_crossing_indices)
            #     mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
            #     no_mean_crossings = len(mean_crossing_indices)
            #     print("[no_zero_crossings, no_mean_crossings]")
            #     print([no_zero_crossings, no_mean_crossings])
            #     return [no_zero_crossings, no_mean_crossings]

            #   def get_features(list_values):
            #     entropy = calculate_entropy(list_values)
            #     crossings = calculate_crossings(list_values)
            #     statistics = calculate_statistics(list_values)
            #     # print ("FEATURES :", [entropy] + crossings + statistics)
            #     # print()
            #     return [entropy] + crossings + statistics
            
            #   import pywt
              
            # wavelet end

              def to_dict(chosenSym):
                y = [0]*Nclass
                y[classes.index(chosenSym)] = 1

                def find_rr_single_detector(signals):
                    detectors = Detectors(config.sample_rate)

                    r_peaks = detectors.pan_tompkins_detector(signals)

                    # -----------------------------------------------------------------------
                    #                     MAX for number of heartbeat
                    # -----------------------------------------------------------------------
    
                    mode_len = config.sample_heartbeat

                    if len(r_peaks) >= mode_len:
                        temp_arr = r_peaks.copy()
                        ann_heartbeat_idx = take_closest(temp_arr,config.input_size//2)#middle indexes
                        temp_arr.remove(ann_heartbeat_idx)
                        std_peaks = [ann_heartbeat_idx]

                        for y in range(mode_len-1):
                            new_heartbeat = take_closest(temp_arr,ann_heartbeat_idx)
                            std_peaks.append(new_heartbeat)
                            temp_arr.remove(new_heartbeat)

                        std_peaks.sort()
                        r_peaks = std_peaks
                    # -----------------------------------------------------------------------
                    else : 
                        return []

                    rr_intervals = list()
                   
                    if len(r_peaks) == mode_len:
                        for i in range(mode_len - 1):
                            if len(rr_intervals)==mode_len-1 :
                                rr_intervals[i] = (abs(r_peaks[i]-r_peaks[i+1]) + rr_intervals[i])/2
                            else :
                                rr_intervals.append(abs(r_peaks[i]-r_peaks[i+1]))


                 

                    import scipy.stats
                    max_rr = max(rr_intervals)
                    min_rr = min(rr_intervals)
                    avg_rr = sum(rr_intervals)/len(rr_intervals)
                    std_rr = np.std(rr_intervals)
                    median_rr = statistics.median(rr_intervals)
                    mode_rr = statistics.mode(rr_intervals)
                    variance_rr = statistics.variance(rr_intervals)
                    range_rr = max_rr-min_rr
                    quartile_rr = interquartile_range(rr_intervals)
                    # skew_rr = scipy.stats.skew(rr_intervals)
                    # kurtosis_rr = scipy.stats.kurtosis(rr_intervals)
                    

                    # wavelet


                    test = r_peaks
                    # time = np.linspace(0, 800/250, 800)
                    # print(test)
                    # samll = list()
                    # for i,x in enumerate(test):
                    #     samll.append(x-200)
                    # print(samll)
                    # # print(signals[:1000])
                    # plt.plot(time,signals[200:1000], color='blue', alpha =1, linewidth=1,markersize=20,marker=".",markerfacecolor='red',markevery=samll, linestyle='-', label='R peak')
                    # plt.legend()
                    # plt.xlabel('Time (s)')
                    # plt.ylabel('Amplitude')
                    # plt.title("R peak detection using Pan & Tompkins Algorithm")
                    # plt.show()

                    # features = [max_rr,min_rr,avg_rr,std_rr,median_rr,mode_rr, variance_rr, range_rr,quartile_rr,skew_rr,kurtosis_rr]
                    features = [max_rr,min_rr,avg_rr,std_rr,median_rr,mode_rr, variance_rr, range_rr,quartile_rr]

                    return features

                sig1res = find_rr_single_detector(signals0[start:end])
                if len(sig1res) != 0 :
                    # list_wavelet = pywt.wavedec(signals0[start:end], 'sym5')
                    
                    # wavelet_features = []
                    # print("length :", len(list_wavelet))
                    # for coeff in list_wavelet:
                        
                    #     wavelet_features += get_features(coeff)

                    # print("len(wavelet_features)")
                    # print(len(wavelet_features))
                    # sig1res.extend(wavelet_features)
                    print("sig1res")
                    print(len(sig1res))
                    print()
                    lappend0(y)
                    dappend0(sig1res)
                else:
                    print("data 1 expunged")

                sig2res = find_rr_single_detector(signals1[start:end])
                if len(sig2res) != 0 :
                    # list_wavelet = pywt.wavedec(signals1[start:end], 'sym5')
                    
                    # wavelet_features = []
                    # for coeff in list_wavelet:
                    #     wavelet_features += get_features(coeff)
                    # sig2res.extend(wavelet_features)
                    lappend1(y)
                
                    dappend1(sig2res)
                else:
                    print("data 2 expunged")
 
              annAuxNote = ann.aux_note
            #   if annAuxNote[0] == '(N':

            #     time = np.linspace(0, 1200/250, 1200)
             
            #     # Plot the preprocessed ECG signal data in blue with a dashed line
            #     plt.plot(time,rawSignals0[start:end], color='red', linestyle='dashed',alpha=0.6, label='Raw ECG signal')

            #     # Plot the processed ECG signal data in red with a solid line
            #     plt.plot(time,signals0[start:end], color='blue', linestyle='solid', label='Processed ECG signal')

            #     # Add a legend to the plot
            #     plt.legend()

            #     # Add x- and y-axis labels
            #     plt.xlabel('Time (s)')
            #     plt.ylabel('Amplitude')
            #     plt.title("Normal Sinus Rhythm ECG Signals")

            #     # Show the plot
            #     plt.show()
              if len(annAuxNote)>0:
                to_dict(annAuxNote[0])
            
        dataprocess()
       
        dd.io.save(datasetname, datadict)
        #pyrightconfig.json to solve unreachable by adjusting config
        dd.io.save(labelsname, datalabel)

    dataSaver(nums, 'mitdb/X-nonfull-rrOnly'+str(config.sample_heartbeat)+'hr.hdf5', 'mitdb/y-nonfull-rrOnly'+str(config.sample_heartbeat)+'hr.hdf5')
    # dataSaver(nums, 'mitdb/test.hdf5', 'mitdb/testlabel.hdf5')

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def main(config):
    config.sample_heartbeat = 6
    preprocess(config.split) 


if __name__=="__main__":
    config = get_config()
    main(config)

     # manual via peaks

            # peaks, _ = find_peaks(signals0, distance=50)
            # for peak in tqdm(peaks[2:-2]):
            #   annStart, annEnd = peak- 25, peak+25
            #   start, end =  peak-input_size//2 , peak+input_size//2
            #   ann = rdann('mitdb/'+ num, extension='atr', sampfrom = annStart, sampto = annEnd) #<--- aux_note used to be symbol
              
            #   def to_dict(chosenSym):
            #     y = [0]*Nclass
            #     y[classes.index(chosenSym)] = 1
            #     lappend0(y)
            #     lappend1(y)
            #     dappend0(signals0[start:end])
            #     dappend1(signals1[start:end])

            #   annAuxNote = ann.aux_note
            #   # remove some of "N" which breaks the balance of dataset 
            #   #if len(annAuxNote) == 1 and (annAuxNote[0] in classes) and (annAuxNote[0] != "N" or np.random.random()<0.15): <-- dont need to balance (N
            #   if len(annAuxNote)>0:
            #     to_dict(annAuxNote[0])
            #     print(num)
            #     print(annAuxNote)


            #   def find_rr(signals):
            #         detectors = Detectors(config.sample_rate)
            #         r_peaks1 = detectors.two_average_detector(signals)
            #         r_peaks2 = detectors.swt_detector(signals)
            #         r_peaks3 = detectors.hamilton_detector(signals)
            #         r_peaks4 = detectors.pan_tompkins_detector(signals)
                    
            #         all_peaks = [r_peaks1,r_peaks2,r_peaks3,r_peaks4]

            #         # test = mode([len(r_peaks1),len(r_peaks2),len(r_peaks3),len(r_peaks4)])
            #         # print("mode : ",test," detector : ",[len(r_peaks1),len(r_peaks2),len(r_peaks3),len(r_peaks4)])
                   

            #         # -----------------------------------------------------------------------
            #         #                     MAX for number of heartbeat
            #         # -----------------------------------------------------------------------
            #         mode_len = config.sample_heartbeat

            #         for i, peaks in enumerate(all_peaks):
            #             if len(peaks) >= mode_len:
            #                 temp_arr = peaks.copy()
            #                 ann_heartbeat_idx = take_closest(temp_arr,config.input_size//2)#middle indexes
            #                 temp_arr.remove(ann_heartbeat_idx)
            #                 std_peaks = [ann_heartbeat_idx]

            #                 for y in range(mode_len-1):
            #                     new_heartbeat = take_closest(temp_arr,ann_heartbeat_idx)
            #                     std_peaks.append(new_heartbeat)
            #                     temp_arr.remove(new_heartbeat)

            #                 std_peaks.sort()
            #                 all_peaks[i] = std_peaks
            #         # -----------------------------------------------------------------------

            #         rr_intervals = list()

            #         # print(len(r_peaks4), "peaks : ", all_peaks)
             
            #         for peaks in all_peaks:
            #             if len(peaks) == mode_len:
            #                 for i in range(mode_len - 1):
            #                     if len(rr_intervals)==mode_len-1 :
            #                         rr_intervals[i] = (abs(peaks[i]-peaks[i+1]) + rr_intervals[i])/2
            #                     else :
            #                         rr_intervals.append(abs(peaks[i]-peaks[i+1]))

            #         max_rr = max(rr_intervals)
            #         min_rr = min(rr_intervals)
            #         avg_rr = sum(rr_intervals)/len(rr_intervals)

            #         rr_intervals.extend([max_rr,min_rr,avg_rr])

                   


            #         return rr_intervals