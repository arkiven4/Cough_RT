import numpy as np
#import librosa
import random, math
#from scipy.ndimage import zoom

SR_MODEL = 16000
window_samples = int(0.5 * SR_MODEL)      # 0.5s window
hop_samples    = int(0.05 * SR_MODEL)     # 50ms hop

def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, adaptive_method='percentile', th_l_multiplier = 0.1, th_h_multiplier = 2):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power
    
    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
    
    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""
                
    cough_mask = np.array([False]*len(x))

    #Define hysteresis thresholds
    
    if adaptive_method == 'percentile':
        signal_power = x**2
        seg_th_l = np.percentile(signal_power, 75) 
        seg_th_h = np.percentile(signal_power, 99.9)  
    elif adaptive_method == 'statistics':
        signal_power = x**2
        mean_power = np.mean(signal_power)
        std_power = np.std(signal_power)
        seg_th_l = mean_power + 1.0 * std_power
        seg_th_h = mean_power + 3.0 * std_power
    elif adaptive_method == 'combination':
        signal_power = x**2
        rms = np.sqrt(np.mean(np.square(x)))
        seg_th_l = np.percentile(signal_power, 75) 
        seg_th_h =  th_h_multiplier * rms
    elif adaptive_method == 'default':
        rms = np.sqrt(np.mean(np.square(x)))
        seg_th_l = th_l_multiplier * rms
        seg_th_h =  th_h_multiplier * rms

    #Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    return coughSegments, cough_mask

# def pre_process_audio_mel(audio, sr=16000):
#     S = librosa.feature.melspectrogram(
#         y=audio, sr=sr, n_fft=1024, hop_length=512,
#         fmin=50, fmax=2000, n_mels=64, power=2.0
#     )
#     S_db = librosa.power_to_db(S, ref=np.max)
#     S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    
#     resized = zoom(S_db, (64/S_db.shape[0], 64/S_db.shape[1]), order=1)
#     return resized.astype(np.float32)  # [64,64]

# def process_audio_with_original(audio_orig, sr_orig, session=None, min_cough_samples=0.0, padding=0.0):
#     min_cough_samples = int(min_cough_samples * SR_MODEL)   # 200ms min duration
#     padding = int(padding * SR_MODEL)
    
#     # Resample for model
#     audio_model = librosa.resample(audio_orig, orig_sr=sr_orig, target_sr=SR_MODEL)

#     # Sliding windows in model space
#     starts = np.arange(0, len(audio_model) - window_samples, hop_samples, dtype=int)
#     windows = [audio_model[s:s+window_samples] for s in starts]

#     processed = [pre_process_audio_mel(w) for w in windows]
#     batch_input = np.stack(processed, axis=0)

#     # Model inference
#     raw_predictions = session.run(None, {"input": batch_input})[0]
#     cough_probs = raw_predictions[:, 1] if raw_predictions.shape[1] > 1 else raw_predictions[:, 0]
#     cough_mask_windows = cough_probs > 0.5

#     # Map window-level mask to sample-level (model rate)
#     cough_mask_samples = np.zeros(len(audio_model), dtype=bool)
#     sel = np.nonzero(cough_mask_windows)[0]
#     if len(sel) > 0:
#         win_idx = np.arange(window_samples)
#         idx = starts[sel, None] + win_idx[None, :]
#         idx = idx.ravel()
#         idx = idx[idx < len(audio_model)]
#         cough_mask_samples[idx] = True

#     # Run-length encode in model space
#     mask = cough_mask_samples.astype(int)
#     diff = np.diff(mask, prepend=0, append=0)
#     seg_starts = np.flatnonzero(diff == 1)
#     seg_ends   = np.flatnonzero(diff == -1)

#     cough_mask_final = np.zeros_like(cough_mask_samples, dtype=bool)
#     segments_orig = []
#     scale = sr_orig / SR_MODEL

#     for s, e in zip(seg_starts, seg_ends):
#         if (e - s) >= min_cough_samples:
#             s_pad = max(0, s - padding)
#             e_pad = min(len(audio_model), e + padding)
#             cough_mask_final[s_pad:e_pad] = True

#             # Map back to original indices
#             s_orig = int(s_pad * scale)
#             e_orig = int(e_pad * scale)
#             segments_orig.append(audio_orig[s_orig:e_orig])

#     return segments_orig, cough_mask_final
