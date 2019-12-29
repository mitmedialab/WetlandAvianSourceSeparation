import torch
from torch.utils import data
import os
import fnmatch
import librosa
from scipy import signal
import numpy as np
import torch.nn.functional as F
import random
import collections

#load 1 audio
def load_file(file):
    audio_raw, rate = librosa.load(file, sr=22050, mono=True)    
    return audio_raw, rate

#cleaning the input audio
def filt(audio_raw, rate):
    band = [800, 7000]  # Desired pass band, Hz
    trans_width = 100    # Width of transition from pass band to stop band, Hz
    numtaps = 250     # Size of the FIR filter.    
    edges = [0, band[0] - trans_width,
     band[0], band[1],
     band[1] + trans_width, 0.5*rate]
    taps = signal.remez(numtaps, edges, [0, 1, 0], Hz=rate, type='bandpass')     
    sig_filt = signal.lfilter(taps, 1, audio_raw)
    return sig_filt

# return the mag and phase for 1 stft in tensor
def _stft(audio):
    spec = librosa.stft(
        audio, n_fft=1022, hop_length=256)
    amp = np.abs(spec)
    phase = np.angle(spec)
    W = np.shape(amp)[0]
    H = np.shape(amp)[1]
    tch_mag = torch.empty(1, 1, W, H, dtype=torch.float)
    tch_mag[0, 0, :, :] = torch.from_numpy(amp)
    tch_phase = torch.empty(1, 1, W, H, dtype=torch.float)
    tch_phase[0, 0, :, :] = torch.from_numpy(phase)    
    return tch_mag, tch_phase

#threshold to clean the spectro and get the reference mask
#return 1 torch matrix of dimensions of the stft
def threshold(mag, mag_noise):
    gt_mask = torch.zeros(mag.shape[2], mag.shape[3])
    av = np.mean(mag[0, 0].numpy())     
    vari = np.var(mag[0, 0].numpy())
    param = av + np.sqrt(vari)*2   #threshold                 
    gt_mask = (mag[0, 0] > param).float()
    final_mask = (gt_mask*mag > mag_noise).float()
    return final_mask

#create the grid for the image
#if warp then Melspectro, if not, go from Mel to linear scale
def warpgrid_log(HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((1, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


#create image from the grid
def create_im(mag):
    magim = mag.unsqueeze(0).unsqueeze(0)
#Zero center data    
    m = torch.mean(magim)
    magim = magim - m
    grid_warp = torch.from_numpy(warpgrid_log(256, magim.shape[3], warp=True))
    magim = F.grid_sample(magim, grid_warp) 
    return torch.from_numpy(np.flipud(magim).copy())


def create_mask(mag):
    magim = mag.unsqueeze(0).unsqueeze(0)
    grid_warp = torch.from_numpy(warpgrid_log(256, magim.shape[3], warp=True))
    magim = F.grid_sample(magim, grid_warp)    
    return torch.from_numpy(np.flipud(magim).copy())


#remove a band in the horizontal direction in the spectrogram
def freq_mask(spec):
    fbank_size = np.shape(spec)
    rows , columns = fbank_size[0], fbank_size[1]
    #width of the band
    fact1 = np.random.randint(int(rows/100), int(rows/60))
    frame = np.zeros([fact1, columns])
    #position of the band on the y axis
    pos = random.randint(10, rows-fact1-1)
    up = np.ones([pos-1, columns])
    down = np.ones([rows-(pos+fact1)+1, columns])
    mask = torch.from_numpy(np.concatenate((up, frame, down), axis=0)).float()
    masked = spec * mask  
    return masked

#remove a band in the vertical direction in the spectrogram
def time_mask(spec):
    fbank_size = np.shape(spec)
    rows , columns = fbank_size[0], fbank_size[1]
    #width of the band
    fact1 = np.random.randint(int(columns/100), int(columns/60))
    frame = np.zeros([rows, fact1])
    #position of the band on the x axis
    pos = random.randint(10, columns-fact1-1)
    left = np.ones([rows, pos-1])
    right = np.ones([rows, columns-(pos+fact1)+1])
    mask = torch.from_numpy(np.concatenate((left, frame, right), axis=1)).float()
    masked = spec * mask
    return masked

#randomly shift calls in the audio temporal signal
def manipulate(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def _rms_energy(x):
    return np.sqrt(np.mean(x**2))

#add noise to signal from a same size vector
def _add_noise(signal, noise_file_name, snr, sample_rate):
    """

    :param signal:
    :param noise_file_name:
    :param snr:
    :return:
    """
    # Open noise file
    if isinstance(noise_file_name, np.ndarray):
        noise = noise_file_name
    else:
        noise, fs_noise = librosa.load(noise_file_name, sample_rate)

    # Generate random section of masker
    if len(noise) < len(signal):
        dup_factor = len(signal) // len(noise) + 1
        noise = np.tile(noise, dup_factor)

    if len(noise) != len(signal):
        idx = np.random.randint(1, len(noise) - len(signal))
        noise = noise[idx:idx + len(signal)]

    # Compute energy of both signals
    N_dB = _rms_energy(noise)
    S_dB = _rms_energy(signal)

    # Rescale N
    N_new = S_dB - snr
    noise_scaled = 10 ** (N_new / 20) * noise / 10 ** (N_dB / 20)
    noisy = signal + noise_scaled

    return (noisy - noisy.mean()) / noisy.std()


#create a new signal of length = max_time
def time_elong(sr, audio, max_time=2):
    final_audio = np.zeros((1, sr*max_time))
    if len(audio) > sr*max_time:
        print('the new audio file has to be longer then the original')
    else:   
        dim = len(audio)
        #windowing to avoid aliasing
        audio  = audio*np.hanning(dim)
        blockl = np.random.randint(0, sr*max_time -dim-1)
        blockr = blockl + dim 
        left   = np.zeros((blockl))
        right  = np.zeros((sr*max_time - blockr))
        new    = np.concatenate((left, audio, right), axis=0)
    return librosa.to_mono(new)


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, name_classes='', nb_class_noise =1, augmentation=True, path_background="./noises"):
      self.dict_classes = self.load_data(path)
      self.dict_noise_calls = self.load_data(path)
      self.name_classes = name_classes
    #delete name of birds from the calls to add as noise
      self.create_noise_calls()
      self.nb_class_noise = nb_class_noise    

      self.augmentation       = augmentation
      self.path_background    = path_background
      if self.augmentation:
          self.dict_noises = self.load_data(path_background)

  
  def load_data(self, path, ext='wav'):
      dict_classes = collections.OrderedDict()
      for root, dirnames, filenames in os.walk(path):
          for filename in fnmatch.filter(filenames, '*' + ext):
              classe = root.split("/")[-1]
              if classe in dict_classes.keys():
                  dict_classes[classe].append(os.path.join(root, filename))
              else:
                  dict_classes[classe] = [os.path.join(root, filename)]
      if len(list(dict_classes.keys() )) == 0:
          print("** WARNING ** No data loaded from " + path)
      return dict_classes
  
   #remove names of the classes on which the network is training from the dataset
#used to add unknown calls as noise
  def create_noise_calls(self):
      for name in self.name_classes:
          del self.dict_noise_calls[name]


  def get_noise(self):
      classe_noise   = random.randint(0, len(list(self.dict_noises.keys()))-1)
      classe_noise   = list(self.dict_noises.keys())[classe_noise]
      #random natural noise augmentation
      filename_noise = self.dict_noises[classe_noise][random.randint(0, len(self.dict_noises[classe_noise])-1)]
      return filename_noise 
   
  def pitch_shift(self, audio, sampling_rate):
      #random pitch shifting
      step_pitch = random.uniform(-0.001, 0.001)
      augment_audio = librosa.effects.pitch_shift(audio, sampling_rate, n_steps=step_pitch)
      return augment_audio

  def time_stretch(self, audio):
      speed_factor = random.uniform(0.7, 1.3)
      return librosa.effects.time_stretch(audio, speed_factor)

  
#apply randomly at list 1 band on the spectrogram
  def spec_augmentation(self, spec):
      n = random.randint(0, 2)
      if n == 0:
          t = random.randint(0, 1)
          if t == 1:
              spec =  time_mask(spec)
          if t == 0:
              spec = freq_mask(spec)
      else:
         for ii in range(n):
             spec =  time_mask(spec)
             spec =  freq_mask(spec)
      return spec


  def __len__(self):
      'Denotes the total number of samples'
#        return len(self.dict_classes)
      nb_samples = 150000
      return nb_samples
    

  def load_classes(self, classe_name):
      files = []
      'Load audio file'
      for cl in classe_name:
     #select a random file in the class
          idx = int(random.random() * len(self.dict_classes[cl]) )
          filename = self.dict_classes[cl][idx]
          files.append([cl, filename])
      return files


  def load_noise_files(self, nb_noise):
      files = []
      for cl in range(nb_noise):
          'Load audio file'
        #pick a class in the order of the dict
          rand_class = random.randint(0, len(self.dict_noise_calls)-1)
          classe_name = list( self.dict_noise_calls.keys())[rand_class]
        #select a random file in the class
          idx = int(random.random() * len( self.dict_noise_calls[classe_name]))
          filename =  self.dict_noise_calls[classe_name][idx]
          files.append([classe_name, filename])
      return files
  
  
  '[class_name, filename, [mask], [magnitude], [phase] ]'
  def __getitem__(self, index):
      'Load audio file for each classe'
      files = self.load_classes(self.name_classes)
      audio_mix = None
      #audio length (s)
      m_time = 3
      # Creation of the noise #############################
      'add calls as noise from a random class'
      classes_noise = self.load_noise_files(self.nb_class_noise)
      for fn in classes_noise:
          audio_raw, sr = load_file(fn[1])
          new           = time_elong(sr, audio_raw, m_time)
          audio_noise   = filt(new, sr)
          audio_noise   += audio_noise

      'Augment Data'
      if self.augmentation:
          if random.randint(0, 1) == 1:
            #natural noise
              n_noise = self.get_noise()
              snr = np.random.randint(-10, 0)
          else:
            #gaussian noise
              n_noise = np.random.normal(loc=0, scale=1, size=(1, m_time*sr))
              n_noise = librosa.to_mono(n_noise)
              snr = np.random.randint(40, 50)  #-10/5 for natural noise, 
      final_audio_noise     = _add_noise(audio_noise, n_noise, snr, sr)
################## Creation of the audio #############################
      for f in files:
          audio_raw, sr   = load_file(f[1])
          audio_raw       = self.pitch_shift(audio_raw, sr)
          audio_raw       = self.time_stretch(audio_raw)
          new             = time_elong(sr, audio_raw, max_time=m_time)
          clean_audio     = filt(new, sr)
          f.append(clean_audio-np.mean(clean_audio)) 
          '[class_name, filename, [clean_audio]]'
          if audio_mix is None:
              audio_mix = clean_audio
          else:
              audio_mix += clean_audio
######################################################################
      'final audio'
      audio_mix           = audio_mix + final_audio_noise               
      return [audio_mix, files]
