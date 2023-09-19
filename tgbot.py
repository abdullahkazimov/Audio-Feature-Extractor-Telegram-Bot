from telegram import Update
from telegram.ext import Updater, CallbackContext, MessageHandler, Filters
from pydub import AudioSegment
import torchaudio
import torch
import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np

matplotlib.use('Agg')

sample_rate=44100

def get_voice(update: Update, context: CallbackContext) -> None:
    new_file = context.bot.get_file(update.message.voice.file_id)
    new_file.download("voice_note.ogg")
    update.message.reply_text('Voice note saved')

    # Load the OGG file
    ogg_file = "voice_note.ogg"
    audio = AudioSegment.from_file(ogg_file, format="ogg")

    # Convert to MP3 format
    wav_file = "output_file.wav"
    audio.export(wav_file, format="wav")
    
    audio_file_path = wav_file
    
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        features = {}
        
        num_channels = waveform.shape[0]
        num_frames = waveform.shape[1]
        duration = num_frames / sample_rate
        
        features['num_channels'] = num_channels
        features['num_frames'] = num_frames
        features['duration'] = duration
        
        response_text = f"Number of Channels: {features['num_channels']}\n"
        response_text += f"Number of Frames: {features['num_frames']}\n"
        response_text += f"Duration: {features['duration']} seconds\n"
    
        update.message.reply_text(response_text)    

        # Waveform
        features['waveform'] = waveform[0].detach().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(features['waveform'], linewidth=1)
        plt.grid(True)
        plt.title("Audio Waveform")
        plt.savefig('waveform.png')
        update.message.reply_photo(photo=open('waveform.png', 'rb'))
        
        # Calculate MFCCs
        n_mfcc = 10  # Number of MFCC coefficients
        n_fft = 20  # You can adjust this based on your audio data
        hop_length = 10  # You can adjust this based on your audio data

        # Calculate the maximum possible padding
        max_padding = n_fft - hop_length

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': max_padding, 'hop_length': hop_length}
        )

        mfcc = mfcc_transform(waveform)

        # Pitch
        n_steps = 2  # Number of semitones to shift the pitch (adjust as needed)
        pitch_transform = torchaudio.transforms.PitchShift(sample_rate=sample_rate, n_steps=n_steps)
        pitched_waveform = pitch_transform(waveform)
        
        plt.figure(figsize=(10, 4))
        plt.plot(pitched_waveform[0].detach().numpy())
        plt.title("Pitch-shifted Audio")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.savefig('pitched_waveform.png')
        update.message.reply_photo(photo=open('pitched_waveform.png', 'rb'))
        
        # Spectrogram
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length
        )
        spectrogram = spectrogram_transform(waveform)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(librosa.power_to_db(spectrogram[0].detach().numpy()), cmap='viridis', origin='lower', aspect='auto',
                   extent=[0, len(features['waveform']) / sample_rate, 0, sample_rate / 2])
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig('spectrogram.png')
        update.message.reply_photo(photo=open('spectrogram.png', 'rb'))
        
        # Mel Spectrogram
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length
        )(waveform)

        # Populate the feature dictionary
        features['pitched_waveform'] = pitched_waveform[0].detach().numpy()
        features['mfcc'] = mfcc.detach().numpy()
        features['spectrogram'] = librosa.power_to_db(spectrogram[0].detach().numpy())
        features['mel_spectrogram'] = librosa.power_to_db(mel_spectrogram_transform[0].detach().numpy())

    except Exception as e:
        print(f"error: {e}")

API_TOKEN = "6440467272:AAEiezF8IEtGDsGu_E0ZbY59XQZ16hdA-kw"
updater = Updater(API_TOKEN)

updater.dispatcher.add_handler(MessageHandler(Filters.voice , get_voice))

updater.start_polling()
updater.idle()