from os import path
from audio2numpy import open_audio
from joblib import Parallel, cpu_count, delayed
from multiprocessing import Manager
import matplotlib.pyplot as plt
import numpy as np
import cv2

channel = "avarage"
framerate = 30
duration = framerate/1000
bins_amount = 64
xlog = 0
ylog = 0
DEFAULT_CHUNKSIZE = 128
VID_CODEC = 'mp4v'
VID_EXT = '.mp4'

def load_audio(audio_file):
    if not path.exists(audio_file):
        raise FileNotFoundError("Audio file does not exist")
    else:
        waveform, sample_rate = open_audio(audio_file)
        print("waveform: " + str(waveform))
        print("sample_rate: " + str(sample_rate))
        return waveform, sample_rate

def calculate_frameData(fileData, samplerate):
    channel = "avarage"
    framerate = 30
    duration = 1000/framerate
    start = 0
    end = len(fileData)/samplerate

    channels = []

    if len(fileData.shape) > 1:
        if channel == "avarage":
            channels.append(np.mean(fileData, axis=1))
        elif channel == "left":
            channels.append(fileData[:,0])
        elif channel == "right":
            channels.append(fileData[:,1])
        else:
            for i in range(fileData.shape[1]):
                channels.append(fileData[:,i])
    else:
        channels.append(fileData)

    print("channels: " + str(channels))
    
    frameData = []
    a = 0
    for channel in channels:
        if a == 0:
            print("channel: " + str(channel))

        channelData = channel[int(start*samplerate):int(end*samplerate)]

        if a == 0:
            print("channelData: " + str(channelData))

        channelFrameData = []
        stepSize = samplerate/framerate
        print("samplerate: " + str(samplerate))
        print("framerate: " + str(framerate))
        print("stepSize: " + str(stepSize))
        b = 0
        for i in range(int(np.ceil(len(channelData)/stepSize))):
            frameDataMidpoint = stepSize*i + (stepSize/2)
            frameDataStart = int(frameDataMidpoint - (duration/1000/2)*samplerate)
            frameDataEnd = int(frameDataMidpoint + (duration/1000/2)*samplerate)

            if a == 0 and b == 0:
                print("frameDataMidpoint: " + str(frameDataMidpoint))
                print("duration: " + str(duration))
                print("frameDataStart: " + str(frameDataStart))
                print("frameDataEnd: " + str(frameDataEnd))

            if frameDataStart < 0:
                emptyFrame = np.zeros(int(duration/1000 * samplerate))
                currentFrameData = channelData[0:frameDataEnd]
                emptyFrame[0:len(currentFrameData)] = currentFrameData
                currentFrameData = emptyFrame
            elif frameDataEnd > len(channelData):
                emptyFrame = np.zeros(int(duration/1000 * samplerate))
                currentFrameData = channelData[frameDataStart:]
                emptyFrame[0:len(currentFrameData)] = currentFrameData
                currentFrameData = emptyFrame
            else:
                currentFrameData = channelData[int(frameDataStart):int(frameDataEnd)]
            
            if a == 0 and b == 0:
                print("currentFrameData: " + str(currentFrameData))

            frameDataAmplitudes = abs(np.fft.rfft(currentFrameData))

            frameDataAmplitudes = frameDataAmplitudes[int(0/(samplerate/2)*len(frameDataAmplitudes)):int((samplerate/2)/(samplerate/2)*len(frameDataAmplitudes))]

            channelFrameData.append(frameDataAmplitudes)
            #print("channelFrameData: " + str(channelFrameData[0]))
            b += 1

        frameData.append(channelFrameData)
        #print("frameData: " + str(frameData[0]))
        a += 1

    return frameData

def create_bins(frameData):
    #print("frameData: " + str(len(frameData)))
    bins = []
    a = 0
    for channel in frameData:
        if a == 0:
            print("channel: " + str(channel[0][0]))
        channelBins = []
        b = 0
        for data in channel:
            if a == 0 and b == 0:
                print("data: " + str(data[0]))
            frameBins = []
            c = 0
            for i in range(bins_amount):
                if xlog == 0:
                    dataStart = int(i * len(data) / bins_amount)
                    dataEnd = int((i + 1) * len(data) / bins_amount)
                else:
                    dataStart = int((i/bins_amount)**xlog * len(data))
                    dataEnd = int(((i+1)/bins_amount)**xlog * len(data))
                if dataStart == dataEnd:
                    dataEnd += 1
                frameBins.append(np.mean(data[dataStart:dataEnd]))
                #print("frameBins: " + str(frameBins[0]))
                if a == 0 and b == 0 and c == 0:
                    print("dataStart: " + str(dataStart))
                    print("dataEnd: " + str(dataEnd))
                    print("frameBins: " + str(frameBins))
                c += 1
            channelBins.append(frameBins)

            #print("channelBins: " + str(channelBins[0]))
            b += 1
        
        bins.append(channelBins)

        #print("bins: " + str(bins[0]))
        a += 1

    return bins

def render_save_frames(bins):
    ylog = 0
    processes = cpu_count()
    chunkSize = int(DEFAULT_CHUNKSIZE / processes)

    bins = bins/np.max(bins)

    if ylog != 0:
        div = np.log2(ylog + 1)
        bins = np.log2(ylog * np.array(bins) + 1) / div
    
    numChunks = int(np.ceil(bins.shape[1]/(processes * chunkSize))) * processes

    shMem = Manager().dict()
    shMem["framecount"] = 0
    Parallel(n_jobs=processes)(delayed(render_save_partial)(j, numChunks, bins, shMem) for j in range(processes))

    print_progress_bar(bins.shape[1], bins.shape[1])
    print()

def render_save_partial(partialCounter, numChunks, bins, shMem):
    imageSequence = False
    destination = "output"
    framerate = 30
    width = 1920
    height = 540
    processes = cpu_count()

    if imageSequence:
        vid = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*VID_CODEC)
        dest = destination+"/part"+str(partialCounter)+VID_EXT
        vid = cv2.VideoWriter(dest, fourcc, framerate, (width, height))

    chunksPerProcess = int(numChunks/processes)
    for i in range(chunksPerProcess):
        chunkCounter = partialCounter * chunksPerProcess + i
        render_save_chunk(chunkCounter, numChunks, bins, vid, shMem)
    
    if not imageSequence:
        vid.release()

def render_save_chunk(chunkCounter, numChunks, bins, vid, shMem):
    processes = cpu_count()
    chunkSize = int(DEFAULT_CHUNKSIZE / processes)
    imageSequence = False
    destination = "output"

    chunksPerProcess = int(numChunks/processes)
    finishedChunkSets = int(chunkCounter/chunksPerProcess)
    framePerProcess = int(bins.shape[1]/processes)
    currentChunkNumInNewSet = chunkCounter - finishedChunkSets * chunksPerProcess
    start = finishedChunkSets * framePerProcess + currentChunkNumInNewSet * chunkSize
    end = start + chunkSize

    if chunkCounter % chunksPerProcess == chunksPerProcess - 1:
        completeChunkSets = int(numChunks/processes) - 1
        fullSetChunks = completeChunkSets * processes
        fullSetFrames = fullSetChunks * chunkSize
        remainingFrames = bins.shape[1] - fullSetFrames
        remainingChunkSize = int(remainingFrames / processes)
        end = start + remainingChunkSize
    
    frames = render_chunk_frames(bins, start, end)
    for i in range(len(frames)):
        if imageSequence:
            plt.imsave(str(destination) + "/" + str(start + i) + ".png", frames[i], vmin=0, vmax=255, cmap="gray")
        else:
            vid.write(frames[i])
        shMem["framecount"] += 1
        print_progress_bar(shMem["framecount"], bins.shape[1])

def render_chunk_frames(bins, start, end):
    frames = []
    for j in range(start, end):
        frame = render_frame(bins, j)
        frames.append(frame)
    return frames

def render_frame(bins, j):
    if len(bins) == 1:
        bins = bins[0]
        frame = render_mono_channel(bins, j)
    if len(bins) == 2:
        frame = render_stereo_channel(bins, j)
    
    return frame

def render_mono_channel(bins, j):
    mirror = 0
    argheight = 540
    argwidth = 1920
    backgroundColor = hex2rgb("000000")

    if mirror == 0:
        height = argheight
    else:
        fullFrame = np.full((argheight, argwidth, 3), backgroundColor).astype(np.uint8)
        height = int(argheight/2)
    
    frame = np.full((height, argwidth, 3), backgroundColor).astype(np.uint8)

    return

def render_stereo_channel(bins, j):
    return

def hex2rgb(hex):
    hexLen = len(hex)
    color = []
    try:
        if hexLen == 6:
            for i in (0, 2, 4):
                color.append(int(hex[i:i+2], 16))
        elif hexLen == 3:
            for i in (0, 1, 2):
                color.append(int(hex[i:i+1]*2, 16))
        else:
            exit("Color is not a valid HEX code, eg. ff0000.")
        return color
    except:
        exit("Color is neither a named color (e.g. red) or a valid HEX code (eg. ff0000).")

def print_progress_bar(iteration, total, prefix = "Progress:", suffix = "Complete", decimals = 2, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total}) {suffix}', end = printEnd)
